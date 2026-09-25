#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Парсер секций PHASE и CONSTITUENT из TDB-файла.

PHASE задаёт саму фазу: имя, магнитный символ (ссылка на TYPE_DEFINITION,
опционально), число подрешёток и их стехиометрические коэффициенты
(site ratios). После координат может идти произвольный многострочный
текст-описание, завершающийся '>> <приоритет> !' (приоритет может
отсутствовать).

    PHASE FCC_A1  %'  2 1   1 >
    Face-centered cubic Austenite phase with Va on interstitial sublattice;
    ...
    >> 6 !

CONSTITUENT задаёт допустимые виды (элементы/VA) в каждой подрешётке,
через ':' между подрешётками и ',' между видами внутри подрешётки; список
может переноситься на несколько строк, а суффикс '%' у вида означает
"вид по умолчанию" для данной подрешётки:

    CONSTITUENT FCC_A1  : AL,CO,CR,CU,FE%,HF,...,Y : B,C,H,N,O,VA% :  !
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PhaseDefinition:
    """Данные одной фазы (PHASE + CONSTITUENT)."""

    name: str
    magnetic_symbol: Optional[str]      # символ из TYPE_DEFINITION, либо None
    n_sublattices: int
    site_ratios: List[float]            # стехиометрия подрешёток, из PHASE
    sublattices: List[List[str]] = field(default_factory=list)  # из CONSTITUENT
    priority: Optional[int] = None      # число после '>>' (может отсутствовать)


class PhaseTdbParser:
    """
    Парсер строк PHASE и CONSTITUENT. Работает на всём тексте TDB-файла:
    сначала убирает строки-комментарии ('$...'), затем ищет все PHASE- и
    CONSTITUENT-выражения (каждое может занимать несколько строк) и
    объединяет их в единый словарь PhaseDefinition по имени фазы.
    """

    _PHASE_PATTERN = re.compile(
        r'\bPHASE\s+(?P<name>\S+)\s+'
        r'%(?P<symbol>[^\s\d])?\s*'
        r'(?P<n_sub>\d+)\s+'
        r'(?P<ratios>[\d.\s]+?)'
        r'(?:>(?P<description>.*?))?'   # '>' + описание есть не всегда (NI3SIL, MNB4 - без него)
        r'!',
        re.DOTALL,
    )

    # Ключевое слово в TDB можно сократить до уникального префикса: встречается
    # как полное CONSTITUENT, так и сокращённое CONST (например, у GAMMA_PRIME)
    _CONSTITUENT_PATTERN = re.compile(
        r'\bCONST(?:ITUENT)?\s+(?P<name>\S+)\s*'
        r':(?P<body>.*?)'
        r'!',
        re.DOTALL,
    )

    _PRIORITY_PATTERN = re.compile(r'>>\s*(\d+)\s*$')

    def __init__(self, tdb_string: str):
        """
        Parameters
        ----------
        tdb_string : str
            Текст TDB-файла (целиком).
        """
        self.raw_data = self._strip_comments(tdb_string)
        self.phases: Dict[str, PhaseDefinition] = {}

    @staticmethod
    def _strip_comments(text: str) -> str:
        """
        Убрать строки-комментарии (начинающиеся с '$'), заменив их пустой
        строкой (чтобы не сдвигать нумерацию и не склеить соседние
        многострочные выражения).
        """
        lines = text.split('\n')
        return '\n'.join(
            '' if line.strip().startswith('$') else line
            for line in lines
        )

    def parse(self) -> Dict[str, PhaseDefinition]:
        """
        Разобрать все PHASE и CONSTITUENT в тексте и связать их вместе.

        Returns
        -------
        Dict[str, PhaseDefinition]
            Словарь: имя фазы -> PhaseDefinition.
        """
        self._parse_phase_statements()
        self._parse_constituent_statements()
        return self.phases

    def _parse_phase_statements(self) -> None:
        for match in self._PHASE_PATTERN.finditer(self.raw_data):
            name = match.group('name').upper()
            symbol = match.group('symbol')
            n_sub = int(match.group('n_sub'))
            ratios = [float(x) for x in match.group('ratios').split()]

            if len(ratios) != n_sub:
                print(
                    f"[PHASE {name}] число site ratios ({len(ratios)}) "
                    f"не совпадает с заявленным числом подрешёток ({n_sub})"
                )

            description = match.group('description') or ''
            priority_match = self._PRIORITY_PATTERN.search(description.strip())
            priority = int(priority_match.group(1)) if priority_match else None

            self.phases[name] = PhaseDefinition(
                name=name,
                magnetic_symbol=symbol,
                n_sublattices=n_sub,
                site_ratios=ratios,
                priority=priority,
            )

    def _parse_constituent_statements(self) -> None:
        for match in self._CONSTITUENT_PATTERN.finditer(self.raw_data):
            name = match.group('name').upper()
            body = match.group('body')

            # Схлопнуть переносы строк и лишние пробелы внутри списка видов
            body = ' '.join(body.split())

            groups = body.split(':')
            # Последняя группа — завершающий пустой маркер (перед '!'), отбрасываем
            if groups and groups[-1].strip() == '':
                groups = groups[:-1]

            sublattices: List[List[str]] = []
            for group in groups:
                species = [
                    token.strip().rstrip('%').upper()
                    for token in group.split(',')
                    if token.strip()
                ]
                sublattices.append(species)

            if name not in self.phases:
                print(f"[CONSTITUENT {name}] фаза не найдена среди PHASE-определений")
                continue

            phase = self.phases[name]
            if len(sublattices) != phase.n_sublattices:
                print(
                    f"[CONSTITUENT {name}] число подрешёток ({len(sublattices)}) "
                    f"не совпадает с PHASE ({phase.n_sublattices})"
                )

            phase.sublattices = sublattices

    def get_phase(self, name: str) -> Optional[PhaseDefinition]:
        """Получить PhaseDefinition по имени (регистронезависимо)."""
        return self.phases.get(name.upper())


# Пример использования
if __name__ == '__main__':
    from TypeDefinitionParserClass import TypeDefinitionTdbParser

    with open('databases/mc_fe_v2062_clean.tdb', 'r', encoding='utf-8') as f:
        tdb_text = f.read()

    type_defs = TypeDefinitionTdbParser(tdb_text).parse()

    parser = PhaseTdbParser(tdb_text)
    phases = parser.parse()

    print(f"Найдено фаз: {len(phases)}")

    for phase_name in ['FCC_A1', 'LIQUID', 'GRAPHITE', 'ETA', 'CHI_A12']:
        phase = parser.get_phase(phase_name)
        print(f"\n{phase_name}:")
        print(f"  подрешёток: {phase.n_sublattices}, site_ratios: {phase.site_ratios}")
        print(f"  подрешётки (виды): {phase.sublattices}")
        print(f"  приоритет: {phase.priority}, магнитный символ: {phase.magnetic_symbol!r}")

        if phase.magnetic_symbol:
            td = type_defs.get(phase.magnetic_symbol)
            if td:
                print(f"  -> магнитная модель: p={td.structure_factor}, "
                      f"factor={td.magnetic_factor}")

    # Проверка: сколько фаз осталось без sublattices (не нашли CONSTITUENT)
    missing = [p.name for p in phases.values() if not p.sublattices]
    print(f"\nФазы без CONSTITUENT: {missing}")
