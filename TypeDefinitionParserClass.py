#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Парсер секции TYPE_DEFINITION из TDB-файла.

Формат строки в TDB:
    TYPE_DEFINITION ' GES A_P_D FCC_A1 MAGNETIC  -3.0    0.28 !
    TYPE_DEFINITION & GES A_P_D BCC_A2 MAGNETIC  -1.0    0.4  !

TYPE_DEFINITION присваивает однобуквенный символ дополнительной физической
модели (в этой базе встречается только MAGNETIC — модель магнитного вклада
Индена/Хиллерта-Ярла), привязанной к конкретной фазе. Этот символ потом
используется в строке PHASE (после '%'), чтобы отметить, что для данной
фазы нужно считать соответствующий дополнительный вклад в энергию Гиббса.

Пример связки:
    TYPE_DEFINITION ' GES A_P_D FCC_A1 MAGNETIC  -3.0    0.28 !
    PHASE FCC_A1  %'  2 1   1 >
                  ^ этот символ ссылается на TYPE_DEFINITION выше
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TypeDefinition:
    """
    Данные одного TYPE_DEFINITION.

    Parameters
    ----------
    symbol : str
        Однобуквенный символ (например, "'", "&", ")"), которым эта
        модель отмечается в строке PHASE после '%'.
    phase : str
        Имя фазы, к которой относится модель.
    model_name : str
        Имя модели (в этой базе — всегда 'MAGNETIC').
    parameters : List[float]
        Числовые константы модели. Для MAGNETIC: [structure_factor, factor].
    """

    symbol: str
    phase: str
    model_name: str
    parameters: List[float] = field(default_factory=list)

    @property
    def structure_factor(self) -> float:
        """p — структурный фактор модели Индена (актуально для MAGNETIC)."""
        return self.parameters[0]

    @property
    def magnetic_factor(self) -> float:
        """Нормировочный коэффициент магнитной энтропии (актуально для MAGNETIC)."""
        return self.parameters[1]


class TypeDefinitionTdbParser:
    """
    Парсер строк TYPE_DEFINITION. Работает на всём тексте TDB-файла —
    находит и разбирает все строки, начинающиеся с ключевого слова
    TYPE_DEFINITION.
    """

    _LINE_PATTERN = re.compile(
        r'^\s*TYPE_DEFINITION\s+'
        r'(?P<symbol>\S)\s+'
        r'GES\s+A_P_D\s+'
        r'(?P<phase>\S+)\s+'
        r'(?P<model>[A-Z_]+)\s+'
        r'(?P<params>[^!]*)'
        r'!'
    )

    def __init__(self, tdb_string: str):
        """
        Parameters
        ----------
        tdb_string : str
            Текст TDB-файла (целиком или его часть, содержащая TYPE_DEFINITION).
        """
        self.raw_data = tdb_string
        self.type_definitions: Dict[str, TypeDefinition] = {}

    def parse(self) -> Dict[str, TypeDefinition]:
        """
        Найти и разобрать все строки TYPE_DEFINITION в тексте.

        Returns
        -------
        Dict[str, TypeDefinition]
            Словарь: однобуквенный символ -> TypeDefinition.
        """
        for line in self.raw_data.split('\n'):
            line = line.strip()

            if not line.upper().startswith('TYPE_DEFINITION'):
                continue

            match = self._LINE_PATTERN.match(line)
            if match is None:
                print(f"Не удалось распарсить строку TYPE_DEFINITION: '{line}'")
                continue

            params_str = match.group('params').strip()
            parameters = [float(x) for x in params_str.split()] if params_str else []

            type_def = TypeDefinition(
                symbol=match.group('symbol'),
                phase=match.group('phase').upper(),
                model_name=match.group('model').upper(),
                parameters=parameters,
            )

            self.type_definitions[type_def.symbol] = type_def

        return self.type_definitions

    def get_by_phase(self, phase: str) -> List[TypeDefinition]:
        """Все TYPE_DEFINITION, привязанные к данной фазе (обычно 0 или 1)."""
        phase = phase.upper()
        return [td for td in self.type_definitions.values() if td.phase == phase]

    def get_magnetic_definitions(self) -> Dict[str, TypeDefinition]:
        """Только определения модели MAGNETIC (единственный тип в этой базе)."""
        return {
            symbol: td
            for symbol, td in self.type_definitions.items()
            if td.model_name == 'MAGNETIC'
        }


# Пример использования
if __name__ == '__main__':
    with open('databases/mc_fe_v2062_clean.tdb', 'r', encoding='utf-8') as f:
        tdb_text = f.read()

    parser = TypeDefinitionTdbParser(tdb_text)
    type_defs = parser.parse()

    print(f"Найдено TYPE_DEFINITION: {len(type_defs)}")
    for symbol, td in type_defs.items():
        print(f"  '{symbol}' -> фаза {td.phase}, модель {td.model_name}, "
              f"p={td.structure_factor}, factor={td.magnetic_factor}")

    # Пример: найти определение для конкретной фазы
    fcc_defs = parser.get_by_phase('FCC_A1')
    print("\nTYPE_DEFINITION для FCC_A1:", fcc_defs)
    
    bcc_defs = parser.get_by_phase('BCC_A2')
    print("\nTYPE_DEFINITION для BCC_A2:", bcc_defs)
    
    
    
    
