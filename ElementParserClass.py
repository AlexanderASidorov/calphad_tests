#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Парсер секции ELEMENT из TDB-файла.

Формат строки в TDB:
    ELEMENT FE   BCC_A2           55.847           4489.0            27.2797    !
    ELEMENT VA   VACUUM            0.0                0.00            0.00      !
    ELEMENT  H   1/2_MOLE_H2(GAS)  1.0079          4234.0            65.285     !

Поля: символ, референсное (стандартное) состояние, молярная масса [г/моль],
энтальпия H298-H0 [Дж/моль], энтропия S298 [Дж/(моль*К)].
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ElementData:
    """Данные одного элемента из секции ELEMENT."""

    symbol: str             # 'FE', 'VA', 'C', ...
    reference_phase: str    # 'BCC_A2', 'VACUUM', '1/2_MOLE_H2(GAS)', ...
    molar_mass: float       # г/моль
    h298: float             # H298 - H0, Дж/моль (стандартная энтальпия элемента, SER)
    s298: float             # S298, Дж/(моль*К) (стандартная энтропия элемента, SER)


class ElementTdbParser:
    """
    Парсер строк ELEMENT. Работает как на всём тексте TDB-файла,
    так и на отдельно извлечённом блоке (например, через TDBBlockExtractor,
    если добавить 'A) Definition of elements' в KNOWN_BLOCKS) —
    в любом случае просто находит и разбирает все строки, начинающиеся
    с ключевого слова ELEMENT.
    """

    # Символ, референсное состояние (без пробелов) и три числа перед '!'
    _LINE_PATTERN = re.compile(
        r'^\s*ELEMENT\s+'
        r'(?P<symbol>\S+)\s+'
        r'(?P<ref_phase>\S+)\s+'
        r'(?P<mass>[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s+'
        r'(?P<h298>[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s+'
        r'(?P<s298>[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?)\s*'
        r'!'
    )

    def __init__(self, tdb_string: str):
        """
        Parameters
        ----------
        tdb_string : str
            Текст TDB-файла (целиком или его часть, содержащая ELEMENT).
        """
        self.raw_data = tdb_string
        self.elements: Dict[str, ElementData] = {}

    def parse(self) -> Dict[str, ElementData]:
        """
        Найти и разобрать все строки ELEMENT в тексте.

        Returns
        -------
        Dict[str, ElementData]
            Словарь: символ элемента (в верхнем регистре) -> ElementData.
        """
        for line in self.raw_data.split('\n'):
            line = line.strip()

            if not line.upper().startswith('ELEMENT'):
                continue

            match = self._LINE_PATTERN.match(line)
            if match is None:
                print(f"Не удалось распарсить строку ELEMENT: '{line}'")
                continue

            symbol = match.group('symbol').upper()

            self.elements[symbol] = ElementData(
                symbol=symbol,
                reference_phase=match.group('ref_phase').upper(),
                molar_mass=float(match.group('mass')),
                h298=float(match.group('h298')),
                s298=float(match.group('s298')),
            )

        return self.elements

    def get_real_elements(self) -> List[str]:
        """
        Список символов элементов без вакансии VA
        (VA — не химический элемент, а служебный вид для незаполненных узлов).
        """
        return [sym for sym in self.elements if sym != 'VA']

    def get_element(self, symbol: str) -> Optional[ElementData]:
        """Получить ElementData по символу (регистронезависимо)."""
        return self.elements.get(symbol.upper())


# Пример использования
if __name__ == '__main__':
    with open('databases/mc_fe_v2062_clean.tdb', 'r', encoding='utf-8') as f:
        tdb_text = f.read()

    parser = ElementTdbParser(tdb_text)
    elements = parser.parse()

    #print(f"Найдено элементов: {len(elements)}")
    #print(f"Химические элементы (без VA): {parser.get_real_elements()}")

    Ti = parser.get_element('TI')
    #print(fe)
    # ElementData(symbol='FE', reference_phase='BCC_A2', molar_mass=55.847,
    #             h298=4489.0, s298=27.2797)
