#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:10:19 2026

@author: alexander
"""

from GibbsEnergyFunctionsDataClasses import GibbsEnergy, TdbFunctionData
from BlockParserClass import TDBBlockExtractor

import re
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import numpy as np




class TdbParser:
    """
    Парсер для формата TDB (Thermodynamic Database).
    
    Преобразует строку TDB в объекты TdbFunctionData.
    Каждая функция может иметь несколько температурных диапазонов,
    поэтому результат — словарь: имя функции -> список TdbFunctionData.
    """
    
    # Захардкоженный список всех имён функций, найденных в данных
    FUNCTION_NAMES = [
        "GHSERAL", "GHSERBB", "GHSERCC", "GHSERCO", "GHSERCR",
        "GHSERCU", "GHSERFE", "GHSERHF", "GHSERHH", "GHSERLA",
        "GHSERMN", "GHSERMO", "GHSERNN", "GHSERNB", "GHSERNI",
        "GHSEROO", "GHSERPD", "GHSERPP", "GHSERSI", "GHSERSS",
        "GHSERTA", "GHSERTI", "GHSERVV", "GHSERWW", "GHSERYY"
    ]
    
    # Соответствие удвоенных букв одинарным (соглашение TDB для однобуквенных элементов)
    _DOUBLED_ELEMENTS = {
        "BB": "B", "CC": "C", "NN": "N", "OO": "O",
        "PP": "P", "SS": "S", "VV": "V", "WW": "W", "YY": "Y", "HH": "H"
    }
    
    def __init__(self, tdb_string: str):
        """
        Parameters
        ----------
        tdb_string : str
            Строка в формате TDB для парсинга.
        """
        self.raw_data = tdb_string
        self.functions: Dict[str, List[TdbFunctionData]] = {
            name: [] for name in self.FUNCTION_NAMES
        }
    
    def parse(self) -> Dict[str, List[TdbFunctionData]]:
        """
        Парсить строку TDB и вернуть словарь функций.
        
        Returns
        -------
        Dict[str, List[TdbFunctionData]]
            Словарь: имя функции -> список TdbFunctionData
            (по одному объекту на каждый температурный диапазон).
        """
        # Разбить на блоки функций по ключевому слову FUNCTION
        function_blocks = re.split(r'\nFUNCTION\s+', '\n' + self.raw_data)
        
        for block in function_blocks:
            block = block.strip()
            if not block:
                continue
            
            lines = block.split('\n')
            func_name = lines[0].strip()
            
            if func_name not in self.FUNCTION_NAMES:
                continue
            
            try:
                self._parse_function_block(func_name, '\n'.join(lines[1:]))
            except Exception as e:
                print(f"Ошибка при парсинге функции {func_name}: {e}")
        
        return self.functions
    
    def _parse_function_block(self, name: str, block: str):
        """Парсить один блок функции (все её температурные диапазоны)."""
        
        # Разбить по границам диапазонов: ; Tmax Y или ; Tmax N
        parts = re.split(r';\s*([\d.]+)\s+([YN])\s*', block)
        
        prev_tmax = 273.00  # Стартовая температура по умолчанию
        elements = self._extract_elements(name)
        
        i = 0
        while i < len(parts) - 2:
            expr = parts[i].strip()
            tmax = float(parts[i + 1])
            is_last = parts[i + 2] == 'N'
            
            # Проверить, начинается ли выражение с явного Tmin
            tmin_match = re.match(r'^(\d+\.?\d*)\s+', expr)
            if tmin_match:
                tmin = float(tmin_match.group(1))
                expr = expr[tmin_match.end():]
            else:
                tmin = prev_tmax
            
            coeffs = self._parse_coefficients(expr)
            
            func_data = TdbFunctionData(
                name=name,
                t_min=tmin,
                t_max=tmax,
                elements=elements,
                **coeffs
            )
            
            self.functions[name].append(func_data)
            prev_tmax = tmax
            
            i += 3
            
            if is_last:
                break
    
    def _parse_coefficients(self, expr: str) -> Dict[str, float]:
        """
        Парсить полиномиальное выражение в словарь коэффициентов.
        
        Поддерживаемые термы:
        - Константа (b)
        - *T (t1)
        - *T*LN(T) (tlnt)
        - *T**2 (t2), *T**3 (t3), *T**4 (t4)
        - *T**(-1) (tm1), *T**(-2) (tm2), *T**(-3) (tm3)
        - *T**7 (t7), *T**(-9) (tm9)
        """
        coeffs = {
            'b': 0.0, 't1': 0.0, 'tlnt': 0.0, 't2': 0.0, 't3': 0.0,
            'tm1': 0.0, 't7': 0.0, 'tm9': 0.0, 't4': 0.0,
            'tm2': 0.0, 'tm3': 0.0
        }
        
        expr = expr.replace(' ', '')
        
        # Паттерн: число (со знаком, десятичное, научная нотация) + опциональная переменная часть
        term_pattern = r'([+-]?\d+\.?\d*(?:[Ee][+-]?\d+)?)(\*T(?:\*LN\(T\))?(?:\*\*[-\d()]+)?)?'
        
        for match in re.finditer(term_pattern, expr):
            coef_str = match.group(1)
            var_part = match.group(2) or ''
            
            try:
                coef = float(coef_str)
            except ValueError:
                continue
            
            mapping = {
                '': 'b',
                '*T': 't1',
                '*T*LN(T)': 'tlnt',
                '*T**2': 't2',
                '*T**3': 't3',
                '*T**4': 't4',
                '*T**(-1)': 'tm1',
                '*T**(-2)': 'tm2',
                '*T**(-3)': 'tm3',
                '*T**7': 't7',
                '*T**(-9)': 'tm9',
            }
            
            key = mapping.get(var_part)
            if key is not None:
                coeffs[key] += coef
        
        return coeffs
    
    def _extract_elements(self, func_name: str) -> List[str]:
        """Извлечь название элемента из имени функции GHSERxxx."""
        if not func_name.startswith("GHSER"):
            return []
        
        suffix = func_name[5:]
        
        if not suffix:
            return []
        
        # Обработка удвоенных букв (соглашение TDB)
        if suffix in self._DOUBLED_ELEMENTS:
            return [self._DOUBLED_ELEMENTS[suffix]]
        
        # Многобуквенные элементы: первая заглавная, остальные строчные
        if len(suffix) > 1:
            return [suffix[0].upper() + suffix[1:].lower()]
        
        return [suffix.upper()]
    
    
# Пример использования
if __name__ == '__main__':
    extractor = TDBBlockExtractor('databases/mc_fe_v2062_clean.tdb')
    
    KNOWN_BLOCKS = extractor.KNOWN_BLOCKS
    
    # Запрашиваем блок
    result = extractor.get_block('SER (Standard elements references)')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    