#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:10:19 2026

@author: alexander
"""

from GibbsEnergyFunctionsDataClasses import GibbsEnergy, TdbFunctionData, TdbFunctionManipulation
from BlockParserClass import TDBBlockExtractor

import re
from typing import Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np




class TdbParser:
    
    
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
        function_blocks = re.split(r'\n[ \t]*FUNCTION\s+', '\n' + self.raw_data)
        
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
        
        # 1. Проверка на извлечение элементов
        try:
            _ = self._DOUBLED_ELEMENTS 
            elements = self._extract_elements(name)
        except AttributeError:
            elements = []
                
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
            
            # 2. Проверка на поиск зависимостей
            try:
                _ = self.DEPENDENCIES  # Без подчеркивания, как в вашем классе
                dep_names, dep_coefs = self._parse_dependencies(expr)
                
            except AttributeError:
                dep_names, dep_coefs = [], []
                
            
            func_data = TdbFunctionData(
                name=name,
                pure_text= expr,
                t_min=tmin,
                t_max=tmax,
                elements=elements,
                dependencies=dep_names,      # Передаем зависимости
                ref_coef=dep_coefs,          # Передаем коэффициенты
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
        """
        coeffs = {
            'b': 0.0, 't1': 0.0, 'tlnt': 0.0, 't2': 0.0, 't3': 0.0,
            'tm1': 0.0, 't7': 0.0, 'tm9': 0.0, 't4': 0.0,
            'tm2': 0.0, 'tm3': 0.0
        }
        
        # Сначала удаляем все ссылки на функции (зависимости) из выражения
        try:
            _ = self.DEPENDENCIES
            for func_name in self.DEPENDENCIES:
                # Паттерн для поиска зависимостей: число (опционально) * имя_функции#
                pattern = rf'[+-]?\d*\.?\d*(?:[Ee][+-]?\d+)?\*?{re.escape(func_name)}#'
                expr = re.sub(pattern, '', expr)
        except AttributeError:
            pass
        
        expr = expr.replace(' ', '')
        
        # Число: принимает и .123, и -.123, и обычную запись
        number_pattern = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?'
        
        # Переменная часть: *T, *T*LN(T), *T**N, *T**(N)
        var_pattern = r'\*T(?:\*LN\(T\))?(?:\*\*(?:\([^)]+\)|-?\d+))?'
        
        term_pattern = f'({number_pattern})({var_pattern})?'
        
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
    
    




    
class SerTdbParser(TdbParser):
    """
    Парсер для SER-функций (Standard Element References).

    Определяет конкретный набор функций и правила
    для удвоенных букв однобуквенных элементов.
    """

    FUNCTION_NAMES = [
        "GHSERAL", "GHSERBB", "GHSERCC", "GHSERCO", "GHSERCR",
        "GHSERCU", "GHSERFE", "GHSERHF", "GHSERHH", "GHSERLA",
        "GHSERMN", "GHSERMO", "GHSERNN", "GHSERNB", "GHSERNI",
        "GHSEROO", "GHSERPD", "GHSERPP", "GHSERSI", "GHSERSS",
        "GHSERTA", "GHSERTI", "GHSERVV", "GHSERWW", "GHSERYY"
    ]
    
    _DOUBLED_ELEMENTS = {
        "BB": "B",  "CC": "C",  "HH": "H",  "NN": "N",  "OO": "O",
        "PP": "P",  "SS": "S",  "VV": "V",  "WW": "W",  "YY": "Y",
        }


class OtherThanSerTdbParser(TdbParser):
    """
    Парсер для SER-функций (Standard Element References).

    Определяет конкретный набор функций и правила
    для удвоенных букв однобуквенных элементов.
    """

    FUNCTION_NAMES = [
    "GHSERNIA", "GHSERTIA", "GDHCNI", "GDHCTI", "GDHCNIA", "GDHCTIA", 
    "GTI2NI", "DGLAV", "GLAVNI", "GLAVTI", "GTIFCCA", "GNIHCPA", 
    "GALBCC", "GCOBCC", "GCUBCC", "GHFBCC", "GLABCC", "GMNBCC", 
    "GNIBCC", "GPBCC", "GPDBCC", "GSIBCC", "GTIBCC", "GYYBCC", 
    "GCOFCC", "GCRFCC", "GFEFCC", "GLAFCC", "GMNFCC", "GMOFCC", 
    "GPFCC", "GSIFCC", "GTAFCC", "GTIFCC", "GWFCC", "GALHCP", 
    "GFEHCP", "GMOHCP", "GNIHCP", "GHEXTNB", "GTAHCP", "GPRED", 
    "GFECEM", "GCOM23C6", "GCRM7C3", "GCRM3C2", "ETCFESI", "GTIN", "GSSLIQ", 
    "GDHCFE", "GDIG", "GTIC", "B2ALNI", "LB2ALNI", "UALNI", 
    "U1ALNI", "U3ALNI", "U4ALNI", "L04ALNI", "L14ALNI", "ALNI3", 
    "AL2NI2", "AL3NI", "FESIW1", "LFESIB0", "LFESIB1", "LFESIB2", 
    "B2NIVA", "LB2NIVA", "LF0", "LF1", "GMNNI3", "GMN2NI2", 
    "GMN3NI", "LRMNNI", "GAL2O3", "GCR2O3", "GCR3O4", "GFEO", 
    "GFE2O3", "GHF1O2_S", "GLA2O3D", "GLA2O3X", "GLA2O3H", "GNIO", 
    "GSIO2", "GTI2O3", "GTRID", "GY2O3C", "GY2O3H", "UNDEF"
    ]
    
    DEPENDENCIES = [
    "DGLAV", "GCR2O3", "GFECEM", "GFEFCC", "GFEHCP", "GHSERAL", 
    "GHSERCC", "GHSERCO", "GHSERCR", "GHSERCU", "GHSERFE", "GHSERMN", 
    "GHSERMO", "GHSERNI", "GHSERNIA", "GHSEROO", "GHSERPD", "GHSERSI", 
    "GHSERSS", "GHSERTA", "GHSERTI", "GHSERTIA", "GHSERWW", "GHSERYY", 
    "GLA2O3D", "GNIHCP", "GNIHCPA", "GTIFCC", "GTIFCCA", "UALNI", 
    "U1ALNI", "U3ALNI", "U4ALNI"
    ]
    
    def _parse_dependencies(self, expr: str) -> Tuple[List[str], List[float]]:
        """
        Найти в выражении ссылки на функции из DEPENDENCIES и извлечь их коэффициенты.
        
        Parameters
        ----------
        expr : str
            Строка с математическим выражением.
        
        Returns
        -------
        Tuple[List[str], List[float]]
            Кортеж из двух списков:
            - Список имен функций, на которые есть ссылки
            - Список соответствующих коэффициентов
        """
        deps = {}
        
        for func_name in self.DEPENDENCIES:
            # Паттерн для поиска функции с необязательным коэффициентом
            # Примеры: "0.5*GNIHCP#", "+GNIHCP#", "-GNIHCP#", "GNIHCP#"
            pattern = rf'([+-]?\d*\.?\d*(?:[Ee][+-]?\d+)?)?\*?{re.escape(func_name)}#'
            
            for match in re.finditer(pattern, expr):
                coef_str = match.group(1)
                
                # Определяем коэффициент
                if coef_str is None or coef_str == '':
                    coef = 1.0
                elif coef_str == '+' or coef_str == '-':
                    coef = 1.0 if coef_str == '+' else -1.0
                else:
                    try:
                        coef = float(coef_str)
                    except ValueError:
                        coef = 1.0
                
                # Если функция уже встречалась, суммируем коэффициенты
                if func_name in deps:
                    deps[func_name] += coef
                else:
                    deps[func_name] = coef
        
        # Преобразуем словарь в два списка
        dep_names = list(deps.keys())
        dep_coefs = list(deps.values())
        
        return dep_names, dep_coefs
    
   
    

    
    
#%%    
        
    
# Пример использования
if __name__ == '__main__':
    extractor = TDBBlockExtractor('databases/mc_fe_v2062_clean.tdb')
    
    #KNOWN_BLOCKS = extractor.KNOWN_BLOCKS
    
    # Запрашиваем блок
    ser = extractor.get_block('SER (Standard elements references)')
    other_than_ser = extractor.get_block('Gibbs energy functions other than SER')
    
    # Создание парсера и запуск парсинга для функций блока SER 
    ser_values, = ser.values()
    
    parser = SerTdbParser(ser_values)
    
    ser_fuctions = parser.parse()
    
    # Создание парсера и запуск парсинга для функций блока other than SER
    other_than_ser_values, =  other_than_ser.values()
    
    parser = OtherThanSerTdbParser(other_than_ser_values)
    other_than_ser_functions = parser.parse()
  
    
    
    
    # Примеры отдельных функций
    # Назначаем интересующую нас темперутуру
    T = 5000.
    # Назначаем интересующую нас функцию
    function_name = 'GTI2NI' 
    # назначаем блок где будем функцию искать
    block = other_than_ser_functions
    
    
    # Находим функцию для интересующей температуры
    GTI2NI = TdbFunctionManipulation.get_function_from_dict(function_name, T, block)
    
        
    # смотрим, есть ли зависимости
    dependencies = GTI2NI.dependencies
    
    # создаем переменные для вспомогательных функций
    GHSERNIA = TdbFunctionManipulation.get_function_from_dict(dependencies[0], T, block)
    GHSERTIA = TdbFunctionManipulation.get_function_from_dict(dependencies[1], T, block)
    
    # Расчитываем энергию Гиббса
    # Сначала для вспомогательных функций
    GHSERNIA.calculateGibbsEnergy (T)
    GHSERTIA.calculateGibbsEnergy (T)
    
    # затем для основной
    GTI2NI.calculateGibbsEnergy (T, [GHSERNIA.G, GHSERTIA.G])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    