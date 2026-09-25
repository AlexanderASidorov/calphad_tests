#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:10:29 2026

@author: alexander
"""

from GibbsEnergyFunctionsDataClasses import GibbsEnergy, TdbFunctionData, TdbFunctionManipulation
from BlockParserClass import TDBBlockExtractor
from FunctionParserClass import TdbParser, SerTdbParser, OtherThanSerTdbParser

import re
from typing import Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np




    
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
    
    ser_functions = parser.parse()
    
    # Создание парсера и запуск парсинга для функций блока other than SER
    other_than_ser_values, =  other_than_ser.values()
    
    parser = OtherThanSerTdbParser(other_than_ser_values)
    other_than_ser_functions = parser.parse()
  
    
    
    
    # Примеры отдельных функций
    # Назначаем интересующую нас темперутуру
    T = 800.
    # Назначаем интересующую нас функцию элементов Fe и C
    function_names = ['GHSERFE', 'GHSERCC'] 
    # назначаем блок где будем функцию искать
    block = ser_functions
    
    # Находим функции для интересующей температуры
    GHSERFE = TdbFunctionManipulation.get_function_from_dict(function_names[0], T, block)
    GHSERCC = TdbFunctionManipulation.get_function_from_dict(function_names[1], T, block)
    
    
    # назначаем интересующую нас функцию из блока other then ser (нам нужен цементит)
    function_names = ['GFECEM']
    # назначаем блок где будем функцию искать
    block = other_than_ser_functions
    # Находим функцию для интересующей температуры
    GFECEM = TdbFunctionManipulation.get_function_from_dict(function_names[0], T, block)
    
    
    
    
    
    
  