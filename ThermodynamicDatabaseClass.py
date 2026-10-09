#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThermodynamicDatabase — единая точка доступа ко всем данным TDB-файла.

Собирает результаты всех парсеров:
    - ElementTdbParser         -> self.elements
    - TypeDefinitionTdbParser  -> self.type_definitions
    - PhaseTdbParser           -> self.phases
    - SerTdbParser / OtherThanSerTdbParser -> self.functions (объединённый словарь)
    - ParameterTdbParser       -> self.parameters

и умеет численно вычислять:
    - любую именованную функцию (GHSERFE, GFEFCC, ...) при заданной T,
      рекурсивно разворачивая ссылки '...#' на другие функции;
    - любой параметр фазы (G, L, TC, BMAGN) при заданной T.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from BlockParserClass import TDBBlockExtractor
from FunctionParserClass import OtherThanSerTdbParser
from GibbsEnergyFunctionsDataClasses import TdbFunctionData, TdbFunctionManipulation
from ElementParserClass import ElementTdbParser, ElementData
from TypeDefinitionParserClass import TypeDefinitionTdbParser, TypeDefinition
from PhaseParserClass import PhaseTdbParser, PhaseDefinition
from ParameterParserClass import ParameterTdbParser, TdbParameterData


class AutoFunctionTdbParser(OtherThanSerTdbParser):
    """
    Парсер FUNCTION-блоков, у которого список имён функций и список возможных
    зависимостей берутся не из захардкоженных списков, а из самого текста TDB
    (все имена после ключевого слова FUNCTION). Это нужно, потому что в файле
    есть функции (GHFFCC, GCRM23C6, ...), которых нет в FUNCTION_NAMES.
    """

    def __init__(self, tdb_string: str, all_names: List[str]):
        # длинные имена вперёд, чтобы короткое имя не "съело" часть длинного
        names = sorted(set(all_names), key=len, reverse=True)
        self.FUNCTION_NAMES = names
        self.DEPENDENCIES = names
        super().__init__(tdb_string)


# Ключ для поиска параметра: (тип, подрешётки, порядок RK)
ParamKey = Tuple[str, Tuple[Tuple[str, ...], ...], int]


class ThermodynamicDatabase:

    SER_BLOCK = 'SER (Standard elements references)'
    OTHER_BLOCK = 'Gibbs energy functions other than SER'

    def __init__(self, tdb_path: str):
        self.tdb_path = tdb_path

        self.elements: Dict[str, ElementData] = {}
        self.type_definitions: Dict[str, TypeDefinition] = {}
        self.phases: Dict[str, PhaseDefinition] = {}
        self.functions: Dict[str, List[TdbFunctionData]] = {}

        # phase -> param_type -> список TdbParameterData (по T-диапазонам)
        self.parameters: Dict[str, Dict[str, List[TdbParameterData]]] = {}
        # быстрый поиск: phase -> (тип, подрешётки, order) -> список T-диапазонов
        self._param_index: Dict[str, Dict[ParamKey, List[TdbParameterData]]] = {}

    # ------------------------------------------------------------------
    # Загрузка
    # ------------------------------------------------------------------
    def load(self) -> 'ThermodynamicDatabase':
        with open(self.tdb_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.elements = ElementTdbParser(text).parse()
        self.type_definitions = TypeDefinitionTdbParser(text).parse()
        self.phases = PhaseTdbParser(text).parse()

        extractor = TDBBlockExtractor(self.tdb_path)
        self._load_functions(extractor)
        self._load_parameters(extractor)
        return self

    def _load_functions(self, extractor: TDBBlockExtractor) -> None:
        ser_text, = extractor.get_block(self.SER_BLOCK).values()
        other_text, = extractor.get_block(self.OTHER_BLOCK).values()

        # имена всех функций берём прямо из текста обоих блоков
        all_names = re.findall(r'^\s*FUNCTION\s+(\S+)', ser_text + '\n' + other_text, flags=re.M)

        functions: Dict[str, List[TdbFunctionData]] = {}
        for text in (ser_text, other_text):
            parsed = AutoFunctionTdbParser(text, all_names).parse()
            functions.update({name: rngs for name, rngs in parsed.items() if rngs})
        self.functions = functions

    def _load_parameters(self, extractor: TDBBlockExtractor) -> None:
        phase_blocks = [
            b for b in extractor.KNOWN_BLOCKS
            if b.startswith('THERMODYNAMIC PARAMETERS:') or b.startswith('THERMODYNAMIC DATA:')
        ]

        for block_name in phase_blocks:
            if not extractor.has_block(block_name):
                continue
            text, = extractor.get_block(block_name).values()
            parsed = ParameterTdbParser(text).parse()

            # фаза берётся из заголовка самого параметра, а не из названия
            # блока (в блоках вроде 'OXIDE PHASES' лежат параметры нескольких фаз)
            for param_type, items in parsed.items():
                for p in items:
                    self.parameters.setdefault(p.phase, {t: [] for t in ParameterTdbParser.KNOWN_PARAM_TYPES})
                    self.parameters[p.phase][param_type].append(p)

                    key: ParamKey = (param_type, self._constituents_key(p.constituents), p.order)
                    self._param_index.setdefault(p.phase, {}).setdefault(key, []).append(p)

    @staticmethod
    def _constituents_key(constituents: List[List[str]]) -> Tuple[Tuple[str, ...], ...]:
        return tuple(tuple(group) for group in constituents)

    # ------------------------------------------------------------------
    # Вычисление функций и параметров
    # ------------------------------------------------------------------
    def evaluate_function(self, name: str, T: float) -> float:
        """
        Значение функции (GHSERFE, GFEFCC, ...) при температуре T.
        Ссылки '...#' на другие функции разворачиваются рекурсивно.
        """
        func = TdbFunctionManipulation.get_function_from_dict(name, T, self.functions)
        if func is None:
            raise KeyError(f"Функция '{name}' не найдена или T={T} вне её диапазонов")
        return self._evaluate_data(func, T)

    def _evaluate_data(self, data: TdbFunctionData, T: float) -> float:
        extras = [self.evaluate_function(dep, T) for dep in data.dependencies]
        # polynome_plus ждёт список значений зависимостей одним аргументом,
        # а обычный polynome (без зависимостей) — только T
        if extras:
            return float(data.calculateGibbsEnergy(T, extras))
        return float(data.calculateGibbsEnergy(T))

    def find_parameter(
        self, phase: str, param_type: str,
        constituents: List[List[str]], order: int = 0,
    ) -> Optional[List[TdbParameterData]]:
        """Все T-диапазоны одного параметра фазы (или None, если такого нет)."""
        key: ParamKey = (param_type, self._constituents_key(constituents), order)
        return self._param_index.get(phase.upper(), {}).get(key)

    def evaluate_parameter(self, param_ranges: List[TdbParameterData], T: float) -> float:
        """Значение параметра при T (выбирается подходящий T-диапазон)."""
        for p in param_ranges:
            if p.t_min <= T <= p.t_max:
                return self._evaluate_data(p, T)
        raise ValueError(
            f"T={T} вне диапазонов параметра {param_ranges[0].name}: "
            f"{[(p.t_min, p.t_max) for p in param_ranges]}"
        )

    # ------------------------------------------------------------------
    # Проверки целостности
    # ------------------------------------------------------------------
    def check_dependencies(self, T: float = 1000.0) -> Dict[str, List[str]]:
        """
        Проверить, что все ссылки '...#' из параметров и функций разрешимы
        при температуре T. Возвращает {имя_ссылки: [где встретилась, ...]}
        только для неразрешённых ссылок (пустой словарь = всё в порядке).
        """
        unresolved: Dict[str, List[str]] = {}

        def check(owner: str, deps: List[str]):
            for dep in deps:
                if TdbFunctionManipulation.get_function_from_dict(dep, T, self.functions) is None:
                    unresolved.setdefault(dep, []).append(owner)

        for rngs in self.functions.values():
            for f in rngs:
                check(f.name, f.dependencies)
        for by_type in self.parameters.values():
            for items in by_type.values():
                for p in items:
                    check(p.name, p.dependencies)
        return unresolved


# Пример использования
if __name__ == '__main__':
    db = ThermodynamicDatabase('databases/mc_fe_v2062_clean.tdb').load()

    print(f"элементов: {len(db.elements)}, фаз: {len(db.phases)}, "
          f"функций: {len(db.functions)}, "
          f"фаз с параметрами: {len(db.parameters)}")

    # Все ссылки на функции должны разрешаться
    missing = db.check_dependencies(T=1000.0)
    print("Неразрешённые ссылки:", missing if missing else "нет")

    # Пример: G(FCC_A1,FE:VA;0) при 1000 К и GHSERFE для сравнения
    T = 1000.0
    ranges = db.find_parameter('FCC_A1', 'G', [['FE'], ['VA']], 0)
    print("G(FCC_A1,FE:VA;0) =", db.evaluate_parameter(ranges, T))
    print("GHSERFE           =", db.evaluate_function('GHSERFE', T))
