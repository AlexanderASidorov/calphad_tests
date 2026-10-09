#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Экспорт всех коэффициентов из ThermodynamicDatabase в pandas.

Скрипт загружает TDB-базу через ThermodynamicDatabase и собирает два
DataFrame (по одному на температурный интервал функции/параметра):

  1. functions_df  — FUNCTION-блоки  (GHSERFE, GFEFCC, GTI2NI, ...)
  2. parameters_df — PARAMETER-блоки (G, L, TC, BMAGN у каждой фазы)

Оба таблицы содержат одинаковый набор коэффициентов полинома SGTE
(b, t1, tlnt, t2, t3, t4, tm1, tm2, tm3, t7, tm9), диапазон температур
[t_min, t_max], ссылки на другие функции (dependencies / ref_coef),
исходное выражение (pure_text) и — для параметров — фазу, подрешётки
и порядок Redlich-Kister.

Запуск (см. пример внизу файла):
    python export_coefficients_to_pandas.py
"""

import numpy as np
import pandas as pd

from ThermodynamicDatabaseClass import ThermodynamicDatabase

# Порядок столбцов: сначала идентификация, потом коэффициенты, потом ссылки
COEFF_COLUMNS = [
    'b', 't1', 'tlnt', 't2', 't3', 't4',
    'tm1', 'tm2', 'tm3', 't7', 'tm9',
]

BASE_COLUMNS = [
    'name', 'kind', 'phase', 'param_type', 'constituents', 'order',
    't_min', 't_max', 'elements', 'dependencies', 'ref_coef', 'pure_text',
]

OUTPUT_COLUMNS = BASE_COLUMNS + COEFF_COLUMNS


def _format_constituents(constituents) -> str:
    """[['FE'], ['VA']] -> 'FE:VA' (пусто для обычных FUNCTION)."""
    if not constituents:
        return ''
    return ':'.join(','.join(group) for group in constituents)


def _row(data, kind: str) -> dict:
    """Собрать строку таблицы из одного TdbFunctionData/TdbParameterData."""
    row = {col: getattr(data, col, None) for col in COEFF_COLUMNS}

    # Параметры фазы имеют дополнительные поля, у обычных функций их нет
    row.update({
        'name': data.name,
        'kind': kind,
        'phase': getattr(data, 'phase', ''),
        'param_type': getattr(data, 'param_type', ''),
        'constituents': _format_constituents(getattr(data, 'constituents', []) or []),
        'order': getattr(data, 'order', ''),
        't_min': data.t_min,
        't_max': data.t_max,
        'elements': ';'.join(data.elements),
        'dependencies': ';'.join(data.dependencies),
        'ref_coef': ';'.join(str(c) for c in data.ref_coef),
        'pure_text': data.pure_text,
    })
    return row


def functions_to_dataframe(db: ThermodynamicDatabase) -> pd.DataFrame:
    """Все FUNCTION-интервалы -> DataFrame."""
    rows = [
        _row(func, kind='FUNCTION')
        for ranges in db.functions.values()
        for func in ranges
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def parameters_to_dataframe(db: ThermodynamicDatabase) -> pd.DataFrame:
    """Все PARAMETER-интервалы -> DataFrame (уровень фазы -> фаза в столбце)."""
    rows = [
        _row(param, kind='PARAMETER')
        for by_type in db.parameters.values()
        for params in by_type.values()
        for param in params
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


if __name__ == '__main__':
    # ==================================================================
    # Пример использования
    # ==================================================================

    # 1. Загрузка TDB-базы и сборка двух pandas DataFrame
    db = ThermodynamicDatabase('databases/mc_fe_v2062_clean.tdb').load()
    functions_df = functions_to_dataframe(db)      # FUNCTION-блоки
    parameters_df = parameters_to_dataframe(db)    # PARAMETER-блоки

    print(f"functions_df:  {functions_df.shape[0]} строк, "
          f"{functions_df.shape[1]} столбцов")
    print(f"parameters_df: {parameters_df.shape[0]} строк, "
          f"{parameters_df.shape[1]} столбцов")

   