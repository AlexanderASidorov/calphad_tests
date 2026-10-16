#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhaseModel — модель энергии Гиббса ОДНОЙ фазы в формализме подрешёток (CEF).

    G_m(T, y) = G_ref + G_ideal [+ G_excess + G_magnetic]   (последние два - следующие шаги)

Энергия считается на одну формульную единицу фазы (с учётом site ratios из
PHASE), как в TDB. Нормировка на моль атомов (без VA) будет отдельным шагом —
она нужна только при расчёте равновесия по составу.

Вектор site fractions y — плоский np.ndarray: сначала все виды 1-й подрешётки,
потом 2-й и т.д., в порядке CONSTITUENT. Соответствие индекс -> (подрешётка, вид)
лежит в self.species.
"""

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ThermodynamicDatabaseClass import ThermodynamicDatabase

R_GAS = 8.31451          # Дж/(моль*К), значение, принятое в TDB/Thermo-Calc
_Y_FLOOR = 1e-30         # защита ln(y) при y -> 0


class PhaseModel:

    def __init__(self, db: ThermodynamicDatabase, phase_name: str, strict: bool = False):
        """
        Parameters
        ----------
        strict : если True, отсутствие G-параметра для крайнего члена (end-member)
                 вызывает ошибку; иначе G такого члена = 0, а он попадает
                 в self.missing_end_members (проверьте этот список!).
        """
        name = phase_name.upper()
        if name not in db.phases:
            raise KeyError(f"Фаза '{phase_name}' не найдена в базе")
        self.db = db
        self.name = name
        self.definition = db.phases[name]
        self.site_ratios: List[float] = list(self.definition.site_ratios)
        self.sublattices: List[List[str]] = [list(s) for s in self.definition.sublattices]
        if not self.sublattices:
            raise ValueError(f"Для фазы {name} нет CONSTITUENT")
        self.n_sub = len(self.sublattices)

        # плоский индекс видов: [(подрешётка, вид), ...]
        self.species: List[Tuple[int, str]] = [
            (s, sp) for s, group in enumerate(self.sublattices) for sp in group
        ]
        self.n_y = len(self.species)
        offsets = np.cumsum([0] + [len(g) for g in self.sublattices])
        self._offsets = offsets
        self._site_of_index = np.array([s for s, _ in self.species])
        self._ratio_of_index = np.array([self.site_ratios[s] for s, _ in self.species])

        self._build_end_members(strict)

    # ------------------------------------------------------------------
    # Крайние члены (end-members)
    # ------------------------------------------------------------------
    def _build_end_members(self, strict: bool) -> None:
        """Все комбинации «по одному виду из каждой подрешётки» и их G-параметры."""
        self.end_members: List[Tuple[str, ...]] = list(itertools.product(*self.sublattices))

        # (n_em, n_sub): плоские индексы y, из которых строится произведение
        self._em_index = np.array([
            [self._offsets[s] + self.sublattices[s].index(sp) for s, sp in enumerate(em)]
            for em in self.end_members
        ], dtype=int)

        self._em_params = []
        self.missing_end_members: List[Tuple[str, ...]] = []
        for em in self.end_members:
            ranges = self.db.find_parameter(self.name, 'G', [[sp] for sp in em], 0)
            if ranges is None:
                self.missing_end_members.append(em)
            self._em_params.append(ranges)

        if strict and self.missing_end_members:
            raise KeyError(
                f"{self.name}: нет G-параметров для {len(self.missing_end_members)} "
                f"крайних членов, например {self.missing_end_members[:3]}"
            )

    def end_member_energies(self, T: float) -> np.ndarray:
        """G каждого крайнего члена при T (отсутствующие = 0)."""
        out = np.zeros(len(self.end_members))
        for k, ranges in enumerate(self._em_params):
            if ranges is not None:
                out[k] = self.db.evaluate_parameter(ranges, T)
        return out

    # ------------------------------------------------------------------
    # Работа с вектором y
    # ------------------------------------------------------------------
    def y_from_sublattices(self, fractions: Sequence[Dict[str, float]]) -> np.ndarray:
        """
        Собрать y из списка словарей (по одному на подрешётку), например
            [{'FE': 0.95, 'CR': 0.05}, {'C': 0.01, 'VA': 0.99}]
        Не указанные виды получают 0.
        """
        if len(fractions) != self.n_sub:
            raise ValueError(f"Нужно {self.n_sub} словарей (по числу подрешёток)")
        y = np.zeros(self.n_y)
        for s, frac in enumerate(fractions):
            for sp, val in frac.items():
                sp = sp.upper()
                if sp not in self.sublattices[s]:
                    raise KeyError(f"Вид {sp} не входит в подрешётку {s}: {self.sublattices[s]}")
                y[self._offsets[s] + self.sublattices[s].index(sp)] = val
        return y

    def check_y(self, y: np.ndarray, tol: float = 1e-8) -> None:
        """Проверка: y >= 0 и сумма по каждой подрешётке равна 1."""
        y = np.asarray(y, dtype=float)
        if y.shape != (self.n_y,):
            raise ValueError(f"y должен иметь длину {self.n_y}, получено {y.shape}")
        if np.any(y < -tol):
            raise ValueError("Отрицательные site fractions")
        for s in range(self.n_sub):
            total = y[self._offsets[s]:self._offsets[s + 1]].sum()
            if abs(total - 1.0) > tol:
                raise ValueError(f"Сумма y в подрешётке {s} = {total}, а не 1")

    # ------------------------------------------------------------------
    # Вклады в энергию Гиббса
    # ------------------------------------------------------------------
    def gibbs_reference(self, T: float, y: np.ndarray) -> float:
        """G_ref = sum_em ( prod_s y_{s,em} ) * G_em   (поверхность референсных состояний)."""
        weights = np.prod(y[self._em_index], axis=1)
        return float(weights @ self.end_member_energies(T))

    def gibbs_ideal(self, T: float, y: np.ndarray) -> float:
        """G_ideal = R*T * sum_s a_s * sum_i y_i ln y_i   (вакансия VA участвует как обычный вид)."""
        ylogy = np.where(y > 0.0, y * np.log(np.maximum(y, _Y_FLOOR)), 0.0)
        return float(R_GAS * T * np.sum(self._ratio_of_index * ylogy))

    def gibbs(self, T: float, y: np.ndarray) -> float:
        """Полная энергия Гиббса на формульную единицу (пока: ref + ideal)."""
        y = np.asarray(y, dtype=float)
        return self.gibbs_reference(T, y) + self.gibbs_ideal(T, y)

    # ------------------------------------------------------------------
    # Градиент по y (без учёта ограничений sum y = 1)
    # ------------------------------------------------------------------
    def gradient_reference(self, T: float, y: np.ndarray) -> np.ndarray:
        g_em = self.end_member_energies(T)
        Y = y[self._em_index]                                   # (n_em, n_sub)
        grad = np.zeros(self.n_y)
        for s in range(self.n_sub):
            others = np.prod(np.delete(Y, s, axis=1), axis=1)   # произведение по остальным подрешёткам
            np.add.at(grad, self._em_index[:, s], g_em * others)
        return grad

    def gradient_ideal(self, T: float, y: np.ndarray) -> np.ndarray:
        return R_GAS * T * self._ratio_of_index * (np.log(np.maximum(y, _Y_FLOOR)) + 1.0)

    def gradient(self, T: float, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return self.gradient_reference(T, y) + self.gradient_ideal(T, y)
