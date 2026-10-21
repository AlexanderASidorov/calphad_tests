#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhaseModel — модель энергии Гиббса ОДНОЙ фазы в формализме подрешёток (CEF).

    G_m(T, y) = G_ref + G_ideal + G_excess [+ G_magnetic]   (магнитный вклад - следующий шаг)

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


class InteractionSet:
    """
    Все параметры одного типа (сейчас 'L'; позже 'TC'/'BMAGN') одной фазы,
    приведённые к виду, пригодному для векторного расчёта.

    К 'L' добавляются G-параметры с несколькими видами в подрешётке (взаимодействие,
    записанное в TDB как G(phase,A,B;n)).

    Каждый член имеет вид  P * F * value(T), где
      P = произведение site fractions всех видов параметра
          (символ '*' означает «любой вид подрешётки», sum y = 1, поэтому
          такая подрешётка просто выпадает из произведения),
      F = множитель Редлиха-Кистера:
          - один вид в каждой подрешётке (end-member) или порядок 0 -> F = 1
          - бинарное взаимодействие в одной подрешётке  -> F = (y_a - y_b)^order
          - взаимодействие 2+2 (A,B:C,D): order 1 -> (y_A - y_B), order 2 -> (y_C - y_D)
      Тернарное взаимодействие в одной подрешётке (Мугьяну):
          P * (v_a*L0 + v_b*L1 + v_c*L2),  v_i = y_i + (1 - y_a - y_b - y_c)/3.
          Если для тройки задан ТОЛЬКО порядок 0, он берётся для всех трёх (L0=L1=L2).
    Виды внутри подрешётки упорядочиваются по алфавиту (как требует TDB).

    Всё, что реализовать нельзя, попадает в self.unsupported (не молча).
    """

    def __init__(self, model: 'PhaseModel', ptype: str):
        self.model, self.ptype = model, ptype
        self.unsupported: List[Tuple] = []
        self.duplicates: List[Tuple] = []
        self._dummy = model.n_y                       # индекс «единицы» в расширенном векторе y

        # --- 1. сгруппировать диапазоны T по каноническому ключу (подрешётки, порядок)
        groups: Dict[Tuple, list] = {}
        by_type = model.db.parameters.get(model.name, {})
        source = list(by_type.get(ptype, []))
        if ptype == 'L':
            # в TDB взаимодействие иногда записано как G(phase,A,B;n) (несколько видов
            # в подрешётке) — по смыслу это тот же L-параметр
            source += [q for q in by_type.get('G', []) if any(len(g) > 1 for g in q.constituents)]
        for p in source:
            key = (tuple(tuple(sorted(g)) for g in p.constituents), p.order)
            groups.setdefault(key, []).append(p)

        entries: Dict[Tuple, int] = {}                # ключ -> индекс значения
        self._ranges: List[list] = []
        for key, plist in groups.items():
            plist = sorted(plist, key=lambda q: (q.t_min, q.t_max))
            kept = []
            for q in plist:
                if kept and q.t_min < kept[-1].t_max - 1e-9:      # перекрытие диапазонов
                    self.duplicates.append(key)
                    continue
                kept.append(q)
            entries[key] = len(self._ranges)
            self._ranges.append(kept)
        self._zero = len(self._ranges)                # индекс нулевого значения

        # --- 2. разобрать каждый ключ в член
        plain, ternary, ternary_seen = [], [], set()
        for (cons, order), pidx in entries.items():
            if len(cons) != model.n_sub:
                self.unsupported.append((cons, order, 'число подрешёток')); continue
            if any('*' in g and len(g) > 1 for g in cons):
                self.unsupported.append((cons, order, "'*' внутри группы")); continue
            if any(sp != '*' and sp not in model.sublattices[s] for s, g in enumerate(cons) for sp in g):
                self.unsupported.append((cons, order, 'вид не из CONSTITUENT')); continue

            members = []
            for s, g in enumerate(cons):
                if g == ('*',):
                    continue
                members += [model._offsets[s] + model.sublattices[s].index(sp) for sp in g]
            multi = [(s, g) for s, g in enumerate(cons) if len(g) > 1]
            fidx = lambda s, sp: model._offsets[s] + model.sublattices[s].index(sp)

            if not multi or (len(multi) == 1 and len(multi[0][1]) >= 4 and order == 0) \
                    or (len(multi) == 2 and order == 0 and all(len(g) == 2 for _, g in multi)):
                plain.append((members, self._dummy, self._dummy, 0, pidx))
            elif len(multi) == 1 and len(multi[0][1]) == 2:
                s, g = multi[0]
                plain.append((members, fidx(s, g[0]), fidx(s, g[1]), order, pidx))
            elif len(multi) == 2 and all(len(g) == 2 for _, g in multi) and order in (1, 2):
                s, g = multi[order - 1]
                plain.append((members, fidx(s, g[0]), fidx(s, g[1]), 1, pidx))
            elif len(multi) == 1 and len(multi[0][1]) == 3 and order in (0, 1, 2):
                tkey = cons
                if tkey in ternary_seen:
                    continue
                ternary_seen.add(tkey)
                s, g = multi[0]
                lids = [entries.get((cons, o)) for o in (0, 1, 2)]
                if lids[1] is None and lids[2] is None:
                    lids = [lids[0]] * 3                              # только L0 -> одинаков для всех
                lids = [self._zero if l is None else l for l in lids]
                ternary.append((members, [fidx(s, sp) for sp in g], lids))
            else:
                self.unsupported.append((cons, order, 'не реализовано'))

        # --- 3. массивы
        def pad(rows):
            width = max((len(r) for r in rows), default=1)
            return np.array([r + [self._dummy] * (width - len(r)) for r in rows], dtype=int).reshape(len(rows), width)

        self.n_plain, self.n_tern = len(plain), len(ternary)
        self.p_m = pad([r[0] for r in plain])
        self.p_a = np.array([r[1] for r in plain], dtype=int)
        self.p_b = np.array([r[2] for r in plain], dtype=int)
        self.p_n = np.array([r[3] for r in plain], dtype=int)
        self.p_v = np.array([r[4] for r in plain], dtype=int)
        self.t_m = pad([r[0] for r in ternary])
        self.t_i = np.array([r[1] for r in ternary], dtype=int).reshape(-1, 3)
        self.t_l = np.array([r[2] for r in ternary], dtype=int).reshape(-1, 3)
        self._cache_T: Optional[float] = None
        self._cache_vals: Optional[np.ndarray] = None

    def values(self, T: float) -> np.ndarray:
        """Значения всех параметров при T (+ нулевой элемент в конце), с кэшем по T."""
        if self._cache_T != T:
            vals = np.zeros(len(self._ranges) + 1)
            for k, rngs in enumerate(self._ranges):
                vals[k] = self.model.db.evaluate_parameter(rngs, T)
            self._cache_T, self._cache_vals = T, vals
        return self._cache_vals

    @staticmethod
    def _others_product(Y: np.ndarray, c: int) -> np.ndarray:
        return np.prod(np.delete(Y, c, axis=1), axis=1)

    def energy(self, T: float, y: np.ndarray) -> float:
        vals, ye, total = self.values(T), np.append(y, 1.0), 0.0
        if self.n_plain:
            P = np.prod(ye[self.p_m], axis=1)
            F = (ye[self.p_a] - ye[self.p_b]) ** self.p_n
            total += float(np.sum(P * F * vals[self.p_v]))
        if self.n_tern:
            P = np.prod(ye[self.t_m], axis=1)
            Yt = ye[self.t_i]
            v = Yt + (1.0 - Yt.sum(axis=1, keepdims=True)) / 3.0
            S = np.sum(v * vals[self.t_l], axis=1)
            total += float(np.sum(P * S))
        return total

    def gradient(self, T: float, y: np.ndarray) -> np.ndarray:
        vals, ye = self.values(T), np.append(y, 1.0)
        grad = np.zeros(len(ye))
        if self.n_plain:
            Y = ye[self.p_m]
            P = np.prod(Y, axis=1)
            D = ye[self.p_a] - ye[self.p_b]
            F = D ** self.p_n
            L = vals[self.p_v]
            for c in range(Y.shape[1]):
                np.add.at(grad, self.p_m[:, c], self._others_product(Y, c) * F * L)
            dF = np.where(self.p_n > 0, self.p_n * D ** np.maximum(self.p_n - 1, 0), 0.0)
            np.add.at(grad, self.p_a, P * dF * L)
            np.add.at(grad, self.p_b, -P * dF * L)
        if self.n_tern:
            Y = ye[self.t_m]
            P = np.prod(Y, axis=1)
            Yt = ye[self.t_i]
            Lt = vals[self.t_l]
            v = Yt + (1.0 - Yt.sum(axis=1, keepdims=True)) / 3.0
            S = np.sum(v * Lt, axis=1)
            for c in range(Y.shape[1]):
                np.add.at(grad, self.t_m[:, c], self._others_product(Y, c) * S)
            dS = Lt - Lt.sum(axis=1, keepdims=True) / 3.0
            for c in range(3):
                np.add.at(grad, self.t_i[:, c], P * dS[:, c])
        return grad[:-1]                                  # отбросить «фиктивную» единицу


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
        self._em_cache: Tuple[Optional[float], Optional[np.ndarray]] = (None, None)
        self.excess = InteractionSet(self, 'L')

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
        """G каждого крайнего члена при T (отсутствующие = 0); кэшируется по T."""
        if self._em_cache[0] == T:
            return self._em_cache[1]
        out = np.zeros(len(self.end_members))
        for k, ranges in enumerate(self._em_params):
            if ranges is not None:
                out[k] = self.db.evaluate_parameter(ranges, T)
        self._em_cache = (T, out)
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

    def gibbs_excess(self, T: float, y: np.ndarray) -> float:
        """Избыточная энергия (Редлих-Кистер/Мугьяну по L-параметрам)."""
        return self.excess.energy(T, y)

    def gibbs(self, T: float, y: np.ndarray) -> float:
        """Энергия Гиббса на формульную единицу: ref + ideal + excess (магнитный вклад - позже)."""
        y = np.asarray(y, dtype=float)
        return self.gibbs_reference(T, y) + self.gibbs_ideal(T, y) + self.gibbs_excess(T, y)

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

    def gradient_excess(self, T: float, y: np.ndarray) -> np.ndarray:
        return self.excess.gradient(T, y)

    def gradient(self, T: float, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return (self.gradient_reference(T, y) + self.gradient_ideal(T, y)
                + self.gradient_excess(T, y))
