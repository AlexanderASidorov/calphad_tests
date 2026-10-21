#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты на реальной базе mc_fe_v2062_clean.tdb.
Запуск:  python test_phase_model.py [путь/к/базе.tdb]
"""
import contextlib
import io
import sys

import numpy as np

from ThermodynamicDatabaseClass import ThermodynamicDatabase
from PhaseModelClass import PhaseModel

DB_PATH = 'databases/mc_fe_v2062_clean.tdb'
T_TEST = 1200.0


def random_y(m, rng):
    """Случайные site fractions: сумма по каждой подрешётке = 1, все y > 0."""
    y = np.zeros(m.n_y)
    for s in range(m.n_sub):
        sl = slice(m._offsets[s], m._offsets[s + 1])
        v = rng.random(sl.stop - sl.start) + 0.05
        y[sl] = v / v.sum()
    return y


def reference_excess(m, T, y):
    """
    НЕЗАВИСИМАЯ медленная реализация избыточной энергии: прямой цикл по сырым
    L-параметрам базы, без массивов и без кода InteractionSet.
    """
    db = m.db
    val = {}                                               # (подрешётки, порядок) -> значение при T
    interactions = list(db.parameters[m.name]['L']) + [
        q for q in db.parameters[m.name]['G'] if any(len(g) > 1 for g in q.constituents)]
    for p in interactions:
        key = (tuple(tuple(sorted(g)) for g in p.constituents), p.order)
        if p.t_min <= T <= p.t_max and key not in val:
            val[key] = db.evaluate_parameter([p], T)

    def yv(s, sp):
        return y[m._offsets[s] + m.sublattices[s].index(sp)]

    total, done_tern = 0.0, set()
    for (cons, order), L in val.items():
        if any(sp != '*' and sp not in m.sublattices[s] for s, g in enumerate(cons) for sp in g):
            continue
        P = 1.0
        for s, g in enumerate(cons):
            if g != ('*',):
                for sp in g:
                    P *= yv(s, sp)
        multi = [(s, g) for s, g in enumerate(cons) if len(g) > 1]
        if not multi:
            total += P * L
        elif len(multi) == 1 and len(multi[0][1]) == 2:
            s, (a, b) = multi[0]
            total += P * (yv(s, a) - yv(s, b)) ** order * L
        elif len(multi) == 1 and len(multi[0][1]) == 3:
            if cons in done_tern:
                continue
            done_tern.add(cons)
            s, g = multi[0]
            Ls = [val.get((cons, o)) for o in (0, 1, 2)]
            if Ls[1] is None and Ls[2] is None:
                Ls = [Ls[0]] * 3
            Ls = [0.0 if x is None else x for x in Ls]
            ys = [yv(s, sp) for sp in g]
            rest = (1.0 - sum(ys)) / 3.0
            total += P * sum((ys[i] + rest) * Ls[i] for i in range(3))
        elif len(multi) == 1:                              # 4+ вида в одной подрешётке, порядок 0
            total += P * L
        elif len(multi) == 2:                              # (A,B:C,D)
            if order == 0:
                total += P * L
            elif order in (1, 2):
                s, (a, b) = multi[order - 1]
                total += P * (yv(s, a) - yv(s, b)) * L
    return total


def run(path=DB_PATH):
    db = ThermodynamicDatabase(path).load()
    print(f"load: {len(db.elements)} elements, {len(db.phases)} phases, "
          f"{len(db.functions)} functions, {len(db.parameters)} phases with parameters")
    assert not db.check_dependencies(1000.0), "есть неразрешённые ссылки '#'"
    assert set(db.phases) == set(db.parameters), "фазы и параметры не совпадают"

    # все функции при нескольких T: конечные значения, без выхода из диапазона
    with contextlib.redirect_stdout(io.StringIO()):   # get_function_from_dict печатает при промахе
        for T in (300.0, 1000.0, 1800.0, 3000.0):
            for name in db.functions:
                assert np.isfinite(db.evaluate_function(name, T)), (name, T)
    print("OK  all functions finite at 300/1000/1800/3000 K")

    # PhaseModel строится для всех фаз
    models = {ph: PhaseModel(db, ph) for ph in db.phases}
    print("OK  PhaseModel built for all", len(models), "phases")
    missing = {ph: len(m.missing_end_members) for ph, m in models.items() if m.missing_end_members}
    print("    phases with missing end-members:", len(missing),
          "| worst:", sorted(missing.items(), key=lambda kv: -kv[1])[:5])

    # то, что НЕ вошло в расчёт избыточной энергии, должно быть видно
    unsup = {ph: m.excess.unsupported for ph, m in models.items() if m.excess.unsupported}
    dups = {ph: m.excess.duplicates for ph, m in models.items() if m.excess.duplicates}
    print("    L-parameters not implemented:", sum(map(len, unsup.values())), unsup)
    print("    duplicate L ranges dropped:", sum(map(len, dups.values())), dups)
    assert not unsup, "есть нереализованные L-параметры"

    # чистый Fe в FCC: G_ref = GFEFCC, ideal = 0, excess = 0
    m = models['FCC_A1']
    y = m.y_from_sublattices([{'FE': 1.0}, {'VA': 1.0}])
    assert np.isclose(m.gibbs(1000.0, y), db.evaluate_function('GFEFCC', 1000.0))
    assert m.gibbs_excess(1000.0, y) == 0.0
    print("OK  G(FCC_A1,FE:VA) == GFEFCC at 1000 K, excess = 0 at the vertex")

    rng = np.random.default_rng(0)

    # excess: векторная реализация против независимого цикла, ВСЕ фазы
    worst = 0.0
    for ph, m in models.items():
        for _ in range(3):
            y = random_y(m, rng)
            fast, ref = m.gibbs_excess(T_TEST, y), reference_excess(m, T_TEST, y)
            worst = max(worst, abs(fast - ref) / (abs(ref) + 1.0))
    assert worst < 1e-9, worst
    print(f"OK  excess: vectorised == independent loop on all {len(models)} phases (max rel diff {worst:.1e})")

    # градиент против конечных разностей, ВСЕ фазы
    worst, worst_ph = 0.0, ''
    for ph, m in models.items():
        y = random_y(m, rng)
        g, h = m.gradient(T_TEST, y), 1e-6
        scale = max(np.abs(g).max(), 1.0)
        for i in range(m.n_y):
            yp, ym = y.copy(), y.copy()
            yp[i] += h
            ym[i] -= h
            fd = (m.gibbs(T_TEST, yp) - m.gibbs(T_TEST, ym)) / (2 * h)
            err = abs(g[i] - fd) / scale
            if err > worst:
                worst, worst_ph = err, ph
    assert worst < 1e-6, (worst, worst_ph)
    print(f"OK  gradient (ref + ideal + excess) vs finite differences on all phases (max scaled err {worst:.1e})")
    print("\nALL TESTS PASSED")


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else DB_PATH)
