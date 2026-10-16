#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты на МИНИ-базе (синтетической), чтобы проверять код без полного mc_fe_v2062.
Запуск:  python test_phase_model.py
Реальная база: python test_phase_model.py --real
"""
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

    # чистый Fe в FCC: G_ref = GFEFCC, ideal = 0
    m = models['FCC_A1']
    y = m.y_from_sublattices([{'FE': 1.0}, {'VA': 1.0}])
    assert np.isclose(m.gibbs(1000.0, y), db.evaluate_function('GFEFCC', 1000.0))
    print("OK  G(FCC_A1,FE:VA) == GFEFCC at 1000 K")

    # градиент против конечных разностей, случайные y
    rng = np.random.default_rng(0)
    for ph in ('FCC_A1', 'BCC_A2', 'HCP_A3', 'LIQUID', 'CEMENTITE', 'SIGMA', 'M23C6'):
        m = models[ph]
        y = np.zeros(m.n_y)
        for s in range(m.n_sub):
            sl = slice(m._offsets[s], m._offsets[s + 1])
            v = rng.random(sl.stop - sl.start) + 0.05
            y[sl] = v / v.sum()
        m.check_y(y)
        g, h, err = m.gradient(1200.0, y), 1e-7, 0.0
        for i in range(m.n_y):
            yp, ym = y.copy(), y.copy()
            yp[i] += h
            ym[i] -= h
            fd = (m.gibbs(1200.0, yp) - m.gibbs(1200.0, ym)) / (2 * h)
            err = max(err, abs(g[i] - fd) / (abs(fd) + 1))
        assert err < 1e-5, (ph, err)
    print("OK  gradient vs finite differences on 7 phases")
    print("\nALL TESTS PASSED")


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else DB_PATH)