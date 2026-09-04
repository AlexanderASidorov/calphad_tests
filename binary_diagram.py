#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:57:55 2026

@author: alexander
"""
import matplotlib.pyplot as plt
from pycalphad import Database, binplot, equilibrium
import pycalphad.variables as v
from pycalphad.plot.eqplot import eqplot


# =========================================================================
# Функция пересчёта массовой доли в мольную для бинарной системы
# =========================================================================

def wt_to_mole_fraction(wt_percent, element1, dbf, element2='FE'):
    """
    Пересчитывает массовую долю (в %) в мольную долю для бинарной системы.
    Молярные массы автоматически берутся из объекта Database (pycalphad).
    
    Parameters:
    -----------
    wt_percent : float
        Массовая доля элемента в процентах (например, 1.2 для 1.2%)
    element1 : str
        Символ элемента, для которого задана массовая доля
    dbf : pycalphad.Database
        Объект базы данных TDB, из которой будут взяты молярные массы
    element2 : str, optional
        Символ второго элемента (по умолчанию 'FE')
    
    Returns:
    --------
    float
        Мольная доля элемента 1
    
    Raises:
    -------
    KeyError
        Если один из элементов не найден в базе данных
    """
    # Приводим символы к верхнему регистру (в TDB-файлах они заглавные)
    el1 = element1.upper()
    el2 = element2.upper()
    
    # Проверяем наличие элементов в базе
    if el1 not in dbf.refstates:
        raise KeyError(f"Элемент '{element1}' не найден в базе данных. "
                       f"Доступные элементы: {sorted(dbf.elements)}")
    if el2 not in dbf.refstates:
        raise KeyError(f"Элемент '{element2}' не найден в базе данных. "
                       f"Доступные элементы: {sorted(dbf.elements)}")
    
    # Достаём молярные массы из базы данных
    M1 = dbf.refstates[el1]['mass']
    M2 = dbf.refstates[el2]['mass']
    
    # Массовая доля в долях единицы (не в процентах)
    w1 = wt_percent / 100.0
    w2 = 1.0 - w1
    
    # Мольная доля элемента 1
    x1 = (w1 / M1) / (w1 / M1 + w2 / M2)
    
    return x1





# 1. Загрузка базы данных (убедитесь, что файл лежит в той же папке, или укажите правильный путь)
db = Database('databases/mc_fe_v2062_clean.tdb')


# =========================================================================
# 1. ДИАГРАММА Fe-C (Железо-Углерод)
# =========================================================================

# Задаем интересующие нас элементы. VA - вакуум
components_fe_c = ['FE', 'C', 'VA']
# Основные фазы для стали и чугуна: Жидкость, Аустенит, Феррит, Цементит, Графит
my_phases_fe_c = ['LIQUID', 'FCC_A1', 'BCC_A2', 'GRAPHITE', 'CEMENTITE']

# Условия
# Расчитываем молярную долю легирующего элемента по значению его объемной доли
wt_alloying = 5
mole_fraction_alloying_element = wt_to_mole_fraction(wt_alloying, 'C', db, element2='FE') 
    
    

conditions_fe_c = {
    v.X('C'): (0, mole_fraction_alloying_element, 0.001), 
    v.T: (500, 2000, 1), 
    v.P: 101325, 
    v.N: 1
}

fig1 = plt.figure(figsize=(9, 6), dpi=150)
axes1 = fig1.gca()

# Вычисление и отрисовка
binplot(db, components_fe_c, my_phases_fe_c, conditions_fe_c, plot_kwargs=dict(ax=axes1))

axes1.set_title('Fe-C Phase Diagram')
axes1.set_xlabel('Mole fraction of C')
axes1.set_ylabel('Temperature (K)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# =========================================================================
# 2. ДИАГРАММА Fe-Mn (Железо-Марганец)
# =========================================================================

# Задаем интересующие нас элементы. VA - вакуум
components_fe_mn = ['FE', 'MN', 'VA']
# Фазы для Fe-Mn из базы MatCalc: Жидкость, FCC, BCC, HCP, ALPHA_MN, BETA_MN
phases_fe_mn = ['LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3', 'ALPHA_MN', 'BETA_MN', 'GRAPHITE', 'CEMENTITE']

# Условия расчета: T от 300K до 2000K, мольная доля Mn от 0 до 1.0 (вся бинарная система)
conds_fe_mn = {
    v.T: (300, 2000, 10),
    v.X('MN'): (0, 1.0, 0.02),
    v.P: 101325
}

# Вычисляем термодинамическое равновесие
eq_fe_mn = equilibrium(db, components_fe_mn, phases_fe_mn, conds_fe_mn)

# Строим график
fig2, ax2 = plt.subplots(figsize=(8, 6))
eqplot(eq_fe_mn, ax=ax2, x=v.X('MN'), y=v.T)
ax2.set_title('Fe-Mn Phase Diagram (MatCalc v2.062)')
ax2.set_xlabel('Mole fraction of Mn')
ax2.set_ylabel('Temperature (K)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()