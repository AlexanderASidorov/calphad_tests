#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:00:21 2026

@author: alexander
"""
from enum import Enum
from typing import Dict, Union, Optional, Callable, Tuple, List
from dataclasses import dataclass, field
import numpy as np
import re



class GibbsEnergy:
    
    @staticmethod
    def polynome (tdbfuncdata: 'TdbFunctionData', T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        расчет энергии Гиббса по стандартной формуле
        G(T) = b + t1·T + tlnt·T·ln(T) + t2·T² + t3·T³ + tm1/T + t7·T⁷ + tm9/T⁹ + t4·T⁴ + tm2/T² + tm3/T³
        """
        if np.any(T <= 0):
            raise ValueError("Температура должна быть > 0 K")
            
        if np.any(T < tdbfuncdata.t_min) or np.any(T > tdbfuncdata.t_max):
            raise ValueError(
                f"Температура выходит за пределы от {tdbfuncdata.t_min} до {tdbfuncdata.t_max}"
                f" для функции '{tdbfuncdata.name}'")
            
            
        
        tdbfuncdata.T = T         
        
        ln_T = np.log(T)
        G = (tdbfuncdata.b + 
             tdbfuncdata.t1 * T + 
             tdbfuncdata.tlnt * T * ln_T + 
             tdbfuncdata.t2 * T**2 + 
             tdbfuncdata.t3 * T**3 + 
             tdbfuncdata.tm1 / T + 
             tdbfuncdata.t7 * T**7 + 
             tdbfuncdata.tm9 / T**9 + 
             tdbfuncdata.t4 * T**4 + 
             tdbfuncdata.tm2 / T**2 + 
             tdbfuncdata.tm3 / T**3)
        
        tdbfuncdata.G = G
        
        return G
    
    
    @staticmethod
    def polynome_plus(tdbfuncdata: 'TdbFunctionData',
                      T: Union[float, np.ndarray],
                      extras: Union[float, np.ndarray]
                      ) -> Union[float, np.ndarray]:
        """
        Полином + дополнительные члены.
        Коэффициенты берутся из tdbfuncdata.ref_coef.

        Пример для GDHCNI = +0.5*GNIHCP# + 0.5*GHSERNI#:
            gdhcni.ref_coef = [0.5, 0.5]
            GibbsEnergy.polynome_plus(gdhcni, T, gnihip_G, ghserni_G)
        """
        G = GibbsEnergy.polynome(tdbfuncdata, T)
        for coef, val in zip(tdbfuncdata.ref_coef, extras):
            G = G + coef * val
        tdbfuncdata.G = G
        return G

    


@dataclass
class TdbFunctionData:
    """Данные термодинамической функции."""
    name: str
    t_min: float
    t_max: float
    pure_text: str = 'expression'
    
    elements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Коэффициенты полинома SGTE
    b:      float = 0.0      # свободный коэффициент
    t1:     float = 0.0      # коэффициент при T
    tlnt:   float = 0.0      # коэффициент при T·ln(T)
    t2:     float = 0.0      # коэффициент при T**2
    t3:     float = 0.0      # коэффициент при T**3
    tm1:    float = 0.0      # коэффициент при T**(-1)
    t7:     float = 0.0      # коэффициент при T**7
    tm9:    float = 0.0      # коэффициент при T**(-9)
    t4:     float = 0.0      # коэффициент при T**4
    tm2:    float = 0.0      # коэффициент при T**(-2)
    tm3:    float = 0.0      # коэффициент при T**(-3)
    
    
    # Коэффициенты перед дополнительными членами (ссылками на другие функции)
    ref_coef: List[float] = field(default_factory=lambda: [])
    
    # Значение энергии Гиббса   
    T: Optional[Union[float, np.ndarray]] = field(default=None, repr=False) # Температура при которой расчитывается энергия Гиббса
    G: Optional[Union[float, np.ndarray]] = field(default=None, repr=False) # Энергия Гиббса расчитываемая по той или иной формуле (см. класс  GibbsEnergy)

    # Формула для расчета энергии Гиббса
    formula: Optional[Callable] = field(default=None, repr=False)
    
    def __post_init__(self):
        """
        Автоматически выбирает формулу расчёта энергии Гиббса
        в зависимости от наличия зависимостей от других функций.
        """
        if self.formula is None:
            if not self.dependencies:  # если список зависимостей пустой
                self.formula = GibbsEnergy.polynome
            else:
                self.formula = GibbsEnergy.polynome_plus
    
    
      
    
    def calculateGibbsEnergy(self, T: Union[float, np.ndarray],
                         *extras: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
       
        """
        Рассчитать G по привязанной формуле (.formula).

        Parameters
        ----------
        T : температура(ы) в Кельвинах.
        *extras : значения .G других функций, на которые ссылается данная.
                    Например, для '+10083-4.813*T+GHSERAL#' сюда передаётся
                    уже рассчитанное значение GHSERAL. Порядок должен совпадать
                    с порядком коэффициентов в .ref_coef и имён в .dependencies.
        
        """
    
        if self.formula is None:
            raise ValueError("Необходимо определить вид функции для расчета энергии Гиббса."
                             " Назначьте атрибуту .formula метод из класса GibbsEnergy.")
            
        # Проверка диапазона температур
        T_arr = np.asarray(T)
        if np.any(T_arr < self.t_min) or np.any(T_arr > self.t_max):
            raise ValueError(
                f"Температура выходит за допустимый диапазон "
                f"[{self.t_min}, {self.t_max}] К. Получено: T = {T}"
                )
        
        # Проверка наличия зависимостей
        if self.dependencies and len(extras) != len(self.dependencies):
            missing_deps = [dep for dep, extra in zip(self.dependencies, extras + (None,) * len(self.dependencies)) if extra is None]
            raise ValueError(
                f"Missing required dependency {', '.join(missing_deps)} for {self.name} at T={T_arr[0]:.1f}K"
            )
        
        self.T = T
        self.G = self.formula(self, T, *extras)
    
        return self.G


class TdbFunctionManipulation:
    """Вспомогательные методы для работы с термодинамическими функциями"""
    
    @staticmethod
    def get_function_from_dict(
        func_name: str,
        temperature: float,
        functions: Dict[str, List['TdbFunctionData']]
    ) -> Optional['TdbFunctionData']:
        """
        Найти объект TdbFunctionData по имени функции и температуре.
        
        Статический метод, который ищет подходящий температурный диапазон
        для указанной функции и возвращает соответствующий объект.
        
        Parameters
        ----------
        func_name : str
            Имя функции (например, 'GHSERNI', 'GTIFCC').
        temperature : float
            Температура в Кельвинах.
        functions : Dict[str, List[TdbFunctionData]]
            Словарь функций, полученный после парсинга TDB файла.
        
        Returns
        -------
        TdbFunctionData или None
            Объект с подходящим температурным диапазоном,
            или None, если функция/диапазон не найдены.
        
        Examples
        --------
        >>> parser = OtherThenSerTdbParser(tdb_string)
        >>> functions = parser.parse()
        >>> func_data = TdbParser.get_function_data("GTI2NI", 1000.0, functions)
        >>> if func_data:
        ...     print(f"Диапазон: {func_data.t_min} - {func_data.t_max} K")
        """
        # 1. Проверяем, существует ли функция в словаре
        if func_name not in functions:
            print(f"Функция '{func_name}' не найдена в словаре.")
            return None
        
        func_ranges = functions[func_name]
        
        # 2. Проверяем, что список диапазонов не пуст
        if not func_ranges:
            print(f"Функция '{func_name}' не имеет спарсенных диапазонов.")
            return None
        
        # 3. Ищем диапазон, в который попадает температура
        for func_data in func_ranges:
            if func_data.t_min <= temperature <= func_data.t_max:
                return func_data
        
        # 4. Если ни один диапазон не подошёл
        print(
            f"Температура {temperature} K вне диапазонов функции '{func_name}': "
            f"доступные диапазоны: "
            f"{[(fd.t_min, fd.t_max) for fd in func_ranges]}"
        )
        return None
        
    





if __name__ == "__main__":
    
    
    """
    Тестовая функция для элемента в равновесном состоянии
    FUNCTION GHSERAL
     273.00 -7976.15+137.093038*T-24.3671976*T*LN(T)
     -1.884662E-3*T**2-0.877664E-6*T**3+74092*T**(-1); 700.00  Y
     
     -11276.24+223.048446*T-38.5844296*T*LN(T)
     +18.531982E-3*T**2-5.764227E-6*T**3+74092*T**(-1); 933.47  Y
     
     -11278.378+188.684153*T-31.748192*T*LN(T)-1.231E+28*T**(-9); 6000.00  N
    REF:0 !  
    """
    # Для первого диапазона
    ghseral_1 = TdbFunctionData(
    name="GHSERAL",
    t_min=273.00,
    t_max=700.00,
    elements=["AL"],
    b = -7976.15,
    t1=137.093038,
    tlnt=-24.3671976,
    t2=-1.884662e-3,
    t3=-0.877664e-6,
    tm1=74092.0,
    formula=GibbsEnergy.polynome)

    # Для второго диапазона
    ghseral_2 = TdbFunctionData(
        name="GHSERAL",
        t_min=700.00,
        t_max=933.47,
        elements=["AL"],
        b=-11276.24,
        t1=223.048446,
        tlnt=-38.5844296,
        t2=18.531982e-3,
        t3=-5.764227e-6,
        tm1=74092.0,
        formula=GibbsEnergy.polynome
    )
    
    # Для третьего диапазона
    ghseral_3 = TdbFunctionData(
        name="GHSERAL",
        t_min=933.47,
        t_max=6000.00,
        elements=["AL"],
        b=-11278.378,
        t1=188.684153,
        tlnt=-31.748192,
        tm9=-1.231e28,      
    )
   
        
    T1 = np.array([500.0, 700.])
    T2 = np.array([700.0, 800., 900])
    
    
    
    ghseral_1.calculateGibbsEnergy (T1)
    
    ghseral_2.calculateGibbsEnergy (T2)
    

    
    
    """
    Тестовая функция для элемена в ОЦК структуре
    FUNCTION GALBCC
    273.00 +10083-4.813*T+GHSERAL#; 6000.00  N
    REF:0 !    
    """
    
    galbcc =  TdbFunctionData(
        name="GALBCC",
        t_min=273,
        t_max=6000.00,
        elements=["AL"],
        dependencies=['GHSERAL'],
        b= 10083.,
        t1=-4.813,
        ref_coef = [1],
        formula = GibbsEnergy.polynome_plus)
    
    
    galbcc.calculateGibbsEnergy(T1, ghseral_1.G)
    

  
    



    
    """
    Тест: проверка ошибки при отсутствии зависимости (недостающий GHSERAL)
    Ожидается: ValueError с сообщением "Missing required dependency GHSERAL for GALBCC at T=500.0K"
    """
    try:
        # Попробуем вызвать calculateGibbsEnergy без передачи значения для GHSERAL
        galbcc.calculateGibbsEnergy(T1)
    except ValueError as e:
        print(f"✅ Ожидаемая ошибка: {e}")
    else:
        raise AssertionError("Ожидалась ошибка из-за отсутствия зависимости, но её не было.")
