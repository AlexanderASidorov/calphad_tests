#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Парсер выражений PARAMETER (G, L, TC, BMAGN) внутри блока одной фазы,
например извлечённого через
    TDBBlockExtractor.get_block('THERMODYNAMIC PARAMETERS: FCC_A1').

Формат выражения:
    PARAMETER G(FCC_A1,FE:VA;0) 273.00 -1462.4+8.282*T-1.15*T*LN(T)
       +0.00064*T**2+GHSERFE#; 1811.00  Y
       -1713.815+0.94001*T+0.4925095E+31*T**(-9)+GHSERFE#; 6000.00  N
    REF:0 !

Заголовок TYPE(phase, sub1:sub2:...;order) разбирается отдельно от
T-полинома; сам T-полином разбирается той же логикой, что и в
FunctionParserClass — единственное отличие в том, что ссылки на другие
функции ('...#') ищутся динамически, по шаблону, а не по жёстко заданному
списку имён (параметр фазы может ссылаться практически на любую из уже
распарсенных функций SER / other-than-SER).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from GibbsEnergyFunctionsDataClasses import TdbFunctionData


@dataclass
class TdbParameterData(TdbFunctionData):
    """
    Параметр фазы (G, L, TC или BMAGN) для одного температурного диапазона.

    Наследует всю полиномиальную часть (b, t1, tlnt, ..., .formula,
    .calculateGibbsEnergy) от TdbFunctionData — параметр вычисляется
    численно точно так же, как обычная функция FUNCTION.
    """

    param_type: str = "G"                                          # 'G' | 'L' | 'TC' | 'BMAGN'
    phase: str = ""
    constituents: List[List[str]] = field(default_factory=list)    # по подрешёткам
    order: int = 0                                                  # порядок члена RK


class ParameterTdbParser:
    """Парсер всех PARAMETER-выражений в тексте блока одной фазы."""

    # Типы параметров, которые нам действительно нужны (HMVA и т.п. пропускаем)
    KNOWN_PARAM_TYPES = ('G', 'L', 'TC', 'BMAGN')

    _HEADER_PATTERN = re.compile(
        r'^\s*(?P<type>[A-Z0-9]+)\((?P<inner>[^)]*)\)',
        re.DOTALL,
    )

    # Границы температурных диапазонов — та же логика, что в TdbParser
    _RANGE_SPLIT_PATTERN = re.compile(r';\s*([\d.]+)\s+([YN])\s*')

    # Ссылка на другую функцию: необязательный коэффициент + ИМЯ + '#'
    _DEPENDENCY_PATTERN = re.compile(
        r'([+-]?\d*\.?\d*(?:[Ee][+-]?\d+)?)\*?([A-Z][A-Z0-9_]*)#'
    )

    def __init__(self, block_text: str):
        """
        Parameters
        ----------
        block_text : str
            Текст блока PARAMETER для одной фазы. Комментарии ('$...')
            будут удалены автоматически.
        """
        self.raw_data = self._strip_comments(block_text)
        self.parameters: Dict[str, List[TdbParameterData]] = {
            t: [] for t in self.KNOWN_PARAM_TYPES
        }

    @staticmethod
    def _strip_comments(text: str) -> str:
        lines = text.split('\n')
        return '\n'.join(
            '' if line.strip().startswith('$') else line
            for line in lines
        )

    def parse(self) -> Dict[str, List[TdbParameterData]]:
        """
        Разобрать все PARAMETER-выражения в блоке.

        Returns
        -------
        Dict[str, List[TdbParameterData]]
            Словарь: тип параметра -> список TdbParameterData
            (по одному объекту на температурный диапазон).
        """
        chunks = re.split(r'\bPARAMETER\s+', self.raw_data)

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            header_match = self._HEADER_PATTERN.match(chunk)
            if header_match is None:
                continue  # текст до первого PARAMETER — не относится к делу

            param_type = header_match.group('type').upper()
            if param_type not in self.KNOWN_PARAM_TYPES:
                continue  # неинтересный нам тип (например, HMVA)

            phase, constituents, order = self._parse_header_inner(
                header_match.group('inner')
            )

            body = chunk[header_match.end():]
            try:
                self._parse_parameter_body(param_type, phase, constituents, order, body)
            except Exception as e:
                print(f"Ошибка при парсинге {param_type}({phase},...;{order}): {e}")

        return self.parameters

    @staticmethod
    def _parse_header_inner(inner: str) -> Tuple[str, List[List[str]], int]:
        """
        Разобрать содержимое скобок PARAMETER: 'PHASE,sub1:sub2:...;order'.

        Examples
        --------
        'FCC_A1,AL:VA;0'        -> ('FCC_A1', [['AL'], ['VA']], 0)
        'GRAPHITE,B,C;0'        -> ('GRAPHITE', [['B', 'C']], 0)
        'GAMMA_PRIME,*:CR,NI;1' -> ('GAMMA_PRIME', [['*'], ['CR', 'NI']], 1)
        """
        composition_part, order_str = inner.rsplit(';', 1)
        phase, sublattice_str = composition_part.split(',', 1)

        constituents = [
            [species.strip().upper() for species in group.split(',') if species.strip()]
            for group in sublattice_str.split(':')
        ]

        return phase.strip().upper(), constituents, int(order_str.strip())

    def _parse_parameter_body(
        self,
        param_type: str,
        phase: str,
        constituents: List[List[str]],
        order: int,
        body: str,
    ) -> None:
        """Разобрать T-полином(ы) параметра — по той же логике, что и FUNCTION."""

        parts = self._RANGE_SPLIT_PATTERN.split(body)

        prev_tmax = 273.00
        i = 0
        while i < len(parts) - 2:
            expr = parts[i].strip()
            tmax = float(parts[i + 1])
            is_last = parts[i + 2] == 'N'

            tmin_match = re.match(r'^(\d+\.?\d*)\s+', expr)
            if tmin_match:
                tmin = float(tmin_match.group(1))
                expr = expr[tmin_match.end():]
            else:
                tmin = prev_tmax

            dep_names, dep_coefs = self._parse_dependencies(expr)
            coeffs = self._parse_coefficients(expr)

            name = f"{param_type}({phase},{self._format_constituents(constituents)};{order})"

            param_data = TdbParameterData(
                name=name,
                pure_text=expr,
                t_min=tmin,
                t_max=tmax,
                dependencies=dep_names,
                ref_coef=dep_coefs,
                param_type=param_type,
                phase=phase,
                constituents=constituents,
                order=order,
                **coeffs,
            )

            self.parameters[param_type].append(param_data)
            prev_tmax = tmax
            i += 3

            if is_last:
                break

    @staticmethod
    def _format_constituents(constituents: List[List[str]]) -> str:
        return ':'.join(','.join(group) for group in constituents)

    def _parse_dependencies(self, expr: str) -> Tuple[List[str], List[float]]:
        """Динамически найти в выражении все ссылки вида '<коэф>*ИМЯ#'."""
        deps: Dict[str, float] = {}

        for coef_str, func_name in self._DEPENDENCY_PATTERN.findall(expr):
            if coef_str in ('', '+'):
                coef = 1.0
            elif coef_str == '-':
                coef = -1.0
            else:
                try:
                    coef = float(coef_str)
                except ValueError:
                    coef = 1.0

            deps[func_name] = deps.get(func_name, 0.0) + coef

        return list(deps.keys()), list(deps.values())

    def _parse_coefficients(self, expr: str) -> Dict[str, float]:
        """
        Та же логика, что в FunctionParserClass._parse_coefficients,
        но ссылки на функции убираются динамически (см. _DEPENDENCY_PATTERN),
        а не по жёстко заданному списку.
        """
        coeffs = {
            'b': 0.0, 't1': 0.0, 'tlnt': 0.0, 't2': 0.0, 't3': 0.0,
            'tm1': 0.0, 't7': 0.0, 'tm9': 0.0, 't4': 0.0,
            'tm2': 0.0, 'tm3': 0.0
        }

        expr = self._DEPENDENCY_PATTERN.sub('', expr)
        expr = expr.replace(' ', '')

        number_pattern = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?'
        var_pattern = r'\*T(?:\*LN\(T\))?(?:\*\*(?:\([^)]+\)|-?\d+))?'
        term_pattern = f'({number_pattern})({var_pattern})?'

        mapping = {
            '': 'b', '*T': 't1', '*T*LN(T)': 'tlnt',
            '*T**2': 't2', '*T**3': 't3', '*T**4': 't4',
            '*T**(-1)': 'tm1', '*T**(-2)': 'tm2', '*T**(-3)': 'tm3',
            '*T**7': 't7', '*T**(-9)': 'tm9',
        }

        for match in re.finditer(term_pattern, expr):
            coef_str = match.group(1)
            var_part = match.group(2) or ''
            try:
                coef = float(coef_str)
            except ValueError:
                continue
            key = mapping.get(var_part)
            if key is not None:
                coeffs[key] += coef

        return coeffs


# Пример использования
if __name__ == '__main__':
    from BlockParserClass import TDBBlockExtractor

    extractor = TDBBlockExtractor('databases/mc_fe_v2062_clean.tdb')

    for phase_block_name in [
        'THERMODYNAMIC PARAMETERS: FCC_A1',
        'THERMODYNAMIC PARAMETERS: GRAPHITE',
        'THERMODYNAMIC PARAMETERS: LIQUID',
    ]:
        block, = extractor.get_block(phase_block_name).values()
        parser = ParameterTdbParser(block)
        params = parser.parse()

        print(f"\n=== {phase_block_name} ===")
        for param_type, items in params.items():
            print(f"  {param_type}: {len(items)} записей")

        # Печатаем первую запись каждого типа для проверки
        for param_type, items in params.items():
            if items:
                p = items[0]
                print(f"    пример {param_type}: phase={p.phase}, "
                      f"constituents={p.constituents}, order={p.order}, "
                      f"[{p.t_min}-{p.t_max}], deps={p.dependencies}")
