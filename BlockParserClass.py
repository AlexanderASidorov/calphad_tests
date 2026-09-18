#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 18:46:53 2026

@author: alexander
"""
import re
from typing import Dict, List, Optional, Tuple


class TDBBlockExtractor:
    """
    Извлекает блоки из TDB файла по комментариям-разделителям.
    Содержит захардкоженный список всех возможных блоков в базе mc_fe_v2062.
    """
    
    # Все возможные названия блоков в базе mc_fe_v2062
    KNOWN_BLOCKS = [
        "SER (Standard elements references)",
        "Gibbs energy functions other than SER",
        "THERMODYNAMIC PARAMETERS: LIQUID",
        "THERMODYNAMIC PARAMETERS: FCC_A1",
        "THERMODYNAMIC PARAMETERS: BCC_A2",
        "THERMODYNAMIC PARAMETERS: H_BCC",
        "THERMODYNAMIC PARAMETERS: HCP_A3",
        "THERMODYNAMIC PARAMETERS: ALPHA_MN",
        "THERMODYNAMIC PARAMETERS: BETA_MN",
        "THERMODYNAMIC PARAMETERS: BETA_RHOMBO_B",
        "THERMODYNAMIC PARAMETERS: DIAMOND_A4",
        "THERMODYNAMIC PARAMETERS: GRAPHITE",
        "THERMODYNAMIC PARAMETERS: BCC_B2",
        "THERMODYNAMIC PARAMETERS: NISI",
        "THERMODYNAMIC PARAMETERS: CHI_A12",
        "THERMODYNAMIC PARAMETERS: CO3MO",
        "THERMODYNAMIC PARAMETERS: CO3V",
        "THERMODYNAMIC PARAMETERS: COV3",
        "THERMODYNAMIC PARAMETERS: CR3MN5",
        "THERMODYNAMIC PARAMETERS: G_PHASE",
        "THERMODYNAMIC PARAMETERS: Mn2Ni3SI_PHASE",
        "THERMODYNAMIC PARAMETERS: GAMMA_PRIME",
        "THERMODYNAMIC PARAMETERS: LAVES_PHASE",
        "THERMODYNAMIC PARAMETERS: MNNI",
        "THERMODYNAMIC PARAMETERS: MNNI2",
        "THERMODYNAMIC PARAMETERS: MNNI_T3",
        "THERMODYNAMIC PARAMETERS: T2_MN3Ni10Si7",
        "THERMODYNAMIC PARAMETERS: T4_MNNISI",
        "THERMODYNAMIC PARAMETERS: Mn3Ni2SI - T7",
        "THERMODYNAMIC PARAMETERS: MU_PHASE",
        "THERMODYNAMIC PARAMETERS: MU_PHASE_I",
        "THERMODYNAMIC PARAMETERS: NI2SIH",
        "THERMODYNAMIC PARAMETERS: NI2SIL",
        "THERMODYNAMIC PARAMETERS: NI3SI2",
        "THERMODYNAMIC PARAMETERS: NI3SI2H",
        "THERMODYNAMIC PARAMETERS: NI3SIL",
        "THERMODYNAMIC PARAMETERS: NI5SI2",
        "THERMODYNAMIC PARAMETERS: ETA",
        "THERMODYNAMIC PARAMETERS: NITI2",
        "THERMODYNAMIC PARAMETERS: PI_PHASE",
        "THERMODYNAMIC PARAMETERS: R_PHASE",
        "THERMODYNAMIC PARAMETERS: SIGMA",
        "THERMODYNAMIC PARAMETERS: FE17Y2",
        "THERMODYNAMIC PARAMETERS: NI5Y",
        "THERMODYNAMIC PARAMETERS: ORDERED PD-FE-MN phases",
        "THERMODYNAMIC PARAMETERS: CEMENTITE",
        "THERMODYNAMIC PARAMETERS: CR2VC2",
        "THERMODYNAMIC PARAMETERS: K_CARB",
        "THERMODYNAMIC PARAMETERS: KSI_CARBIDE",
        "THERMODYNAMIC PARAMETERS: M3C2",
        "THERMODYNAMIC PARAMETERS: M6C",
        "THERMODYNAMIC PARAMETERS: M7C3",
        "THERMODYNAMIC PARAMETERS: M12C",
        "THERMODYNAMIC PARAMETERS: M23C6",
        "THERMODYNAMIC PARAMETERS: MOC_ETA",
        "THERMODYNAMIC PARAMETERS: V3C2",
        "THERMODYNAMIC PARAMETERS: WC",
        "THERMODYNAMIC PARAMETERS: EPS_CARB",
        "THERMODYNAMIC PARAMETERS: ETA_CARB",
        "THERMODYNAMIC PARAMETERS: FE24C10",
        "THERMODYNAMIC PARAMETERS: KSI_FE5C2",
        "THERMODYNAMIC PARAMETERS: ALN",
        "THERMODYNAMIC PARAMETERS: ALN_EQU",
        "THERMODYNAMIC PARAMETERS: BN_HP4",
        "THERMODYNAMIC PARAMETERS: FE4N",
        "THERMODYNAMIC PARAMETERS: MN6N4",
        "THERMODYNAMIC PARAMETERS: MN6N5",
        "THERMODYNAMIC PARAMETERS: MNSIN2",
        "THERMODYNAMIC PARAMETERS: SI3N4",
        "THERMODYNAMIC DATA: TAN_EPS",
        "THERMODYNAMIC PARAMETERS: ZET",
        "THERMODYNAMIC PARAMETERS: CRB",
        "THERMODYNAMIC PARAMETERS: CR2B",
        "THERMODYNAMIC PARAMETERS: CR5B3",
        "THERMODYNAMIC PARAMETERS: FEB",
        "THERMODYNAMIC PARAMETERS: FENBB",
        "THERMODYNAMIC PARAMETERS: FE3NB3B4",
        "THERMODYNAMIC PARAMETERS: M2B",
        "THERMODYNAMIC PARAMETERS: MNB2",
        "THERMODYNAMIC PARAMETERS: MNB4",
        "THERMODYNAMIC PARAMETERS: O_MN2B",
        "THERMODYNAMIC PARAMETERS: MN3B4",
        "THERMODYNAMIC PARAMETERS: MOB",
        "THERMODYNAMIC PARAMETERS: MOB2",
        "THERMODYNAMIC PARAMETERS: MO2M1B2",
        "THERMODYNAMIC PARAMETERS: NBB",
        "THERMODYNAMIC PARAMETERS: NB3B2",
        "THERMODYNAMIC PARAMETERS: NB5B6",
        "THERMODYNAMIC PARAMETERS: TIB",
        "THERMODYNAMIC PARAMETERS: TIB2",
        "THERMODYNAMIC PARAMETERS: TI3B4",
        "THERMODYNAMIC PARAMETERS: A_CHALC",
        "THERMODYNAMIC PARAMETERS: ANILITE",
        "THERMODYNAMIC PARAMETERS: B_CHALC",
        "THERMODYNAMIC PARAMETERS: COVELLITE",
        "THERMODYNAMIC PARAMETERS: CU2S",
        "THERMODYNAMIC PARAMETERS: DIGENITE",
        "THERMODYNAMIC PARAMETERS: DISULF",
        "THERMODYNAMIC PARAMETERS: DJURLEITE",
        "THERMODYNAMIC PARAMETERS: FC_MONO",
        "THERMODYNAMIC PARAMETERS: FES_P",
        "THERMODYNAMIC PARAMETERS: MNS_Q",
        "THERMODYNAMIC PARAMETERS: PYRR",
        "THERMODYNAMIC PARAMETERS: TIS",
        "THERMODYNAMIC PARAMETERS: TI4C2S2",
        "THERMODYNAMIC PARAMETERS: OXIDE PHASES",
        "THERMODYNAMIC PARAMETERS: M2P",
        "THERMODYNAMIC PARAMETERS: M3P",
        "THERMODYNAMIC PARAMETERS: BCC_DISL",
        "THERMODYNAMIC PARAMETERS: OXYGEN",
    ]
    
    def __init__(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.raw_text = f.read()
        self._lines = self.raw_text.split('\n')
        self._separator_pattern = re.compile(r'^\$[#*]{10,}')
    
    @staticmethod
    def _normalize(name: str) -> str:
        """Убирает все пробелы и приводит к верхнему регистру."""
        return re.sub(r'\s+', '', name).upper()
    
    def _find_block_bounds(self, name: str) -> Optional[Tuple[int, int]]:
        """
        Ищет блок по имени в файле.
        Возвращает (start_line_idx, end_line_idx) или None.
        """
        norm_name = self._normalize(name)
        
        # 1. Ищем строку-заголовок
        target_idx = None
        for i, line in enumerate(self._lines):
            stripped = line.strip()
            if stripped.startswith('$'):
                content = stripped.lstrip('$').strip()
                norm_content = self._normalize(content)
                if norm_content == norm_name:
                    target_idx = i
                    break
        
        if target_idx is None:
            return None
        
        # 2. Находим начало блока
        # Ищем первый разделитель после заголовка
        start = target_idx + 1
        found_separator = False
        while start < len(self._lines):
            if self._separator_pattern.match(self._lines[start].strip()):
                found_separator = True
                start += 1
                break
            start += 1
        
        if not found_separator:
            # Если разделителя нет, начало блока — следующая строка после заголовка
            start = target_idx + 1
            # Пропускаем пустые строки с $
            while start < len(self._lines):
                stripped = self._lines[start].strip()
                if stripped in ('', '$'):
                    start += 1
                else:
                    break
        
        # 3. Находим конец блока (следующий разделитель или заголовок другого блока)
        end = start
        while end < len(self._lines):
            stripped = self._lines[end].strip()
            
            # Проверяем, не разделитель ли это
            if self._separator_pattern.match(stripped):
                break
            
            # Проверяем, не заголовок ли это другого блока
            if stripped.startswith('$'):
                content = stripped.lstrip('$').strip()
                norm_content = self._normalize(content)
                if ('THERMODYNAMICPARAMETERS:' in norm_content or 
                    'THERMODYNAMICDATA:' in norm_content):
                    break
            
            end += 1
        
        return (start, end)
    
    def _remove_comments(self, text: str) -> str:
        """Удаляет строки-комментарии (начинающиеся с $)."""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            if line.strip().startswith('$'):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)
    
    def get_block(self, name: str) -> Dict[str, str]:
        """
        Извлекает блок по имени.
        
        Parameters
        ----------
        name : str
            Имя блока (например, 'THERMODYNAMIC PARAMETERS: TI3B4')
            
        Returns
        -------
        Dict[str, str]
            {имя_блока: содержимое_без_комментариев}
        """
        # 1. Проверяем, что имя известно
        if name not in self.KNOWN_BLOCKS:
            raise ValueError(
                f"Блок '{name}' отсутствует в списке KNOWN_BLOCKS.\n"
                f"Всего известно {len(self.KNOWN_BLOCKS)} блоков."
            )
        
        # 2. Ищем блок в файле
        bounds = self._find_block_bounds(name)
        if bounds is None:
            raise ValueError(f"Блок '{name}' не найден в файле.")
        
        start, end = bounds
        
        # 3. Извлекаем содержимое и удаляем комментарии
        block_lines = self._lines[start:end]
        cleaned = [line for line in block_lines if not line.strip().startswith('$')]
        
        return {name: '\n'.join(cleaned).strip()}
    
    def has_block(self, name: str) -> bool:
        """Проверяет, существует ли блок в файле."""
        return self._find_block_bounds(name) is not None
    
    def is_known(self, name: str) -> bool:
        """Проверяет, есть ли имя в списке KNOWN_BLOCKS."""
        return name in self.KNOWN_BLOCKS


# Пример использования
if __name__ == '__main__':
    extractor = TDBBlockExtractor('databases/mc_fe_v2062_clean.tdb')
    
    KNOWN_BLOCKS = extractor.KNOWN_BLOCKS
    
    # Запрашиваем блок
    result = extractor.get_block('SER (Standard elements references)')
