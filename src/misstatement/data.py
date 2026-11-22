"""Loaders for the Bao et al. (2020) dataset and the AAER firm-year file."""
import numpy as np
import pandas as pd
from .config import DATA_RAW

# 28 raw Compustat items used by Bao et al. (2020)
RAW28 = ['act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib', 'invt', 'ivao',
         'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk', 're', 'rect', 'sale', 'sstk', 'txp', 'txt', 'xint', 'prcc_f']
RAW27 = [c for c in RAW28 if c != 'prcc_f']    # items available in SEC XBRL data (no market price)
# 14 Dechow et al. (2011) financial ratios supplied with the Bao data
RAT14 = ['dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets', 'ch_cs', 'ch_cm', 'ch_roa', 'issue',
         'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf']
ACCRUAL_RATIOS = ['dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets']

LABELS = {
    'act': 'Current assets', 'ap': 'Accounts payable', 'at': 'Total assets', 'ceq': 'Common equity',
    'che': 'Cash & short-term inv.', 'cogs': 'Cost of goods sold', 'csho': 'Shares outstanding',
    'dlc': 'Debt in current liab.', 'dltis': 'LT debt issued', 'dltt': 'LT debt', 'dp': 'Depreciation & amort.',
    'ib': 'Income before extraord.', 'invt': 'Inventory', 'ivao': 'Other investments', 'ivst': 'ST investments',
    'lct': 'Current liabilities', 'lt': 'Total liabilities', 'ni': 'Net income', 'ppegt': 'Gross PP&E',
    'pstk': 'Preferred stock', 're': 'Retained earnings', 'rect': 'Receivables', 'sale': 'Sales',
    'sstk': 'Equity issued', 'txp': 'Income taxes payable', 'txt': 'Income tax expense', 'xint': 'Interest expense',
    'prcc_f': 'Share price', 'dch_wc': 'Change in WC accruals', 'ch_rsst': 'Change in RSST accruals',
    'dch_rec': 'Change in receivables', 'dch_inv': 'Change in inventory', 'soft_assets': 'Soft assets',
    'ch_cs': 'Change in cash sales', 'ch_cm': 'Change in cash margin', 'ch_roa': 'Change in ROA',
    'issue': 'Securities issuance', 'bm': 'Book-to-market', 'dpi': 'Depreciation index', 'reoa': 'RE / assets',
    'EBIT': 'EBIT / assets', 'ch_fcf': 'Change in free cash flow',
}


def _find(name):
    for cand in (DATA_RAW / name, DATA_RAW / (name + '.gz')):
        if cand.exists():
            return cand
    raise FileNotFoundError(f'{name} not found in {DATA_RAW}. See README, section "Data".')


def load_bao(start=1991):
    """Bao et al. (2020) firm-year panel (Compustat, fiscal 1990-2014). Sample starts in 1991 as in the paper."""
    d = pd.read_csv(_find('data_FraudDetection_JAR2020.csv'))
    return d[d.fyear >= start].reset_index(drop=True)


def load_aaer():
    """AAER firm-years (CIK, misstated fiscal year, AAER number) released up to AAER #4012 (Dec 2018)."""
    a = pd.read_csv(_find('AAER_firm_year.csv'))
    return a.rename(columns={'P_AAER': 'p_aaer', 'CIK': 'cik', 'YEARA': 'fyear', 'UNDERSTATEMENT': 'understatement'})
