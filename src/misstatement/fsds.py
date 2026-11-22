"""SEC Financial Statement Data Sets (FSDS): quarter discovery, 10-K value extraction, firm-year panel
construction mapped to the Bao et al. (2020) Compustat items, XBRL-derived restatement labels and ratios.

FSDS files: sub.txt (one row per filing), num.txt (one row per reported value). See the SEC readme.htm."""
import glob
import hashlib
import os
import numpy as np
import pandas as pd

FORMS_ORIG = ['10-K', '10-KT']
FORMS_AMEND = ['10-K/A', '10-KT/A']
SUB_COLS = ['adsh', 'cik', 'name', 'sic', 'countryba', 'stprinc', 'afs', 'fye', 'form', 'period', 'fy', 'fp',
            'filed', 'prevrpt']

# Priority lists: the first tag a filing reports wins.
MAP = {
    'act': ['AssetsCurrent'],
    'ap': ['AccountsPayableCurrent', 'AccountsPayableAndAccruedLiabilitiesCurrent'],
    'at': ['Assets', 'LiabilitiesAndStockholdersEquity'],
    'seq': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'cash': ['CashAndCashEquivalentsAtCarryingValue', 'Cash', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'],
    'cashsti': ['CashAndShortTermInvestments'],
    'ivst': ['ShortTermInvestments', 'MarketableSecuritiesCurrent', 'AvailableForSaleSecuritiesCurrent',
             'AvailableForSaleSecuritiesDebtSecuritiesCurrent'],
    'cogs': ['CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfGoodsSold',
             'CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization', 'CostOfServices'],
    'csho': ['CommonStockSharesOutstanding'],
    'csho_dei': ['EntityCommonStockSharesOutstanding'],
    'debtcur': ['DebtCurrent', 'LongTermDebtAndCapitalLeaseObligationsCurrent'],
    'ltdcur': ['LongTermDebtCurrent'],
    'stb': ['ShortTermBorrowings'],
    'dltis': ['ProceedsFromIssuanceOfLongTermDebt', 'ProceedsFromIssuanceOfDebt'],
    'dltt': ['LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations'],
    'dp': ['DepreciationDepletionAndAmortization', 'DepreciationAndAmortization',
           'DepreciationAmortizationAndAccretionNet', 'Depreciation'],
    'ib': ['IncomeLossFromContinuingOperations', 'NetIncomeLoss', 'ProfitLoss'],
    'invt': ['InventoryNet'],
    'ivao': ['LongTermInvestments'],
    'lct': ['LiabilitiesCurrent'],
    'lt': ['Liabilities'],
    'ni': ['NetIncomeLoss', 'ProfitLoss'],
    'ppegt': ['PropertyPlantAndEquipmentGross'],
    'ppent': ['PropertyPlantAndEquipmentNet'],
    'pstk': ['PreferredStockValue'],
    're': ['RetainedEarningsAccumulatedDeficit'],
    'rect': ['AccountsReceivableNetCurrent', 'ReceivablesNetCurrent'],
    'sale': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet',
             'RevenueFromContractWithCustomerIncludingAssessedTax', 'SalesRevenueGoodsNet'],
    'sstk': ['ProceedsFromIssuanceOfCommonStock'],
    'txp': ['AccruedIncomeTaxesCurrent', 'TaxesPayableCurrent'],
    'txt': ['IncomeTaxExpenseBenefit'],
    'xint': ['InterestExpense', 'InterestExpenseDebt', 'InterestPaidNet', 'InterestPaid'],
    'float': ['EntityPublicFloat'],
    'cfo': ['NetCashProvidedByUsedInOperatingActivities'],
    'capx': ['PaymentsToAcquirePropertyPlantAndEquipment'],
    'oiadp': ['OperatingIncomeLoss'],
}
TAGS = sorted({t for v in MAP.values() for t in v} | {'CashAndShortTermInvestments'})
DEI_TAGS = {'EntityCommonStockSharesOutstanding', 'EntityPublicFloat'}
SHARE_ITEMS = {'csho', 'csho_dei'}


# ----------------------------------------------------------------------------------------------- discovery
def _dominant_quarter(sub_path):
    f = pd.read_csv(sub_path, sep='\t', usecols=['filed'], dtype=str, encoding='latin-1', quoting=3)
    if f.empty:
        return None, 0
    q = pd.to_datetime(f.filed, format='%Y%m%d').dt.to_period('Q').astype(str).value_counts()
    return q.index[0], len(f)


def _first_adsh(num_path):
    with open(num_path, encoding='latin-1') as fh:
        fh.readline()
        line = fh.readline()
    return line.split('\t')[0] if line else None


def discover_quarters(root):
    """Find every (sub, num) pair under `root`, label it by the dominant filing quarter, drop duplicate
    downloads and verify the pairing by checking that the num file's first accession number is in sub.
    Robust to the inconsistent '(2)', '(3)' download suffixes."""
    subs = sorted(glob.glob(os.path.join(root, '**', 'sub*.txt'), recursive=True))
    nums = sorted(glob.glob(os.path.join(root, '**', 'num*.txt'), recursive=True))
    first = {n: _first_adsh(n) for n in nums if os.path.getsize(n) > 1000}
    rows, seen = [], {}
    for s in subs:
        if os.path.getsize(s) < 1000:
            rows.append(dict(sub=s, num=None, quarter=None, status='empty sub file')); continue
        h = hashlib.md5(open(s, 'rb').read()).hexdigest()
        if h in seen:
            rows.append(dict(sub=s, num=None, quarter=seen[h], status='duplicate download')); continue
        q, n = _dominant_quarter(s)
        seen[h] = q
        adsh = set(pd.read_csv(s, sep='\t', usecols=['adsh'], dtype=str, encoding='latin-1', quoting=3).adsh)
        cands = [m for m in nums if os.path.dirname(m) == os.path.dirname(s) and first.get(m) in adsh]
        if not cands:
            rows.append(dict(sub=s, num=None, quarter=q, n_sub=n, status='no matching num file')); continue
        best = max(cands, key=os.path.getsize)        # duplicates or partial downloads: keep the largest
        rows.append(dict(sub=s, num=best, quarter=q, n_sub=n, status='ok'))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- extraction
def extract_quarter(sub_path, num_path, chunksize=2_000_000):
    """Keep 10-K / 10-KT (and amendments) filings; from num keep consolidated, unsegmented, annual (qtrs=4) or
    instant (qtrs=0) values in USD or shares for the mapped tags, dated at the fiscal period end or one year
    before (comparatives). Returns (subs, values)."""
    sub = pd.read_csv(sub_path, sep='\t', dtype=str, usecols=SUB_COLS, encoding='latin-1', quoting=3)
    sub = sub[sub.form.isin(FORMS_ORIG + FORMS_AMEND)]
    per = dict(zip(sub.adsh, sub.period.astype(int)))
    out = []
    header = pd.read_csv(num_path, sep='\t', nrows=0).columns
    usecols = [c for c in ['adsh', 'tag', 'ddate', 'qtrs', 'uom', 'coreg', 'segments', 'value'] if c in header]
    for ch in pd.read_csv(num_path, sep='\t', usecols=usecols, dtype=str, chunksize=chunksize,
                          encoding='latin-1', quoting=3):
        ch = ch[ch.adsh.isin(per) & ch.tag.isin(TAGS) & ch.qtrs.isin(['0', '4']) & ch.uom.isin(['USD', 'shares'])]
        if 'coreg' in ch:
            ch = ch[ch.coreg.isna()]
        if 'segments' in ch:
            ch = ch[ch.segments.isna()]
        p = ch.adsh.map(per)
        dd = ch.ddate.astype(int)
        ok = ch.tag.isin(DEI_TAGS) | ((dd <= p + 100) & (dd >= p - 10100))
        out.append(ch.loc[ok, ['adsh', 'tag', 'ddate', 'qtrs', 'uom', 'value']])
    vals = pd.concat(out) if out else pd.DataFrame(columns=['adsh', 'tag', 'ddate', 'qtrs', 'uom', 'value'])
    return sub, vals


def extract_all(root, out_dir, verbose=True):
    """Run discovery + extraction for every quarter; one parquet pair per quarter, skipped if present."""
    os.makedirs(out_dir, exist_ok=True)
    q = discover_quarters(root)
    for r in q[q.status == 'ok'].itertuples():
        key = r.quarter
        if os.path.exists(os.path.join(out_dir, f'{key}_num.parquet')):
            continue
        sub, vals = extract_quarter(r.sub, r.num)
        sub.to_parquet(os.path.join(out_dir, f'{key}_sub.parquet'))
        vals.to_parquet(os.path.join(out_dir, f'{key}_num.parquet'))
        if verbose:
            print(key, len(sub), 'filings', len(vals), 'values', flush=True)
    return q


def load_interim(out_dir):
    sub = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(out_dir, '*_sub.parquet')))])
    num = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(out_dir, '*_num.parquet')))])
    return sub.drop_duplicates('adsh'), num.drop_duplicates()


# ----------------------------------------------------------------------------------------------- panel
def _pick(wide, tags):
    out = pd.Series(np.nan, index=wide.index)
    for t in tags:
        if t in wide:
            m = out.isna() & wide[t].notna()
            out[m] = wide.loc[m, t]
    return out


def _fiscal_year(period):
    d = pd.to_datetime(period.astype(int).astype(str), format='%Y%m%d')
    return np.where(d.dt.month <= 5, d.dt.year - 1, d.dt.year)      # Compustat fyear convention


def wide_values(sub, num):
    """Per filing (adsh): mapped items for the current period ('cur') and the comparative year ('lag', suffix _l)."""
    num = num[num.adsh.isin(sub.adsh)].merge(sub[['adsh', 'period']], on='adsh')
    pdt = pd.to_datetime(num.period.astype(int).astype(str), format='%Y%m%d')
    dd = pd.to_datetime(num.ddate.astype(str), format='%Y%m%d', errors='coerce')
    lag = (pdt - dd).dt.days
    num['rel'] = np.select([lag.abs() <= 15, (lag - 365).abs() <= 20], ['cur', 'lag'], 'other')
    num.loc[num.tag.isin(DEI_TAGS), 'rel'] = 'cur'
    num = num[num.rel != 'other'].copy()
    num['value'] = num.value.astype(float)
    num['qtrs'] = num.qtrs.astype(int)
    num = num.sort_values('ddate').drop_duplicates(['adsh', 'tag', 'rel', 'qtrs'], keep='last')
    parts = {}
    for rel in ('cur', 'lag'):
        w = num[num.rel == rel].pivot_table(index='adsh', columns='tag', values='value', aggfunc='first')
        f = pd.DataFrame({k: _pick(w, tags) for k, tags in MAP.items()})
        f['che'] = f.cashsti.fillna(f.cash.fillna(0) + f.ivst.fillna(0)).where(f.cash.notna() | f.cashsti.notna())
        f['dlc'] = f.debtcur.fillna(f.ltdcur.fillna(0) + f.stb.fillna(0)).where(
            f.debtcur.notna() | f.ltdcur.notna() | f.stb.notna())
        f['ceq'] = f.seq - f.pstk.fillna(0)
        f['csho'] = f.csho.fillna(f.csho_dei)
        f['lt'] = f['lt'].fillna(f['at'] - f.seq)          # total liabilities is often untagged
        f = f / 1e6                                         # $ millions and millions of shares, as Compustat
        parts[rel] = f
    return parts['cur'].join(parts['lag'].add_suffix('_l'), how='outer')


def add_ratios(P):
    """Dechow et al. (2011)-style indicators computable from a filing and its comparative year."""
    avg_at = (P['at'] + P['at_l']) / 2
    wc = (P.act - P.che.fillna(0)) - (P.lct - P.dlc.fillna(0))
    wc_l = (P.act_l - P.che_l.fillna(0)) - (P.lct_l - P.dlc_l.fillna(0))
    P['x_dch_wc'] = (wc - wc_l) / avg_at
    P['x_dch_rec'] = (P.rect - P.rect_l) / avg_at
    P['x_dch_inv'] = (P.invt - P.invt_l) / avg_at
    P['x_soft_assets'] = (P['at'] - P.ppent.fillna(0) - P.che.fillna(0)) / P['at']
    P['x_ch_roa'] = P.ni / P['at'] - P.ni_l / P.at_l
    P['x_tacc'] = (P.ib - P.cfo) / avg_at                    # total accruals (Sloan, 1996)
    P['x_issue'] = ((P.sstk.fillna(0) > 0) | (P.dltis.fillna(0) > 0)).astype(float)
    P['x_reoa'] = P.re / P['at']
    P['x_ebit'] = (P.ib + P.xint.fillna(0) + P.txt.fillna(0)) / P['at']
    P['x_ch_sale'] = P.sale / P.sale_l - 1
    P['x_loss'] = (P.ni < 0).astype(float)
    P['x_lev'] = P['lt'] / P['at']
    P['x_size'] = np.log1p(P['at'].clip(lower=0))
    for c in [c for c in P.columns if c.startswith('x_')]:
        P[c] = P[c].replace([np.inf, -np.inf], np.nan)
    return P


XRATIOS = ['x_dch_wc', 'x_dch_rec', 'x_dch_inv', 'x_soft_assets', 'x_ch_roa', 'x_tacc', 'x_issue', 'x_reoa',
           'x_ebit', 'x_ch_sale', 'x_loss', 'x_lev', 'x_size']


def build_panel(sub, num):
    """Firm-year panel of ORIGINAL 10-K / 10-KT filings (as first reported), one row per CIK and fiscal year."""
    sub = sub.copy()
    sub['cik'] = sub.cik.astype(int)
    sub['fyear'] = _fiscal_year(sub.period)
    orig = sub[sub.form.isin(FORMS_ORIG)]
    W = wide_values(orig, num)
    P = orig.set_index('adsh').join(W).reset_index()
    P = P.sort_values('filed').drop_duplicates(['cik', 'fyear'], keep='last')
    P['sic'] = pd.to_numeric(P.sic, errors='coerce')
    P['spac'] = (P.sic == 6770).astype(int)
    P['financial'] = P.sic.between(6000, 6999).astype(int)
    return add_ratios(P).reset_index(drop=True)


def restatement_labels(sub, num, tol=0.005):
    """XBRL-derived restatement proxy: an original 10-K is flagged when a later 10-K/A for the same company and
    fiscal period reports net income, total assets or stockholders' equity that differs from the original by
    more than `tol` (relative). Amendments without financial statements (e.g. Part III only) carry no values
    and are not flagged."""
    sub = sub.copy()
    sub['cik'] = sub.cik.astype(int)
    sub['fyear'] = _fiscal_year(sub.period)
    W = wide_values(sub, num)[['ni', 'at', 'seq']]
    S = sub.set_index('adsh').join(W).reset_index()
    o = S[S.form.isin(FORMS_ORIG)].sort_values('filed').drop_duplicates(['cik', 'period'], keep='first')
    a = S[S.form.isin(FORMS_AMEND)]
    m = a.merge(o[['cik', 'period', 'ni', 'at', 'seq', 'filed']], on=['cik', 'period'], suffixes=('_a', '_o'))

    def chg(x, y):
        return x.notna() & y.notna() & ((x - y).abs() > tol * y.abs().clip(lower=1e-3))
    m['ni_chg'], m['at_chg'], m['seq_chg'] = chg(m.ni_a, m.ni_o), chg(m.at_a, m.at_o), chg(m.seq_a, m.seq_o)
    m['restated'] = m[['ni_chg', 'at_chg', 'seq_chg']].any(axis=1)
    m['days_to_amend'] = (pd.to_datetime(m.filed_a) - pd.to_datetime(m.filed_o)).dt.days
    m = m[m.days_to_amend > 0]
    r = m[m.restated].sort_values('filed_a').drop_duplicates(['cik', 'fyear'])
    return r[['cik', 'fyear', 'period', 'filed_o', 'filed_a', 'days_to_amend', 'ni_chg', 'at_chg', 'seq_chg']]
