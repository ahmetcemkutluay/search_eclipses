""" Download TESS lightcurves using given criteria.

Note on why this does not use lightkurve.search_lightcurve: that call resolves
the TIC as a name, cone-searches around it, collects every observation in the
cone, and only then asks MAST for the product list -- applying the author /
sector / exptime filters afterwards.  The product-list call is the expensive
one, so filtering cannot help, and a single target can take minutes.  Going
through astroquery directly, with the obsids the catalogue stage already
recorded, keeps the expensive call batched and bounded.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from astroquery.mast import Observations
from astropy.table import vstack

import warnings
try:
    from erfa import ErfaWarning
    warnings.filterwarnings('ignore', category=ErfaWarning)
except ImportError:
    pass


class Criteria:
    """Select pipeline, cadence and sectors for each target in the catalogue.

    Reads the long product inventory written by tess_products.py, so no
    selection relies on a guessed cadence or on a sector list pooled
    across pipelines.
    """

    def __init__(self, information_file, product_file,
                 provs=('SPOC', 'TGLC', 'QLP', 'T16'), ignore_exptimes=(20,)):
        self.info = pd.read_csv(information_file, delimiter=',')
        self.prod = pd.read_csv(product_file, delimiter=',')
        self.provs = tuple(provs)
        self.ignore = set(ignore_exptimes)

        missing = [c for c in ('tic', 'prov', 'sector')
                   if c not in self.prod.columns]
        if missing:
            raise ValueError(f'columns not found in product table: {missing}')

        missing = [c for c in ('tic', 'short_period', 'n_sec_needed')
                   if c not in self.info.columns]
        if missing:
            raise ValueError(f'columns not found in catalogue: {missing}')

        self.info['tic'] = self.info['tic'].astype(int)
        self.prod['tic'] = self.prod['tic'].astype(int)
        self.prod['sector'] = self.prod['sector'].astype(int)

        if 'exptime' in self.prod.columns:
            self.prod['exptime'] = pd.to_numeric(
                self.prod['exptime'], errors='coerce').astype('Int64')
        else:
            warnings.warn("product table has no 'exptime' column; "
                          "cadence selection falls back to pipeline order")
            self.prod['exptime'] = pd.Series([pd.NA] * len(self.prod),
                                             dtype='Int64')

        if 'obsid' not in self.prod.columns:
            warnings.warn("product table has no 'obsid' column; the download "
                          "stage will have to re-query MAST for every target")
            self.prod['obsid'] = pd.NA

        # what we are willing to download: 20 s dropped, unknown kept
        self.usable = self.prod[~self.prod['exptime'].isin(list(self.ignore))].copy()

    # ---------- pipeline ----------

    def assign_prov(self, tic):
        """First pipeline in priority order with a usable product.

        Single-target lookup, for inspection.  assign_all() does not use it:
        one scan of the product table per target is O(targets x products),
        which is minutes on a small catalogue and hours on a large one.
        """
        have = set(self.usable.loc[self.usable['tic'] == tic, 'prov'])
        for prov in self.provs:
            if prov in have:
                return prov
        return None

    def assign_all(self):
        """Add a 'prov' column to the catalogue, in one pass."""
        rank = {p: i for i, p in enumerate(self.provs)}
        u = self.usable[['tic', 'prov']].copy()
        u['rank'] = u['prov'].map(rank)
        u = u.dropna(subset=['rank'])

        best = u.groupby('tic')['rank'].min()
        inverse = {i: p for p, i in rank.items()}

        self.info['prov'] = self.info['tic'].map(best).map(inverse)
        return self.info

    # ---------- cadence ----------

    def pick_exptime(self, available):
        """Pick the highest cadence (shortest exposure), ignoring 20 s."""
        allowed = [e for e in available
                   if pd.notna(e) and e not in self.ignore]
        return min(allowed) if allowed else None

    # ---------- sectors ----------

    def n_sectors(self, row):
        """Number of sectors required: 1 for short-period targets, else n_sec_needed."""
        sp = row.get('short_period')
        if pd.notna(sp) and int(sp) == 1:
            return 1
        n = row.get('n_sec_needed')
        return 1 if pd.isna(n) else int(n)

    def select(self, row):
        """Return (prov, exptime, [sectors]) for one catalogue row."""
        tic = int(row['tic'])

        prov = row.get('prov')
        if prov is None or pd.isna(prov):
            prov = self.assign_prov(tic)
        if prov is None:
            return None, None, []

        sub = self.usable[(self.usable['tic'] == tic) &
                          (self.usable['prov'] == prov)]
        if sub.empty:
            return prov, None, []

        # cadence first, then the latest sectors that actually have it
        exp = self.pick_exptime(sub['exptime'].tolist())
        if exp is not None:
            sub = sub[sub['exptime'] == exp]

        sectors = sorted(set(int(s) for s in sub['sector']))
        n = min(self.n_sectors(row), len(sectors))

        return prov, exp, (sectors[-n:] if n > 0 else [])

    # ---------- output ----------

    def write_output(self, path='download_list.csv'):
        """Write one row per (tic, prov, exptime, sector, obsid), in one pass."""
        if 'prov' not in self.info.columns:
            self.assign_all()

        need = (self.info[['tic', 'prov', 'short_period', 'n_sec_needed']]
                .dropna(subset=['prov']))
        u = self.usable.merge(need, on=['tic', 'prov'], how='inner')

        cols = ['tic', 'prov', 'exptime', 'sector', 'obsid']
        if u.empty:
            out = pd.DataFrame(columns=cols)
            out.to_csv(path, index=False)
            return out

        # cadence: the shortest exposure this target has, where any is known
        best = u.groupby('tic')['exptime'].transform('min')
        same = (u['exptime'] == best).fillna(False).astype(bool)
        u = u[same | best.isna()]

        # one row per sector; if MAST holds several observations for the same
        # sector, keep the first obsid
        u = (u.sort_values(['tic', 'sector', 'obsid'])
              .drop_duplicates(subset=['tic', 'prov', 'sector']))

        # how many sectors this target needs
        short = pd.to_numeric(u['short_period'], errors='coerce').fillna(0)
        want = pd.to_numeric(u['n_sec_needed'], errors='coerce').fillna(1)
        u['n_want'] = np.where(short.astype(int) == 1, 1, want.astype(int))

        # keep the latest n_want sectors of each target
        u['rk'] = u.groupby('tic')['sector'].rank(method='first',
                                                  ascending=False)
        u = u[u['rk'] <= u['n_want']]

        out = (u[cols].sort_values(['tic', 'sector']).reset_index(drop=True))
        out.to_csv(path, index=False)
        return out


class Download():
    """Resolve products in batches, then transfer them target by target."""

    def __init__(self, list_file, out_dir='lightcurves', status_file=None,
                 criteria=None, ignore_exptimes=(20,)):

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.criteria = criteria
        self.ignore = set(ignore_exptimes)

        self.targets = self._load_list(list_file)
        self.status = self._load_status(status_file)

        self._products = None        # astropy Table, filled by resolve_products
        self._index = None           # pandas view: tic / prov / sector / row

    # ---------- preparation ----------

    def _load_list(self, target_file):
        """ load the download list """
        df = pd.read_csv(target_file, delimiter=',')
        df = df.dropna(subset=['prov']).copy()
        df['tic'] = df['tic'].astype(int)
        df['sector'] = df['sector'].astype(int)

        if 'exptime' in df.columns:
            df['want_exptime'] = pd.to_numeric(df['exptime'],
                                               errors='coerce').astype('Int64')
        else:
            df['want_exptime'] = pd.Series([pd.NA] * len(df), dtype='Int64')

        if 'obsid' in df.columns:
            df['obsid'] = df['obsid'].astype(str)
        else:
            df['obsid'] = ''

        return df.reset_index(drop=True)

    def _load_status(self, status_file=None):
        """Resume from an existing status file, or start a fresh one."""
        self.status_file = (Path(status_file) if status_file is not None
                            else self.out_dir / 'download_status.csv')

        if self.status_file.is_file():
            status = pd.read_csv(self.status_file, delimiter=',')
            status['tic'] = status['tic'].astype(int)
            status['sector'] = status['sector'].astype(int)
            status['obsid'] = status['obsid'].astype(str)
            for col in ('want_exptime', 'exptime'):
                status[col] = pd.to_numeric(status[col],
                                            errors='coerce').astype('Int64')
            return status

        status = self.targets[['tic', 'prov', 'sector',
                               'want_exptime', 'obsid']].copy()
        status['status'] = 'pending'
        status['exptime'] = pd.Series([pd.NA] * len(status), dtype='Int64')
        status['n_files'] = pd.Series([pd.NA] * len(status), dtype='Int64')

        return status

    def _save_status(self):
        """Write the status table to disk, atomically."""
        tmp = self.status_file.with_suffix('.tmp')
        self.status.to_csv(tmp, index=False)
        tmp.replace(self.status_file)

    # ---------- helpers ----------

    def pending_rows(self):
        return self.status[self.status['status'] == 'pending']

    def pending_sectors(self, tic, prov):
        """Sectors of this target that have not been attempted yet."""
        rows = self.status[(self.status['tic'] == tic) &
                           (self.status['prov'] == prov) &
                           (self.status['status'] == 'pending')]
        return sorted(int(s) for s in rows['sector'])

    def _mark(self, tic, prov, sectors, state, exptime=None, n_files=None):
        """Set status for the given sectors of one target."""
        if not len(sectors):
            return
        mask = ((self.status['tic'] == tic) &
                (self.status['prov'] == prov) &
                (self.status['sector'].isin(list(sectors))))
        self.status.loc[mask, 'status'] = state
        if exptime is not None:
            self.status.loc[mask, 'exptime'] = int(exptime)
        if n_files is not None:
            self.status.loc[mask, 'n_files'] = int(n_files)

    @staticmethod
    def _col(table, *names):
        """First of `names` present in an astropy table or DataFrame."""
        cols = table.colnames if hasattr(table, 'colnames') else list(table.columns)
        for n in names:
            if n in cols:
                return n
        return None

    def is_lightcurve(self, filename, subgroup):
        """Keep light-curve FITS files, both SPOC and the HLSP pipelines."""
        name = str(filename).lower()
        if not name.endswith('.fits'):
            return False
        if str(subgroup).upper() == 'LC':          # SPOC
            return True
        return 'lc' in name.split('_')[-1] or '_llc' in name   # HLSP naming

    # ---------- phase 1: resolve products (the expensive call, batched) ----------

    def resolve_products(self, batch_size=100, obsids=None):
        """One get_product_list call per batch of observations."""
        if obsids is None:
            obsids = [o for o in self.pending_rows()['obsid'].unique()
                      if o and o != 'nan']
        obsids = list(dict.fromkeys(str(o) for o in obsids))

        if not obsids:
            raise ValueError('no obsids to resolve: rebuild the catalogue so '
                             'the product table carries an obsid column')

        tables = []
        for i in tqdm(range(0, len(obsids), batch_size), desc='product lists'):
            batch = obsids[i:i + batch_size]
            tab = Observations.get_product_list(batch)
            if len(tab):
                tables.append(tab)

        if not tables:
            self._products, self._index = None, None
            return None

        prod = vstack(tables) if len(tables) > 1 else tables[0]

        fname_col = self._col(prod, 'productFilename', 'dataURI')
        sub_col = self._col(prod, 'productSubGroupDescription')
        parent_col = self._col(prod, 'parent_obsid', 'obsID', 'obs_id')

        keep = [self.is_lightcurve(prod[fname_col][k],
                                   prod[sub_col][k] if sub_col else None)
                for k in range(len(prod))]
        prod = prod[np.asarray(keep)]

        # map each product back to the (tic, prov, sector) that asked for it
        want = self.status[['tic', 'prov', 'sector', 'obsid']].copy()
        idx = pd.DataFrame({
            'row': np.arange(len(prod)),
            'obsid': [str(x) for x in prod[parent_col]],
        }).merge(want, on='obsid', how='inner')

        self._products, self._index = prod, idx
        print(f'{len(prod)} light-curve products for {len(obsids)} observations, '
              f'{idx["tic"].nunique()} targets matched')
        return idx

    # ---------- phase 2: transfer ----------

    def download_target(self, tic, prov):
        """Download the pending sectors of one target. Returns a verdict."""
        wanted = self.pending_sectors(tic, prov)
        if not wanted:
            return 'skip'

        if self._index is None:
            raise RuntimeError('call resolve_products() first')

        sel = self._index[(self._index['tic'] == tic) &
                          (self._index['prov'] == prov) &
                          (self._index['sector'].isin(wanted))]
        if sel.empty:
            self._mark(tic, prov, wanted, 'empty')
            self._save_status()
            return 'empty'

        target_dir = self.out_dir / f'tic_{tic}'
        target_dir.mkdir(parents=True, exist_ok=True)

        rows = self._products[sel['row'].to_numpy()]
        manifest = Observations.download_products(rows,
                                                  download_dir=str(target_dir))

        ok_sectors = sorted(set(int(s) for s in sel['sector']))
        exp = self.status.loc[(self.status['tic'] == tic) &
                              (self.status['prov'] == prov),
                              'want_exptime'].dropna()
        exp = int(exp.iloc[0]) if len(exp) else None

        n_ok = 0
        if manifest is not None:
            st_col = self._col(manifest, 'Status')
            n_ok = (sum(1 for s in manifest[st_col] if str(s).upper() == 'COMPLETE')
                    if st_col else len(manifest))

        self._mark(tic, prov, ok_sectors, 'ok', exptime=exp, n_files=n_ok)
        missing = [s for s in wanted if s not in ok_sectors]
        self._mark(tic, prov, missing, 'empty')
        self._save_status()

        return 'ok' if not missing else 'partial'

    # ---------- the whole list ----------

    def run(self, limit=None, batch_size=100, resolve=True):
        """Resolve every pending product, then transfer target by target."""
        if resolve or self._index is None:
            self.resolve_products(batch_size=batch_size)

        pend = self.pending_rows()
        pairs = pend[['tic', 'prov']].drop_duplicates().values.tolist()
        if limit is not None:
            pairs = pairs[:limit]

        for tic, prov in tqdm(pairs, desc='downloading'):
            try:
                self.download_target(int(tic), prov)
            except Exception as err:
                self._mark(int(tic), prov,
                           self.pending_sectors(int(tic), prov), 'error')
                self._save_status()
                print(f'TIC {tic} ({prov}) failed: {type(err).__name__}: {err}')

        return self.summary()

    def summary(self):
        """Per-sector counts, plus how many finished targets fell short."""
        counts = self.status['status'].value_counts().to_dict()

        short = attempted = 0
        for _, s in self.status.groupby(['tic', 'prov'])['status']:
            if (s == 'pending').any():
                continue                 # not finished yet: no verdict
            attempted += 1
            if (s == 'ok').sum() < len(s):
                short += 1

        print('sector counts:', counts)
        print(f'targets attempted: {attempted}, '
              f'of which short of the requested sectors: {short}')
        return counts

if __name__ == '__main__':

    N_TARGETS = 100           # how many targets to try this run
    SEED = 42                 # same sample every time you re-run
    BATCH = 200               # observations per get_product_list call

    # ---- 1. build the catalogue selections and the download list ----
    c = Criteria('test/tess_products_summary.csv',
                 'test/tess_products_long.csv')
    c.assign_all()

    out = c.write_output('test/download_list.csv')
    print(f'{len(out)} rows, {out.tic.nunique()} targets')
    print('\npipeline distribution:')
    print(c.info['prov'].value_counts(dropna=False).to_string())
    print('\ncadence distribution:')
    print(out['exptime'].value_counts(dropna=False).to_string())

    # ---- 2. set up the downloader ----
    d = Download('test/download_list.csv', out_dir='lightcurves', criteria=c)
    print(f'\nstatus table: {len(d.status)} sector-rows, '
          f'{int((d.status["status"] == "pending").sum())} pending')

    # ---- 3. pick N targets at random, keeping the pipeline mix ----
    pend = d.pending_rows()
    targets = pend[['tic', 'prov']].drop_duplicates()

    if len(targets) > N_TARGETS:
        targets = targets.sample(n=N_TARGETS, random_state=SEED)
    targets = targets.sort_values(['prov', 'tic']).reset_index(drop=True)

    print(f'\ntrying {len(targets)} targets:')
    print(targets['prov'].value_counts().to_string())

    rows = pend.merge(targets, on=['tic', 'prov'], how='inner')
    ids = [o for o in rows['obsid'].unique() if str(o) not in ('', 'nan')]
    print(f'{len(rows)} sector-rows, {len(ids)} observations to resolve')

    # ---- 4. resolve their products, then transfer ----
    def disk_gb(path='lightcurves'):
        return sum(f.stat().st_size for f in Path(path).rglob('*.fits')) / 1024 ** 3

    gb0 = disk_gb()

    t0 = time.perf_counter()
    d.resolve_products(obsids=ids, batch_size=BATCH)
    t_resolve = time.perf_counter() - t0
    print(f'resolve_products: {t_resolve / 60:.1f} min')

    t0 = time.perf_counter()
    for k, (tic, prov) in enumerate(targets[['tic', 'prov']].values.tolist(), 1):
        try:
            d.download_target(int(tic), prov)
        except Exception as err:
            d._mark(int(tic), prov, d.pending_sectors(int(tic), prov), 'error')
            d._save_status()
            print(f'TIC {tic} ({prov}) failed: {type(err).__name__}: {err}')

        if k % 50 == 0:
            el = time.perf_counter() - t0
            print(f'  {k}/{len(targets)} targets, {el / 60:.1f} min, '
                  f'{disk_gb() - gb0:.2f} GB, '
                  f'eta {el / k * (len(targets) - k) / 60:.0f} min')

    t_transfer = time.perf_counter() - t0
    gb = disk_gb() - gb0

    # ---- 5. what it cost ----
    print(f'\n=== {len(targets)} targets ===')
    print(f'resolve   : {t_resolve / 60:6.1f} min')
    print(f'transfer  : {t_transfer / 60:6.1f} min')
    print(f'disk      : {gb:6.2f} GB  ({1024 * gb / max(len(targets), 1):.1f} MB per target)')

    print()
    print(d.status[d.status['status'] != 'pending']
            .groupby(['prov', 'status']).size().to_string())
    d.summary()

    # ---- 6. extrapolate to the whole catalogue ----
    all_targets = d.status[['tic', 'prov']].drop_duplicates()
    scale = len(all_targets) / max(len(targets), 1)
    total_min = (t_resolve + t_transfer) / 60 * scale
    print(f'\nthe full list has {len(all_targets)} targets '
          f'({scale:.1f}x this run):')
    print(f'  serial    : {total_min / 60:.1f} h, {gb * scale:.1f} GB')
    print(f'  8 threads : {(t_resolve / 60 + t_transfer / 60 / 8) * scale / 60:.1f} h '
          f'(transfers only; resolve does not parallelise)')