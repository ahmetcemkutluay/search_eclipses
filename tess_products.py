""" Check the available TESS products """

import numpy as np
import pandas as pd

from astroquery.mast import Observations
from astropy.table import vstack

import re
import time
from tqdm import tqdm
from pathlib import Path
import warnings

# '-s0001-s0036-' marks a multi-sector observation: SPOC's data-validation
# products, which span a sector range and hold no light curve.  MAST reports
# their sequence_number as the LAST sector of the span, so they look like an
# ordinary single-sector observation of a sector the target never sat in.
SECTOR_TOKEN = re.compile(r'-s(\d{4})')


class product():

    def __init__(self, rec_path, meas_path, file_out_path, fig_out_path, provs = ('SPOC', 'TGLC', 'QLP', 'T16')):
        self.rec_path = rec_path
        self.meas_path = meas_path

        self.file_out_path = file_out_path
        self.fig_out_path = fig_out_path

        self.provs = tuple(provs)


    def read_records(self, type='csv', delim='None'):

        file_path = Path(self.rec_path)

        if type == 'csv':
            delim=','
            file = pd.read_csv(file_path, delimiter=delim)
        elif type == 'parquet':
            file = pd.read_parquet(file_path, engine='pyarrow')
        else:
            raise ValueError(f"Unsupported file type: {type}. Please try with \"parquet\" or \"csv\" files.")

        return file


    def read_measurements(self, type='csv', delim='None'):
        file_path = Path(self.meas_path)

        if type == 'csv':
            delim=','
            file = pd.read_csv(file_path, delimiter=delim)
        elif type == 'parquet':
            file = pd.read_parquet(file_path, engine='pyarrow')
        else:
            raise ValueError(f"Unsupported file type: {type}. Please try with \"parquet\" or \"csv\" files.")

        return file


    def get_tics(self, col_name='tic'):
        file = self.read_records()

        return file[col_name]

    def query(self, tics=None, chunk=5000):
        if tics is None:
            tics = self.get_tics()
        tics = sorted(set(str(t) for t in tics))

        tables = []
        for i in tqdm(range(0, len(tics), chunk)):
            batch = tics[i:i+chunk]
            tab = Observations.query_criteria(target_name=batch, provenance_name = list(self.provs))

            if len(tab) > 0:
                tables.append(tab)
                print(f"{i+len(batch)} / {len(tics)}")

        return vstack(tables) if tables else None

    # ------------------------------------------------------------------
    # product inventory: one row per (tic, prov, sector, exptime).
    # Everything else below is derived from it, so the summary columns
    # cannot drift apart, and no selection decision is taken here.
    # ------------------------------------------------------------------

    @staticmethod
    def is_single_sector(obs_id):
        """False for a multi-sector (data-validation) observation."""
        return len(set(SECTOR_TOKEN.findall(str(obs_id)))) <= 1

    def drop_multisector(self, df):
        """Remove observations that span a sector range."""
        col = 'obs_id' if 'obs_id' in df.columns else None
        if col is None:
            warnings.warn("no 'obs_id' column: cannot spot multi-sector "
                          "data-validation observations, which will be "
                          "mistaken for light curves in their last sector")
            return df

        multi = ~df[col].map(self.is_single_sector)
        if multi.any():
            msg = (f'{int(multi.sum())} multi-sector observations dropped '
                   f'(data-validation products, no light curves)')
            if 'calib_level' in df.columns:
                lv = sorted(set(df.loc[multi, 'calib_level'].dropna()))
                msg += f'; their calib_level values: {lv}'
            warnings.warn(msg)

        return df[~multi]

    def product_table(self, obs, tic_col='tic'):
        """tic / prov / sector / exptime, one row per distinct product."""
        df = obs.to_pandas() if hasattr(obs, 'to_pandas') else obs.copy()
        df = df[df['provenance_name'].isin(list(self.provs))].copy()
        df = self.drop_multisector(df)

        df[tic_col] = pd.to_numeric(df['target_name'], errors='coerce')
        df['sector'] = pd.to_numeric(df['sequence_number'], errors='coerce')
        if 't_exptime' in df.columns:
            df['exptime'] = pd.to_numeric(df['t_exptime'], errors='coerce')
        else:
            warnings.warn("no 't_exptime' column in the query result; "
                          "cadence selection will have nothing to go on")
            df['exptime'] = np.nan

        df = df.dropna(subset=[tic_col, 'sector'])
        df = df.astype({tic_col: 'int64', 'sector': 'int64'})
        df = df.rename(columns={'provenance_name': 'prov'})

        # keep every distinct exposure time: one sector can hold both a 20 s
        # and a 120 s product, and collapsing them would hide the usable one
        df['exptime'] = df['exptime'].round().astype('Int64')

        # keep the MAST observation id: the download stage can then call
        # get_product_list directly and skip re-querying altogether
        cols = [tic_col, 'prov', 'sector', 'exptime']
        for cand in ('obsid', 'obs_id'):
            if cand in df.columns:
                df['obsid'] = df[cand].astype(str)
                cols.append('obsid')
                break
        else:
            warnings.warn("no 'obsid' column in the query result; the "
                          "download stage will have to re-query MAST")

        return (df[cols]
                .drop_duplicates()
                .sort_values([tic_col, 'prov', 'sector', 'exptime'])
                .reset_index(drop=True))

    def count_sectors(self, prod, tic_col='tic'):
        """distinct sectors across all pipelines"""
        return (prod.groupby(tic_col)['sector'].nunique()
                    .rename('n_sectors').reset_index())

    def count_prov_sectors(self, prod, tic_col='tic'):
        """number of distinct sectors per target per pipeline"""
        wide = (prod.groupby([tic_col, 'prov'])['sector'].nunique()
                    .unstack('prov').reindex(columns=list(self.provs)))

        return wide.fillna(0).astype('Int64').add_prefix('n_sec_').reset_index()

    def sector_lists(self, prod, tic_col='tic', sep=','):
        """per-pipeline sector strings: sec_SPOC, sec_TGLC, ..."""
        wide = (prod.groupby([tic_col, 'prov'])['sector']
                    .apply(lambda s: sep.join(str(x) for x in sorted(set(s))))
                    .unstack('prov').reindex(columns=list(self.provs)))

        return wide.fillna('').add_prefix('sec_').reset_index()

    def exptime_lists(self, prod, tic_col='tic', sep=','):
        """per-pipeline exposure times present: exp_SPOC, exp_TGLC, ..."""
        wide = (prod.dropna(subset=['exptime'])
                    .groupby([tic_col, 'prov'])['exptime']
                    .apply(lambda s: sep.join(str(int(x)) for x in sorted(set(s))))
                    .unstack('prov').reindex(columns=list(self.provs)))

        return wide.fillna('').add_prefix('exp_').reset_index()

    def count_products(self, obs, tic_col='tic'):
        """number of MAST records per target per pipeline"""
        all_obs = obs.to_pandas() if hasattr(obs, 'to_pandas') else obs.copy()
        all_obs['target_name'] = all_obs['target_name'].astype(int)
        all_obs = all_obs[all_obs['provenance_name'].isin(list(self.provs))]

        counts = pd.crosstab(all_obs['target_name'], all_obs['provenance_name'])
        counts = counts.reindex(columns=list(self.provs), fill_value=0)

        return counts.add_prefix('n_prod_').reset_index().rename(columns={'target_name': tic_col})


    def period_flags(self, threshold=10.0, n_cycles=3, sector_len=27.4, max_sectors=5,
                     default_sectors=1, tics=None, tic_col='tic', id_col='id', rec_id_col='record_id',
                     qty='period'):

        rec = self.read_records()[[id_col, tic_col]].copy()
        meas = self.read_measurements()

        rec[tic_col] = pd.to_numeric(rec[tic_col], errors='coerce')
        rec = rec.dropna(subset=[tic_col])
        rec[tic_col] = rec[tic_col].astype('int64')

        if tics is not None:
            keys = pd.to_numeric(pd.Series(list(tics)), errors='coerce').dropna().astype('int64')
            rec = rec[rec[tic_col].isin(set(keys))]

        n_dup = rec[tic_col].duplicated().sum()
        if n_dup:
            warnings.warn(
            f"{n_dup} TICs map to more than one record; "
            f"their measurements will be pooled.")


        per = meas.loc[meas['quantity'] == qty, [rec_id_col, 'value']].copy()
        per['value'] = pd.to_numeric(per['value'], errors='coerce')
        per = per[per['value'] > 0].dropna(subset=['value'])

        if per.empty:
            raise ValueError(
            f"No measurements found for quantity '{qty}'. "
            f"Available values: {sorted(meas['quantity'].unique())[:20]}")

        per = per.merge(rec, left_on=rec_id_col, right_on=id_col, how='inner')

        agg = (per.groupby(tic_col)['value']
          .agg(n_period='size', period_min='min', period_max='max')
          .reset_index())

        # "all measurements below threshold" is equivalent to "largest one below"
        agg['short_period'] = (agg['period_max'] <= threshold).astype('Int64')

        needed = np.ceil(n_cycles * agg['period_max'] / sector_len)
        agg['n_sec_needed'] = needed.clip(1, max_sectors).astype('Int64')

        # targets without a period keep a NaN flag but still get a default budget
        out = (rec[[tic_col]].drop_duplicates().merge(agg, on=tic_col, how='left'))

        out['n_period'] = out['n_period'].fillna(0).astype('Int64')
        out['n_sec_needed'] = out['n_sec_needed'].fillna(default_sectors).astype('Int64')

        return out

    @staticmethod
    def _to_tic(values):
        """any iterable of TIC-like values -> sorted unique int64 Series"""
        s = pd.to_numeric(pd.Series(list(values)), errors='coerce').dropna()
        return pd.Series(sorted(set(s.astype('int64'))), name='tic')

    def resulting(self, obs, tics=None, mag_col='tess_mag', tic_col='tic', save=True):

        prod = self.product_table(obs, tic_col=tic_col)

        sectors      = self.count_sectors(prod, tic_col=tic_col)
        prov_sectors = self.count_prov_sectors(prod, tic_col=tic_col)
        sec_cols     = self.sector_lists(prod, tic_col=tic_col)
        exp_cols     = self.exptime_lists(prod, tic_col=tic_col)
        prod_counts  = self.count_products(obs, tic_col=tic_col)
        periods      = self.period_flags(threshold=10, n_cycles=3, tics=tics)

        # master index: sorgulanan hedefler (obs'ta ürünü olmayanlar da kalsın)
        if tics is None:
            tics = obs.to_pandas()['target_name']
        out = self._to_tic(tics).to_frame(name=tic_col)

        for df in (sectors, prov_sectors, prod_counts, periods, sec_cols, exp_cols):
            out = out.merge(df, on=tic_col, how='left')

        # katalog magnitüdü
        cat = self.read_records()[[tic_col, mag_col]].copy()
        cat[tic_col] = pd.to_numeric(cat[tic_col], errors='coerce')
        cat = cat.dropna(subset=[tic_col])
        cat[tic_col] = cat[tic_col].astype('int64')
        out = out.merge(cat.drop_duplicates(tic_col), on=tic_col, how='left')

        # sayımlar: eşleşmeyen = 0. period_min/max NaN kalır (ölçüm yok demek)
        count_cols = [c for c in out.columns
                      if c.startswith(('n_sectors', 'n_prod_', 'n_sec_', 'n_period'))]
        out[count_cols] = out[count_cols].fillna(0).astype('Int64')

        list_cols = [c for c in out.columns if c.startswith(('sec_', 'exp_'))]
        out[list_cols] = out[list_cols].fillna('')

        assert len(out) == len(self._to_tic(tics)), "row count drifted during merge"
        assert (out['n_sec_needed'] >= 1).all(), \
            f"n_sec_needed == 0: {out.loc[out['n_sec_needed'] < 1, 'tic'].tolist()}"

        if save:
            Path(self.file_out_path).mkdir(parents=True, exist_ok=True)
            out.to_csv(Path(self.file_out_path) / 'tess_products_summary.csv', index=False)
            prod.to_csv(Path(self.file_out_path) / 'tess_products_long.csv', index=False)

        return out


if __name__ == '__main__':

    rec_path = Path('.') / 'example_records_list.csv'
    meas_path = Path('.') / 'example_measurements_list.csv'
    fig_out_path = Path('.') / 'figures'
    file_out_path = Path('.') / 'test_files'

    products = product(rec_path, meas_path, file_out_path, fig_out_path)
    tics = products.get_tics()

    sub = tics[:]

    t0 = time.perf_counter()
    obs = products.query(tics=sub)
    dt = time.perf_counter() - t0

    print(dt)

    print(products.resulting(obs, tics=sub))

    ### Final notes

    ### 1500 objects takes ~30 second to gather. Since it is a one time job; It is estimated for 180,000 objects ->