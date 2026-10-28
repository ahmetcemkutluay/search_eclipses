""" A pipeline to measure period, t0, eclipse widths and eclipse depths with ECLIPSR module from IJspeert et al. (2021).

    Aim : Measure P, t0, depth and width for every available TESS light curves. 

    Steps: Find folder, check if there is more than one light curve available. 
           Check which pipeline it belongs. 
           Do ECLIPSR accordingly. 
           Plot every light curve to its folder.
           Save results.

    Output: 
        - results.csv -- one row per target (scalars + width/depth statistics)
        - eclipses.csv -- one row per eclipse (long format, tics)
        - failed.csv -- target that raised error
        - plots/sep/ -- one plot panel per sector / target
        - plots/all/ -- stitched light curve
        - plots/results/ -- phase-folded light curve
"""

## Possible problems: ECLIPSR underestimate the eclipse depths. We will solve this problem in our two Gaussian + cosine fitting.

import numpy as np #numerical operations
import pandas as pd #to see all the targets.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt #visualise finalized analysis

import eclipsr as ecl #MAIN ANALYSIS
import time #measure computing time
from pathlib import Path #filepath control
from astropy.io import fits #opening fits files

import traceback
import sys
from datetime import datetime
from collections import Counter


class Tee:
    def __init__(self, path, mode='w', stderr=True):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, mode, buffering=1)
        self.stdout , self.stderr = sys.stdout , sys.stderr
        self.also_stderr = stderr

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self
        if self.also_stderr:
            sys.stderr = self
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.file.close()


class ECLIPSR():
    """ Select records and status file to pick targets, then search for lightcurves to make analysis.
    
        initialization reads the records and status file
    """

    @staticmethod
    def _load_table(path, **kwargs):
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path, **kwargs)
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(path, engine="pyarrow")
        raise ValueError(f"unsupported file format: {ext}")

    def __init__(self, path_records, path_status, path_lc, path_output):
        self.path_recs = path_records
        self.path_status = path_status
        self.path_lc = Path(path_lc)
        self.path_out = Path(path_output)

        self.recs = self._load_table(self.path_recs)
        self.status = self._load_table(self.path_status)

        self.tics = self.recs['tic'].astype(str)

    def if_file_exists(self, tic_id):
        """make a check to see which tic_ids exists in lightcurve folder."""
        folder_name = "tic_" + tic_id  

        if (Path(self.path_lc) / folder_name).is_dir():
            print(f"Folder {folder_name} exists in {self.path_lc}")
            return True
        else:
            return False

    def if_multiple(self, tic_id):
        """check if the object will be analysed with single or multiple sector data.
        
        Returns True for fits_files > 1 and False for fits_files <= 1"""
        if not self.if_file_exists(tic_id):
            return False

        folder = Path(self.path_lc) / ("tic_" + tic_id)
        fits_files = list(folder.rglob("*.fits"))

        print(f"{len(fits_files)} fits files found for tic_{tic_id}")
        return len(fits_files) > 1

    def results_make_file(self, path):
        """ creating results file if not exists"""
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def pipelines_of(self, tic_id):
        """check what kind of pipelines is processing the given target"""
        folder = Path(self.path_lc) / ("tic_" + tic_id)
        files = list(folder.rglob("*.fits"))
        counts = Counter(self.which_pipeline(f) for f in files)
        print(f"tic_{tic_id}: {dict(counts)}")
        return counts

    def which_pipeline(self, fits_path):
        name = Path(fits_path).name.lower()
        if name.startswith("hlsp_"):
            return name.split("_")[1].upper()   # QLP, TGLC, T16, TESS-SPOC
        return "SPOC"

    def get_fits(self, tic_id):
        """ return all fits files for given tic, empty list if folder missing"""
        folder = Path(self.path_lc) / ("tic_" + tic_id)
        if not folder.is_dir():
            return []
        return sorted(folder.rglob("*.fits"))

    def read_lc(self, path):
        """ read single fits -> (time, normalized flux, pipeline, sector, fcol, tessmag) """
        FLUX_COLUMNS = ['SAP_FLUX', 'cal_aper_flux', 'KSPSAP_FLUX', 'DET_FLUX', 'FLUX']

        with fits.open(path) as hdul:
            h0, d = hdul[0].header, hdul[1].data
            cols = hdul[1].columns.names

            fcol = next((c for c in FLUX_COLUMNS if c in cols), None)
            if fcol is None:
                raise ValueError(f"No flux column found for given set of columns.")

            t = np.asarray(d['TIME'], dtype=float)
            f = np.asarray(d[fcol], dtype=float)
            q = np.asarray(d['QUALITY'] if 'QUALITY' in cols else d['TGLC_flags'] if 'TGLC_flags' in cols else np.zeros_like(t)
                           , dtype=int)

            #finite time and flux, positive flux and quality flags applied.
            good = np.isfinite(t) & np.isfinite(f) & (f > 0) & (q == 0)

            t, f = t[good], f[good]

            f = f / np.median(f) #normalize

            pipe = ('SPOC' if h0.get('ORIGIN') == 'NASA/Ames' 
                    else 'TGLC' if h0.get('ORIGIN') == 'UCSB/TGLC'
                    else 'QLP' if h0.get('ORIGIN') == 'MIT/QLP'
                    else 'T16')
            sector = h0.get('SECTOR')
            tessmag = h0.get('TESSMAG')

        return t, f, pipe, sector, fcol, tessmag

    def phase(self, time, flux, period, t_0):
        """ return a phased light curve between -0.25 and 0.75, so that both primary and secondary can be seen."""
        phase = ((time - t_0) / period + 0.25) % 1 - 0.25
        o = np.argsort(phase)
        return phase[o], flux[o]

    def eclipse_tables(self, tic_id, chosen, secs, result): #this will produce the eclipses.csv in the end
        (t_0, period, score, sine_like, wide, n_kernel, width_stats, depth_stats,
         ecl_mid, widths, depths, ratios, added_snr, ecl_indices,
         flags_lrf, flags_pst) = result

        ecl_mid = np.asarray(ecl_mid, dtype=float).ravel()
        n = ecl_mid.size
        if n == 0:
            return None

        def col(arr, dtype=float):
            a = np.asarray(arr).ravel()
            if a.size != n:
                print(f' WARNING tic {tic_id} expected {n} values, got {a.size}')
                return np.full(n, np.nan)
            return a.astype(dtype)

        df = pd.DataFrame({
            'tic': str(tic_id),
            'pipeline': chosen,
            'sectors': secs,
            'ecl_no': np.arange(n),
            'ecl_mid': ecl_mid,
            'width': col(widths),
            'depth': col(depths),
            'ratio': col(ratios),
            'added_snr': col(added_snr),
            'flag_lrf': col(flags_lrf, dtype=int),
            'flag_pst': col(flags_pst, dtype=int),
        })

        df['phase'] = ((ecl_mid - t_0) / period + 0.25) % 1 - 0.25

        ei = np.asarray(ecl_indices)
        if ei.ndim == 2 and ei.shape == (n,2):
            df['idx_start'], df['idx_end'] = ei[:,0], ei[:,1]

        return df

    def summary_row(self, tic_id, meta, result):
        (t_0, period, score, sine_like, wide, n_kernel, width_stats, depth_stats,
         ecl_mid, widths, depths, ratios, added_snr, ecl_indices,
         flags_lrf, flags_pst) = result

        ws = np.atleast_2d(np.asarray(width_stats, dtype=float))
        ds = np.atleast_2d(np.asarray(depth_stats, dtype=float))

        def cell(m, i, j):
            return m[i, j] if m.shape[0] > i and m.shape[1] > j else np.nan

        pst = np.asarray(flags_pst).ravel()

        row = dict(meta)                       # pipeline, sectors, tessmag, ...

        row.update({
            'tic': str(tic_id),
            't_0': float(t_0),
            'period': float(period),
            'score': float(score),
            'sine_like': bool(sine_like),
            'wide': bool(wide),
            'n_eclipses': int(np.size(ecl_mid)),
            'n_prim': int(np.sum(pst == 1)),
            'n_sec': int(np.sum(pst == 2)),
            'width_prim':  cell(ws, 0, 0), 'width_prim_err': cell(ws, 0, 1),
            'width_sec':   cell(ws, 1, 0), 'width_sec_err':  cell(ws, 1, 1),
            'depth_prim':  cell(ds, 0, 0), 'depth_prim_err': cell(ds, 0, 1),
            'depth_sec':   cell(ds, 1, 0), 'depth_sec_err':  cell(ds, 1, 1),
        })

        return row


    def _append(self, df, filename):
        """ append rows; write the header only when the file is created """
        out = self.results_make_file(self.path_out) / filename
        df.to_csv(out, mode='a', index=False, header=not out.exists())
        return out

    def save_records(self, res):
        """ write one analysis result to disk.

        Append per target rather than collecting everything and writing at the end:
        a crash at target 1500 then costs you one target instead of 1499.
        The price is that the column order is frozen by the first row.
        """
        if res is None:
            return
        self._append(pd.DataFrame([res['summary']]), 'results.csv')
        if res.get('eclipses') is not None:
            self._append(res['eclipses'], 'eclipses.csv')


    def save_failed(self, failed, filename='failed.csv'):
        """ keep the message, not just the type """
        if failed:
            pd.DataFrame(failed, columns=['tic', 'error']).to_csv(self.results_make_file(self.path_out) / filename, index=False)

    def reset_outputs(self, filenames=('results.csv', 'eclipses.csv', 'failed.csv')):
        """ start a test run from scratch -- append mode otherwise doubles every file """
        self.results_make_file(self.path_out)
        for fn in filenames:
            p = Path(self.path_out) / fn
            if p.exists():
                p.unlink()
    
    def eclipsr_analysis(self, tic_id, verbose=True, savefig=True):
        results_path = self.results_make_file(self.path_out)

        if self.if_multiple(tic_id): #targets with len(fits_files) > 1. 
            PATH_LC = self.path_lc
            folder = PATH_LC / ('tic_' + tic_id)
            fits_files = sorted(folder.rglob('*.fits')) #find existing fits files using rglob

            if verbose:
                print(f"{len(fits_files)} file(s) found.\n")
                for f in fits_files:
                    print(f"{f.relative_to(folder)}")

            lcs = [self.read_lc(f) for f in fits_files]
            if verbose:
                for (t,f,pipe,sec,fcol,tmag), j in zip (lcs, fits_files):
                    print(f"{pipe:8s} SECTOR{sec:<4} {len(t):6d} points; cad~{np.median(np.diff(t))*24*60:5.1f} min [{fcol}]")

            PRIORITY = ['SPOC', 'QLP', 'TGLC', 'T16']

            available = {pipe for _, _, pipe, _, _, _ in lcs}
            chosen = next((p for p in PRIORITY if p in available), sorted(available)[0])

            sel = [l for l in lcs if l[2] == chosen]
            tmag = sel[0][5]

            secs = '-'.join(str(l[3]) for l in sel)
            plot_all_dir = results_path / 'plots' / 'all'
            plot_sep_dir = results_path / 'plots' / 'sep'
            plot_results = results_path / 'plots' / 'results'
            plot_all_dir.mkdir(parents=True, exist_ok=True)
            plot_sep_dir.mkdir(parents=True, exist_ok=True)
            plot_results.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f'\nPipeline: {chosen}; {len(sel)} sector(s)')

            ## now make a plot of the complete lightcurve. 
            fig, axes = plt.subplots(len(sel), 1, figsize=(11, 2.2 * len(sel)), squeeze=False)
            for ax, (t,f,pipe,sec,fcol,_tmag) in zip(axes[:,0], sel):
                ax.plot(t,f,'.', ms=1)
                ax.set_title(f'TIC {tic_id}, TESSmag = {tmag}, sector {sec}, {pipe}, cadence: {np.median(np.diff(t))*24*60:5.1f} min', fontsize= 9)
                ax.set_ylabel('normalized flux')
            axes[-1, 0].set_xlabel('BTJD - 2,450,000')
            plt.tight_layout()
            if savefig:
                fig.savefig(plot_sep_dir / f'{tic_id}_{chosen}_S{secs}_panels.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

            ## now concatenate the light curves.
            times, signal = np.concatenate([l[0] for l in sel]), np.concatenate([l[1] for l in sel])
            order = np.argsort(times)
            times, signal = times[order], signal[order]
            cad = np.median(np.diff(times)) * 24 * 60

            gaps = np.diff(times)
            big = np.where(gaps > 1.0)[0]

            if verbose:
                print(f'{len(times)} pts, baseline [{times[-1] - times[0]:.1f} days\n')
                print(f'number of gaps > 1 day; {len(big)}, max : {gaps.max():.1f} days')

            fig2 = plt.figure(figsize=(12,3))
            plt.plot(times, signal, '.', ms=1)
            plt.xlabel('BTJD - 2,450,000')
            plt.ylabel('normalized flux')
            plt.title(f'TIC {tic_id}, {chosen}, TESSmag = {tmag}, cadence: {np.median(np.diff(times))*24*60:5.1f} min')
            plt.tight_layout()
            if savefig:
                fig2.savefig(plot_all_dir / f'{tic_id}_{chosen}_S{secs}_all.png', dpi=150, bbox_inches='tight')
            plt.close(fig2)
            
            times_ecl, flux_ecl = ecl.utility.ingest_signal(times, signal, tess_sectors=True)
            result = ecl.find_eclipses(times_ecl, flux_ecl, mode=2, tess_sectors=True)

            (t_0, period, score, sine_like, wide, n_kernel, width_stats, depth_stats, ecl_mid,
            widths, depths, ratios, added_snr, ecl_indices, flags_lrf, flags_pst) = result

            if not np.isfinite(period) or period <= 0:
                if verbose:
                    print(f'No valid period (P = {period}) -- skipped')
                return None

            meta = {'pipeline': chosen,
                    'sectors': secs, 
                    'n_sectors': len(sel),
                    'tess_mag': tmag,
                    'cadence_min': cad,
                    'n_points': len(times),
                    'baseline_d': times[-1] - times[0],
                    'n_gaps': len(big),
                    }

            row = self.summary_row(tic_id, meta, result)
            df_ecl = self.eclipse_tables(tic_id, chosen, secs, result)

            if savefig:  
                ph, f1 = self.phase(times, signal, period, t_0)
                fig = plt.figure(figsize=(8,3.2))
                plt.plot(ph, f1, '.', ms=1, alpha=0.5)
                plt.xlabel('Phase')
                plt.ylabel('normalized flux')
                plt.title(f'TIC {tic_id}, {chosen}, P = {period:.5f}, t0= {t_0:.6f}, score= {score:.3f}')
                plt.tight_layout()
                fig.savefig(plot_results/ f'{tic_id}_{chosen}_S{secs}_phase.png', dpi=150, bbox_inches='tight')
                plt.close(fig)

            return {'summary': row, 'eclipses': df_ecl, 'result': result}
        
        else: #target with len(fits_files) = 1
            PATH_LC = self.path_lc
            folder = PATH_LC / ('tic_' + tic_id)
            fits_files = sorted(folder.rglob('*.fits')) #find existing fits file

            if not fits_files:
                raise FileNotFoundError(f'TIC {tic_id}: fits file cannot be found under {folder}.\n')

            f_path = fits_files[0]

            if verbose:
                print(f"{len(fits_files)} file(s) found.\n")

            times, signal, chosen, sec, fcol, tmag = self.read_lc(f_path)

            order = np.argsort(times)
            times, signal = times[order], signal[order]
            cad = np.median(np.diff(times)) * 24 * 60
            tmag_s = f'{tmag:.2f}' if tmag is not None else 'N/A'

            if verbose:
                print(f"{chosen:8s} SECTOR{sec:<4} {len(times):6d} points; "
                      f"cad~{cad:5.1f} min [{fcol}]")
                print(f'\nPipeline: {chosen}; 1 sector')


            secs = str(sec)
            plot_all_dir = results_path / 'plots' / 'all'
            plot_sep_dir = results_path / 'plots' / 'sep'
            plot_results = results_path / 'plots' / 'results'
            plot_all_dir.mkdir(parents=True, exist_ok=True)
            plot_sep_dir.mkdir(parents=True, exist_ok=True)
            plot_results.mkdir(parents=True, exist_ok=True) 

            fig, axes = plt.subplots(1,1, figsize=(11, 2.2), squeeze=False)
            ax = axes[0,0]
            ax.plot(times, signal, '.', ms=1)
            ax.set_title(f'TIC {tic_id}, TESSmag = {tmag_s}, sector {sec}, {chosen}, '
                         f'cadence: {cad:5.1f} min', fontsize=9)
            ax.set_ylabel('normalized flux')
            ax.set_xlabel('BTJD - 2,450,000')
            plt.tight_layout()
            if savefig:
                fig.savefig(plot_sep_dir / f'{tic_id}_{chosen}_S{secs}_panels.png',
                            dpi=150, bbox_inches='tight')

            plt.close(fig)

            ## gap statistics (one sector usually contains only downlink gap)
            gaps = np.diff(times)
            big = np.where(gaps > 1.0)[0]

            if verbose:
                print(f'{len(times)} pts, baseline {times[-1] - times[0]:.1f} days\n')
                print(f'number of gaps > 1 day: {len(big)}, max : {gaps.max():.1f} days')

            ## 'all' plot -- normally it is not necessary to save these but the structure requires all objects to be recorded
            ## in one folder
            fig2 = plt.figure(figsize=(12, 3))
            plt.plot(times, signal, '.', ms=1)
            plt.xlabel('BTJD - 2,450,000')
            plt.ylabel('normalized flux')
            plt.title(f'TIC {tic_id}, {chosen}, TESSmag = {tmag_s}, cadence: {cad:5.1f} min')
            plt.tight_layout()
            if savefig:
                fig2.savefig(plot_all_dir / f'{tic_id}_{chosen}_S{secs}_all.png',
                             dpi=150, bbox_inches='tight')
            plt.close(fig2)

            ##ECLIPSR
            times_ecl, flux_ecl = ecl.utility.ingest_signal(times, signal, tess_sectors=True)
            result = ecl.find_eclipses(times_ecl, flux_ecl, mode=2, tess_sectors=True)

            (t_0, period, score, sine_like, wide, n_kernel, width_stats, depth_stats, ecl_mid, 
            widths, depths, ratios, added_snr, ecl_indices, flags_lrf, flags_pst) = result

            if not np.isfinite(period) or period <= 0:
                if verbose:
                    print(f'No valid period (P = {period}) -- skipped')
                return None

            meta = {'pipeline': chosen,
                    'sectors': secs, 
                    'n_sectors': 1,
                    'tess_mag': tmag,
                    'cadence_min': cad,
                    'n_points': len(times),
                    'baseline_d': times[-1] - times[0],
                    'n_gaps': len(big),
                    }

            row = self.summary_row(tic_id, meta, result)
            df_ecl = self.eclipse_tables(tic_id, chosen, secs, result)

            if savefig:  
                ph, f1 = self.phase(times, signal, period, t_0)
                fig = plt.figure(figsize=(8,3.2))
                plt.plot(ph, f1, '.', ms=1, alpha=0.5)
                plt.xlabel('Phase')
                plt.ylabel('normalized flux')
                plt.title(f'TIC {tic_id}, {chosen}, P = {period:.5f}, t0= {t_0:.6f}, score={score:.3f}')
                plt.tight_layout()
                fig.savefig(plot_results / f'{tic_id}_{chosen}_S{secs}_phase.png', dpi=150, bbox_inches='tight')
                plt.close(fig)

            return {'summary': row, 'eclipses': df_ecl, 'result': result}



if __name__ == "__main__":

    path_records = Path('.') / 'example_records_list.csv'
    path_status  = Path('.') / 'status' / 'download_status_all.csv'
    path_lc      = Path('.') / 'test_files' / 'lightcurves'
    path_out     = Path('.') / 'test_results'

    log_file = path_out / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with Tee(log_file):
        print(f'log -> {log_file}')
    
        e = ECLIPSR(path_records, path_status, path_lc, path_out) #initialize ECLIPSR
    
        print(f'tic column dtype: {e.tics.dtype}')
        print(f'first tic: {e.tics.iloc[0]!r}')
    
        e.reset_outputs()
    
        test_tics = [t for t in e.tics.drop_duplicates() if e.if_file_exists(t)]
    
        n_multi = sum(1 for t in test_tics if e.if_multiple(t))
        print(f'{len(test_tics)} target,  ({n_multi} multiple sectors, {len(test_tics)-n_multi} single sectors)')
        print(f'without a file: {e.tics.nunique() - len(test_tics)}\n')
    
        ok, skipped, failed = [], [], []
    
        try:
            for tic in test_tics:
                print(f'\n{"="*60}\nTIC {tic}')
                try:
                    res = e.eclipsr_analysis(tic)
                    if res is None:
                        skipped.append(tic)
                        continue
                    e.save_records(res)
                    ok.append(tic)
                except Exception as err:
                    print(f'  FAILED: {type(err).__name__}: {err}')
                    traceback.print_exc()
                    failed.append((tic, f'{type(err).__name__}: {err}'))
        finally:
            e.save_failed(failed)     
    
        print(f'\n{"="*60}')
        print(f'ok: {len(ok)} | skipped: {len(skipped)} | failed: {len(failed)}')
        for tic, msg in failed:
            print(f'  {tic}: {msg}')
    
        if ok:
            res_df = pd.read_csv(path_out / 'results.csv', dtype={'tic': str})
            ecl_df = pd.read_csv(path_out / 'eclipses.csv', dtype={'tic': str})
        
            print(f'\nresults.csv : {len(res_df)} satir')
            print(f'eclipses.csv: {len(ecl_df)} satir')
        
            cnt = (ecl_df.groupby('tic').size().rename('rows').to_frame()
                       .join(res_df.set_index('tic')['n_eclipses']))
            bad = cnt[cnt['rows'] != cnt['n_eclipses']]
            if len(bad):
                print(bad)