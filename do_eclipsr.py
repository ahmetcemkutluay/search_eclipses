""" A pipeline to measure period, t0, eclipse widths and eclipse depths with ECLIPSR module from IJspeert et al. (2021).

    Aim : Calculate P, t0, depth and width for every available light curves. 

    Steps: Find folder, check if there is more than one light curve available. 
           Check which pipeline it belongs. 
           Do ECLIPSR accordingly. 
           Plot every light curve to its folder.
           Save results to results.csv with tic_id

"""

## Possible problems: ECLIPSR underestimate the eclipse depths. Calculate depths yourself from Gaussian + triangle fit.

import numpy as np #numerical operations
import pandas as pd #to see all the targets.
import matplotlib.pyplot as plt #visualise finalized analysis

import eclipsr as ecl #MAIN ANALYSIS
import time #measure computing time
import os
from pathlib import Path #filepath control
from astropy.io import fits #opening fits files

import lightkurve as lk #to visualize light curves initially

from scipy.signal import medfilt #to remove outliers
from scipy.optimize import curve_fit

from collections import Counter


class ECLIPSR():
    """ Select records and status file to pick targets, then search for lightcurves to make analysis.
    
        initialization reads the records and status file
        
        default records file type is parquet and status file type is .csv
    """

    def __init__(self, path_records, path_status, path_lc, path_output):
        self.path_recs = path_records
        self.path_status = path_status
        self.path_lc = Path(path_lc)
        self.path_out = path_output

        self.recs = pd.read_parquet(path_records, engine= 'pyarrow')
        self.status = pd.read_csv(path_status, delimiter=',')

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

    def eclipse_tables(self, tic_id, chosen, secs, result):

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
                print(f' ')

    def eclipsr_analysis(self, tic_id, verbose=True, savefig=True):
        FLUX_COLUMNS = ['SAP_FLUX', 'cal_aper_flux', 'KSPSAP_FLUX', 'DET_FLUX', 'FLUX']

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

            secs = '-'.join(str(l[3]) for l in sel)
            plot_all_dir = results_path / 'plots' / 'all'
            plot_sep_dir = results_path / 'plots' / 'sep'
            plot_all_dir.mkdir(parents=True, exist_ok=True)
            plot_sep_dir.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f'\nPipeline: {chosen}; {len(sel)} sector(s)')

            ## now make a plot of the complete lightcurve. 
            fig, axes = plt.subplots(len(sel), 1, figsize=(11, 2.2 * len(sel)), squeeze=False)
            for ax, (t,f,pipe,sec,fcol,tmag) in zip(axes[:,0], sel):
                ax.plot(t,f,'.', ms=1)
                tmag_s = f'{tmag:.2f}' if tmag is not None else 'N/A'
                ax.set_title(f'TIC {tic_id}, TESSmag = {tmag_s}, sector {sec}, {pipe}, cadence: {np.median(np.diff(t))*24*60:5.1f} min', fontsize= 9)
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

            gaps = np.diff(times)
            big = np.where(gaps > 1.0)[0]

            if verbose:
                print(f'{len(times)} pts, baseline [{times[-1] - times[0]:.1f} days\n')
                print(f'number of gaps > 1 day; {len(big)}, max : {gaps.max():.1f} days')

            fig2 = plt.figure(figsize=(12,3))
            plt.plot(times, signal, '.', ms=1)
            plt.xlabel('BTJD - 2,450,000')
            plt.ylabel('normalized flux')
            plt.title(f'TIC {tic_id}, {chosen}, TESSmag = {tmag_s}, cadence: {np.median(np.diff(times))*24*60:5.1f} min')
            plt.tight_layout()
            if savefig:
                fig2.savefig(plot_all_dir / f'{tic_id}_{chosen}_S{secs}_all.png', dpi=150, bbox_inches='tight')
            plt.close(fig2)
            
            times_ecl, flux_ecl = ecl.utility.ingest_signal(times, signal, tess_sectors=True)

            result = ecl.find_eclipses(times_ecl, flux_ecl, mode=2, tess_sectors=True)
            (t_0, period, score, sine_like, wide, n_kernel, width_stats, depth_stats, ecl_mid,
            widths, depths, ratios, added_snr, ecl_indices, flags_lrf, flags_pst) = result




            
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
            plot_all_dir.mkdir(parents=True, exist_ok=True)
            plot_sep_dir.mkdir(parents=True, exist_ok=True)


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



            

if __name__ == "__main__":
    path_records = Path('..') / 'platolops2_catalog' / 'records_v8.parquet'
    path_status  = Path('.') / 'status' / 'download_status_all.csv'
    path_lc      = Path('.') / 'lightcurves'
    path_out     = Path('.') / 'test_results'

    e = ECLIPSE(path_records, path_status, path_lc, path_out)

    print(f'tic column dtype: {e.tics.dtype}')
    print(f'first tic: {e.tics.iloc[0]!r}')

    N = 10
    test_tics = e.tics.drop_duplicates().head(N)
    print(f'Total records: {len(e.recs)} | To test: {len(test_tics)}\n')

    ok, failed, skipped = [], [], []

    for tic in test_tics:
        print(f'\n{"="*60}\nTIC {tic}')
        try:
            result = e.eclipsr_analysis(tic)
            if result is None:
                skipped.append(tic)
            else:
                ok.append(tic)
        except Exception as err:
            print(f'  FAILED: {type(err).__name__}: {err}')
            failed.append((tic, f'{type(err).__name__}: {err}'))

    print(f'\n{"="*60}')
    print(f'ok: {len(ok)} | skipped: {len(skipped)} | failed: {len(failed)}')
    for tic, msg in failed:
        print(f'  {tic}: {msg}')








