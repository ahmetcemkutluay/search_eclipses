<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/logo-dark.png">
    <img alt="search_eclipses" src="assets/logo-light.svg" width="440">
  </picture>
</p>

Eclipse detection in TESS light curves using ECLIPSR and combinations of two-Gaussian + cosine models.

Given a list of candidate eclipsing binaries, the pipeline runs in three stages:

1. **Photometry retrieval.** TESS light curves are collected from the
   [SPOC](https://archive.stsci.edu/hlsp/tess-spoc/),
   [TGLC](https://archive.stsci.edu/hlsp/tglc),
   [QLP](https://archive.stsci.edu/hlsp/qlp) and
   [T16](https://archive.stsci.edu/hlsp/t16) pipelines, with one pipeline and
   cadence selected per target according to availability.
2. **Eclipse detection.** [ECLIPSR](https://github.com/LucIJspeert/eclipsr)
   flags candidate eclipsing systems and measures their mid-eclipse times and
   orbital periods.
3. **Eclipse characterisation.** Different combinations of two-Gaussian plus cosine model are fitted to
   the phase-folded light curves to describe the eclipse profiles and determine eclipse depths & widths

The pipeline was developed for the PLATO LOPS2 eclipsing binary catalogue (Kutluay & Southworth, in prep.), where it was applied to approximately
145,000 candidates compiled from several surveys. Two example target lists, `example_records_list.csv` and `example_measurements_list.csv`, are included so
the pipeline can be run end to end without the full input catalogue.


## Installation

```bash
git clone https://github.com/ahmetcemkutluay/search_eclipses.git
cd search_eclipses 
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Usage

You can start using the pipeline writing

```bash
python tess_products.py
```

This writes two CSV files:

| File | Contents |
| --- | --- |
| `tess_products_summary.csv` | One row per target, summarising what is available across all pipelines: sector counts per pipeline, the sectors themselves (`sec_SPOC`, `sec_QLP`, ...), available exposure times, and the period range (`period_min`, `period_max`)  | 
| `tess_products_long.csv` | One row per available observation: `tic`, `prov` (pipeline), `sector`, `exptime`, `obsid`|

Then build the download list and fetch the light curves:

```bash
python download_main.py --list-only   # write download_list.csv and stop
python download_main.py               # build the list (if absent) and download
```

`--list-only` lets you inspect the pipeline and cadence distribution before committing to the transfer. The download runs in `N` parallel processes (set at the top of `download_main.py`), sharded by `tic % N` so no two processes touch the same target. It is safe to interrupt: each process writes its progress to `status/download_status_<n>.csv` and resumes from there on the next run.

Light curves land in `lightcurves/tic_<tic>/`.





Below, brief descriptions of all files are given.

| Module | Purpose |
| --- | --- |
| `download_main.py` | Entry point for downloading TESS products |
| `tess_products.py` | resolving / querying TESS data products |
| `get_lightcurve.py` | loading and preparing light curves |
| `do_eclipsr.py` | ECLIPSR wrapper, per-target analysis |
| `eclipsr_main.py` | main driver for the eclipse search, recording results |
| `utils/why_few_spocs.py` | check why the number of SPOC targets are low compared to others |
| `utils/build_tess_sectors.py` | Rebuilds `tess_sectors.dat` from the `TSTART`/`TSTOP` headers of your own TESS FITS files, extending ECLIPSR's coverage beyond sector 55|
| `utils/measure_run.py` | Estimates download time and disk usage from a small sample |


## Estimating cost before you start

To find out how much time and disk the full list needs, download a small stratified sample and extrapolate:

```bash
python utils/measure_run.py --list test/download_list.csv --n 60
```

## Note on TESS sector coverage

The `tess_sectors.dat` file bundled with ECLIPSR covers sectors up to 55. If you intend to run `search_eclipses` on later sectors, regenerate it first:

```bash
python utils/build_tess_sectors.py
```

Then copy the generated file over the one shipped with ECLIPSR:

```bash
cp tess_sectors.dat $(python -c "import eclipsr, pathlib; print(pathlib.Path(eclipsr.__file__).parent)")/data/tess_sectors.dat
```

Without this step, light curves from sectors beyond 55 will not be assigned the correct timing reference.

## Examples

Worked examples are in `examples/`, covering typical cases encountered in the survey (detached systems, shallow eclipses, noisy light curves, sampling artefacts, downlink gaps, and failure modes).

## Citation

If you use this code, please cite:

- ECLIPSR — IJspeert et al., *Automated eclipse classification of OBAF-type eclipsing binaries from TESS*, A&A 2021
- Two Gaussian + cosine model — Mowlavi et al., *Two-Gaussian models applied to OGLE-III eclipsing binary light curves in the Large Magellanic Cloud*, A&A 2017 
- Kutluay & Southworth, in prep --- An eclipsing binary catalogue in PLATO LOPS2 Field. 
