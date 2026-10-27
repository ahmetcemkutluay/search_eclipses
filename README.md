# search_eclipses

Eclipse detection in TESS light curves using ECLIPSR and a two-Gaussian model.

This pipeline downloads TESS photometry, runs [ECLIPSR](https://github.com/LucIJspeert/eclipsr) to flag candidate eclipsing systems, and fits a two-Gaussian model to the folded light curves to characterise the eclipses.

<!-- TODO: add a line on the scope of the survey — which TESS sectors, which target list, roughly how many stars. -->

## Installation

```bash
git clone https://github.com/ahmetcemkutluay/search_eclipses.git
cd search_eclipses
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

<!-- TODO: note the Python version you developed on, e.g. "Developed on Python 3.11." -->

## Data

Light curves are **not** included in this repository (~15 GB). They are downloaded from MAST:

```bash
python download_main.py
```

Files are written to `lightcurves/`, which is git-ignored. Intermediate and output directories (`results/`, `status/`, `shards/`) are generated at runtime and are also ignored.

<!-- TODO: fill in the specifics —
     - TESS sectors / cadence used
     - product type (SPOC 2-min, TGLC, QLP, ...)
     - expected directory layout under lightcurves/
     - approximate disk space required
-->

## Usage

Run the pipeline in this order:

```bash
python download_main.py     # 1. fetch TESS products
python eclipsr_main.py      # 2. run the eclipse search
```

<!-- TODO: correct the order and add any required arguments or config files. -->

| Module | Purpose |
| --- | --- |
| `download_main.py` | Entry point for downloading TESS products |
| `tess_products.py` | TODO — resolving / querying TESS data products |
| `get_lightcurve.py` | TODO — loading and preparing light curves |
| `do_eclipsr.py` | TODO — ECLIPSR wrapper, per-target analysis |
| `eclipsr_main.py` | TODO — main driver for the eclipse search |
| `utils/` | TODO — shared helper functions |

## Note on TESS sector coverage

The `tess_sectors.dat` file bundled with ECLIPSR covers sectors up to 55.
If you intend to run `search_eclipses` on later sectors, regenerate it first:

```bash
python utils/build_tess_sectors.py
```

Then copy the generated file over the one shipped with ECLIPSR:

```bash
cp tess_sectors.dat $(python -c "import eclipsr, pathlib; print(pathlib.Path(eclipsr.__file__).parent)")/data/tess_sectors.dat
```

Without this step, light curves from sectors beyond 55 will not be
assigned the correct timing reference.

## Output

<!-- TODO: describe what lands in results/ — file format, one row per target?,
     which columns (period, eclipse depth, width, separation, ECLIPSR score, ...).
     This is the part a reader will care about most. -->

## Examples

Worked examples are in `examples/`, covering typical cases encountered in the survey (detached systems, shallow eclipses, noisy light curves, sampling artefacts, downlink gaps, and failure modes).

## Citation

If you use this code, please cite:

- ECLIPSR — IJspeert et al., *Automated eclipse classification of OBAF-type eclipsing binaries from TESS*, A&A 2021
- <!-- TODO: reference for the two-Gaussian model you implement -->
- <!-- TODO: your own paper / thesis, once available -->

## License

<!-- TODO: choose one. MIT is common for research code; without a license file
     others technically have no right to reuse the code. -->
