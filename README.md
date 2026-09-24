# AT2025abao

Data, analysis code, and notebooks for AT2025abao. The notebooks cover optical spectral analysis and SED modeling with [DUSTY](https://github.com/ivezic/dusty) radiative-transfer models. The models are evaluated fast through trained neural-network emulators and fit with MCMC.

## Installation

The code needs Python 3.10 or newer. Clone the repository and install the dependencies into a fresh environment:

```bash
git clone https://github.com/virajkaram/at2025abao.git
cd at2025abao

conda create -n at2025abao python=3.11
conda activate at2025abao
pip install .
```

`pip install .` installs every dependency listed in `pyproject.toml`. That includes [pydusty](https://github.com/virajkaram/pydusty) and [bbFit](https://github.com/virajkaram/bbFit), which come directly from GitHub. The scripts in `code/` are not installed as a package. The notebooks add that folder to the import path themselves.

## Running the notebooks

Start Jupyter and open the notebooks from inside the `notebooks/` directory. They load data with paths relative to that directory, such as `../observational_data/`.

```bash
cd notebooks
jupyter lab
```

## Repository contents

### Notebooks (`notebooks/`)

| Notebook | Description |
|---|---|
| `spectral_analysis.ipynb` | Loads the optical spectra, fits H-alpha line profiles, tracks FWHM velocity evolution, and compares against similar transients. It also includes a simple ejecta–CSM interaction model. |
| `dusty_modeling_progenitor.ipynb` | Fits archival progenitor photometry, from optical to Spitzer bands, with a DUSTY emulator and MCMC. |
| `dusty_modeling_spherex.ipynb` | Fits the first-epoch SPHEREx spectrum plus contemporaneous g- and r-band photometry with a DUSTY emulator and MCMC. |
| `dusty_modeling_jwst_mrs.ipynb` | Jointly fits the JWST MIRI MRS spectrum, MIRI photometry, the second-epoch SPHEREx spectrum, and optical/NIR photometry with a 6-parameter DUSTY emulator. |

### Code (`code/`)

| File | Description |
|---|---|
| `dusty_utils.py` | Helper for reading DUSTY model SED files. |
| `dusty_nn_mcmc_utils.py` | Likelihood, priors, and `emcee` MCMC wrapper for fitting observed SEDs with the DUSTY neural-network emulator. |
| `train_dusty_phoenix_nn.py` | Defines the SED emulator networks and trains them on a grid of DUSTY models. |
| `train_dusty_r1_nn.py` | Trains a network that predicts the DUSTY inner dust radius (r1) from model parameters. |

### Trained models (`dusty_models/`)

| File | Description |
|---|---|
| `dusty_ml_interpolator_g00_phoenix_wavs_logtau_predictor.pth` | SED emulator weights on the PHOENIX wavelength grid, used by the progenitor and SPHEREx notebooks. |
| `dusty_ml_interpolator_g00_phoenix_wavs_logtau_r1_predictor.pth` | Matching inner-radius (r1) emulator weights. |
| `dusty_model_6pars_logY_predictor.pth` | 6-parameter SED emulator weights, used by the JWST MRS notebook. |
| `dusty_model_r1_6pars_logY_predictor.pth` | Matching 6-parameter inner-radius (r1) emulator weights. |
| `sed_*.dat` | Example DUSTY output SEDs that provide the emulators' wavelength grids. |

### Observational data (`observational_data/`)

| File | Description |
|---|---|
| `at2025abao_full_lc.csv` | Multi-band light curve, including archival photometry, with MJD, magnitude, uncertainty, and filter. |
| `at2025abao_jwst_miri_mrs_spectrum.csv` | JWST MIRI MRS spectrum, with wavelength in μm and flux in Jy. |
| `at2025abao_jwst_miri_photometry.csv` | JWST MIRI imaging photometry, with wavelength in μm and flux in μJy. |
| `spherex_epoch1.csv`, `spherex_epoch2.csv` | SPHEREx spectrophotometry from two epochs. |
| `spectra/` | Optical and near-infrared spectra from several instruments, named by observation date and instrument. |
| `comparison_spectra/` | Spectra of comparison transients (iPTF15t, AT2021blu, AT2019zhd). |

### Fit results (`fit_parameters/`)

| File | Description |
|---|---|
| `eruption_bb_fit_parameters.csv` | Parameters from MCMC blackbody fits to the eruption data. |
| `precursor_dusty_fit_parameters.csv` | Parameters from MCMC DUSTY fits to the precursor data. |
Column descriptions are provided as headers in the two files.