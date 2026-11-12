# Hybrid GCN-Transformer Encoder for Landslide Prediction

Source code for the manuscript:

> **Hybrid GCN-Transformer Encoder Framework for Spatio-Temporal Landslide Prediction Using Multivariate Geospatial Data**
> V G Gopika, Shameem Ansar A, Rezni S, Suresh Francis
> Submitted to *Computers & Geosciences*

The code predicts landslide occurrence (binary: `0 = No Risk`, `1 = Land Slide`) using a hybrid **Gated Convolutional Network (GCN) + Transformer Encoder** model. The inputs are 50 multivariate geospatial conditioning factors for the Idukki district, Kerala, India.

## Repository structure

```
├── data/
│   ├── soil_data2.csv          # full dataset — NOT included (see "Data availability"); place it here yourself
│   └── sample/
│       └── sample_data.csv     # 300-row example subset of soil_data2.csv (for the quick test)
├── notebooks/
│   └── Landslide_Prediction_GCN_Transformer.ipynb   # original Google Colab notebook, with outputs
├── src/
│   ├── data_utils.py   # loading, preprocessing, metrics, plots
│   ├── models.py       # GCN-Transformer model definition
│   └── train.py        # command-line training / evaluation script
├── tests/
│   └── quick_test.py   # quick end-to-end test on the sample file
├── requirements.txt
└── LICENSE
```

## Installation

Requires Python 3.9 or newer.

```bash
git clone https://github.com/vg-gopika/gcn-transformer-landslide.git
cd gcn-transformer-landslide
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick test

The quick test trains the model for 3 epochs on the 300-row example file `data/sample/sample_data.csv`. It checks that the whole pipeline runs: preprocessing, training, evaluation and saved outputs.

```bash
python tests/quick_test.py
```

It takes well under a minute on a normal laptop CPU. A successful run ends with:

```
[PASS] pipeline ran, accuracy on sample = ...
Quick test passed.
```

The quick test only checks that the code works. Its accuracy is **not** the paper result: it comes from 240 training samples and 3 epochs, and the learning rate (1e-5) is very small, so accuracy stays low.

## Reproducing the results (full dataset)

> **The full dataset (`soil_data2.csv`) is not part of this repository.** See [Data availability](#data-availability) for how to obtain it. Once you have it, save it as `data/soil_data2.csv` (this path is git-ignored), or point the script to it with `--data PATH`.

Run from the repository root:

```bash
python src/train.py
```

The run prints the classification report, accuracy, precision, recall, F1-score, misclassification rate, ROC AUC and the confusion matrix. It also saves the following to `outputs/`:

- `metrics.txt`, `test_predictions.csv`
- `accuracy.png`, `loss.png`, `confusion_matrix.png`, `roc_curve.png`
- the trained model, `model.keras`

Useful options (`python src/train.py --help`):

| Option | Default | Meaning |
|---|---|---|
| `--data PATH` | `data/soil_data2.csv` | Use a different CSV with the same columns |
| `--epochs N` | 100 | Training epochs |
| `--batch-size N` | 16 | Batch size |
| `--lr X` | 1e-5 | Adam learning rate |
| `--seed N` | none | Fix the random seed for repeatable runs |
| `--out DIR` | `outputs` | Output directory |

On a laptop CPU, a full run (100 epochs) takes about 3 minutes.

### Pipeline (same as the notebook)

1. Remove `Label`, `Longitude` and `Latitude` from the inputs, leaving 50 features; `Label` is the target.
2. Apply z-score standardisation to every feature (`StandardScaler`).
3. Repeat each sample over **T = 10** steps to form a sequence of shape `(N, 10, 50)`.
4. Split 80:20 into train and test, stratified by class (`random_state=42, stratify=y`). The test set is also used as validation data during training.
5. Train for 100 epochs with Adam (learning rate 1e-5), binary cross-entropy and batch size 16.
6. Classify with a decision threshold of 0.5.

### Reported results (manuscript)

| Metric | Value (%) |
|---|---|
| Accuracy | 97.1762 |
| Precision (weighted) | 97.1708 |
| Recall (weighted) | 97.1762 |
| F1-score (weighted) | 97.1724 |
| Misclassification rate | 2.8238 |
| ROC AUC | 0.9955 |

The test set has 2,231 samples (1,528 No Risk, 703 Land Slide).

The original experiments were run on Google Colab without a fixed random seed. Weight initialisation, dropout and batch order are random, so a fresh run gives slightly different values. Pass `--seed` to make a run repeatable. On a laptop CPU, three seeded runs gave:

| Command | Accuracy (%) | F1 (%) | AUC | Confusion matrix (actual × predicted) |
|---|---|---|---|---|
| `python src/train.py --seed 1` | 96.9969 | 96.9928 | 0.9947 | `[[1498, 30], [37, 666]]` |
| `python src/train.py --seed 2` | 97.0865 | 97.0893 | 0.9947 | `[[1493, 35], [30, 673]]` |
| `python src/train.py --seed 3` | 97.2210 | 97.2252 | 0.9949 | `[[1493, 35], [27, 676]]` |

The manuscript's 97.1762 % lies within this range.

## Running the original notebook

`notebooks/Landslide_Prediction_GCN_Transformer.ipynb` is the original Google Colab notebook, saved with its outputs. To re-run it in Google Colab:

1. Obtain the dataset (see [Data availability](#data-availability)) and upload `soil_data2.csv` to the top level of your Google Drive (`My Drive`).
2. Open the notebook in Colab and run all cells. The second code cell mounts your Drive at `/content/gdrive`, and the third reads `/content/gdrive/MyDrive/soil_data2.csv`.

If you put the file somewhere else in your Drive, change the path in the third code cell to match.

To run the notebook outside Colab (e.g. locally with Jupyter), skip the `drive.mount(...)` cell and change the path in the third code cell to `../data/soil_data2.csv`.

## Dataset

The dataset combines 50 topographic, geological, hydrological, soil, climatic, vegetation and anthropogenic conditioning factors for 11,155 locations in the Idukki district: 3,516 landslide and 7,639 non-landslide points. The sources are:

- OpenTopography / USGS EarthExplorer: SRTM and Copernicus DEM
- FAO: soil data
- Geological Survey of India (Bhukosh): geology and lithology
- NRSC–ISRO Bhuvan: land use / land cover
- Central Ground Water Board: aquifer and groundwater data
- India Meteorological Department: rainfall
- NDMA / KSDMA: landslide inventory

Full descriptions of all 50 parameters are in Table 3 of the manuscript.

## Data availability

The full dataset (`soil_data2.csv`, 11,155 samples × 50 features + `Label`, `Longitude`, `Latitude`) is **not publicly released** and is not included in this repository or its history. It is available from the corresponding author on reasonable request (see [Contact](#contact)).

The code expects it at `data/soil_data2.csv`; that path is listed in `.gitignore` so it cannot be committed by accident. Without it, everything except the full training run still works: the quick test only needs the bundled example file `data/sample/sample_data.csv`.

## License

Copyright © 2026 the authors. **All rights reserved.** The code (and the example data file) are public only so the results of the manuscript can be reviewed and reproduced. Any other use, modification or redistribution requires written permission from the authors. See [LICENSE](LICENSE).

## Contact

Corresponding author: **V G Gopika**, 242014@tkmce.ac.in
