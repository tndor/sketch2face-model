# Sketch2Face

Sketch2Face is a small research project for sketch-to-photo face retrieval. Given a sketch query, the notebooks rank a gallery of face photos and measure where the matching identity appears.

```text
query sketch -> encode sketch -> rank photo gallery -> evaluate correct match rank
```

The project starts with simple CUFSF baselines, trains contrastive ResNet18 models, tests out-of-distribution transfer to FS2K, then evaluates the best multi-dataset model on FEI photos with generated sketches.

## Repository Contents

```text
sketch2face/
  notebooks/
    01_cufsf_baseline_retrieval.ipynb
    02_cufsf_contrastive_resnet18_training.ipynb
    03_standardized_ingestion_ood_experiment.ipynb
    04_multidataset_standardized_training_fei_eval.ipynb
  progress.md
  requirements.txt
  README.md
```

Large datasets, model checkpoints, generated figures, and CSV outputs are intentionally not stored in the repo. The notebooks expect them in a sibling `data/` folder:

```text
main-challenge/
  sketch2face/
  data/
```

If you clone the repo somewhere else, update the `PROJECT_ROOT`, `DATA_ROOT`, or `ROOT` path in the first configuration cell of each notebook.

## Environment

Use Python 3.10 or newer. The original runs used Python 3.12.

```bash
cd sketch2face
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name sketch2face --display-name "Python (sketch2face)"
```

PyTorch will use CUDA if it is available. CPU works, but the training notebooks are slower.

The notebooks use `torchvision` pretrained ResNet18 weights. On the first run, `torchvision` may download those weights. If the download fails, the notebooks fall back to a randomly initialized ResNet18, but that will not reproduce the reported results.

## Data Sources

You need three public face datasets plus one small generated-sketch set:

| Dataset | Used for | Where to get it |
|---|---|---|
| CUFSF sketches | Main paired sketch/photo dataset | CUHK Face Sketch / CUFSF dataset page, commonly listed at http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html, or a mirrored CUFSF archive. |
| Color FERET photos | Matching photos for CUFSF identities | NIST Color FERET database: https://www.nist.gov/itl/products-and-services/color-feret-database |
| FS2K | Second paired sketch/photo dataset | Official FSGAN/FS2K repository: https://github.com/DengPingFan/FSGAN |
| FEI Face Database | Third-dataset photo gallery | FEI Face Database page, commonly distributed as `originalimages_part1.zip` through `originalimages_part4.zip`: https://fei.edu.br/~cet/facedatabase.html |
| FEI generated sketches | FEI sketch queries | Generate these yourself from the selected FEI photos and place them in `data/FEI/fei-selected-sketches/`. |

Follow each dataset's license and access rules. Color FERET in particular requires a NIST account request and release agreement.

## Expected Data Layout

After downloading and extracting the data, arrange it like this:

```text
main-challenge/data/
  cufs-dataset/
    cufsf-cropped_sketch/
    misc/photo_points/

  colorferet/
    dvd1/
    dvd2/

  FS2K/
    photo/
      photo1/
      photo2/
      photo3/
    sketch/
      sketch1/
      sketch2/
      sketch3/

  FEI/
    originalimages_part1/
    originalimages_part2/
    originalimages_part3/
    originalimages_part4/
    fei-selected-set/
      manifest.csv
    fei-centralized-set/
      manifest.csv
    fei-controlled-set/
      manifest.csv
    fei-selected-sketches/
```

The notebooks do not need every raw CUFS file. For CUFSF they depend on:

- `data/cufs-dataset/cufsf-cropped_sketch/`
- `data/cufs-dataset/misc/photo_points/*.3pts`
- matching Color FERET `.tif.bz2` photo files under `data/colorferet/`

## Preparing CUFSF

Notebook 01 expects a processed CUFSF folder:

```text
data/processed/cufsf_224_gray/
  pairs.csv
  sketches/
  photos/
```

The `pairs.csv` file must contain:

```text
id,dataset,sketch_path,photo_path,feret_basename
```

The local preparation used for this project did two steps:

1. Match CUFSF sketch ids to Color FERET photos using the CUFSF `photo_points/*.3pts` filenames. For example, a point file named `00001fb010_930831.3pts` maps to sketch id `00001`.
2. Convert each matched sketch/photo pair to grayscale 224 x 224 RGB-compatible JPEGs and write `data/processed/cufsf_224_gray/pairs.csv`.

If you already have a processed CUFSF release, you can skip the raw matching step and create the same `cufsf_224_gray` folder directly. The important rule is that each row in `pairs.csv` points to one sketch and its one matching photo.

## Preparing FS2K

Extract FS2K into `data/FS2K/` with the `photo/photo1`, `photo/photo2`, `photo/photo3`, `sketch/sketch1`, `sketch/sketch2`, and `sketch/sketch3` folders.

The notebooks build FS2K pairs automatically by matching numeric stems:

```text
photo/photo1/image0001.jpg  <->  sketch/sketch1/sketch0001.jpg
photo/photo2/image0001.jpg  <->  sketch/sketch2/sketch0001.png
photo/photo3/image0001.jpg  <->  sketch/sketch3/sketch0001.jpg
```

The expected total is 2,104 pairs.

## Preparing FEI

FEI is used only in notebook 04 as a third-dataset transfer test.

The expected raw FEI folders are:

```text
data/FEI/originalimages_part1/
data/FEI/originalimages_part2/
data/FEI/originalimages_part3/
data/FEI/originalimages_part4/
```

This project evaluates 20 generated sketch queries against 200-photo FEI galleries. The FEI helper cells expect:

```text
data/FEI/fei-selected-set/manifest.csv
data/FEI/fei-centralized-set/manifest.csv
data/FEI/fei-controlled-set/manifest.csv
data/FEI/fei-selected-sketches/
```

The selected FEI query identities used here are:

```text
1, 8, 15, 22, 29, 36, 43, 51, 58, 65,
72, 79, 86, 93, 101, 108, 115, 122, 129, 136
```

Their selected photos alternate between orientations 11 and 13, starting with `1-11.jpg`, then `8-13.jpg`, and so on. The controlled gallery uses orientation 13 for every FEI identity. The centralized gallery uses one photo per identity sampled from orientations 06, 11, and 13; for selected query identities, do not use the exact selected query photo in the gallery.

The generated sketch filenames may match either the selected photo stem, such as `1-11.png`, or the person id, such as `1.png`.

## Running the Experiments

Run the notebooks in order from top to bottom.

1. `notebooks/01_cufsf_baseline_retrieval.ipynb`

   Runs random, pixel cosine, and pretrained ResNet18 baselines on CUFSF.

2. `notebooks/02_cufsf_contrastive_resnet18_training.ipynb`

   Creates frozen CUFSF train/validation/test splits, trains a contrastive ResNet18 on CUFSF, saves the best checkpoint, and optionally evaluates on FS2K.

3. `notebooks/03_standardized_ingestion_ood_experiment.ipynb`

   Reuses the notebook 02 splits, writes standardized CUFSF and FS2K folders, evaluates the old CUFSF checkpoint on standardized inputs, then trains a standardized CUFSF-only model.

4. `notebooks/04_multidataset_standardized_training_fei_eval.ipynb`

   Trains on standardized CUFSF plus standardized FS2K, evaluates CUFSF/FS2K test splits, and evaluates FEI if the FEI folders and generated sketches are present.

Notebook 04 can reload the saved multi-dataset checkpoint and rerun evaluation without retraining.

## Generated Outputs

The notebooks write outputs under:

```text
data/outputs/
  baseline_cufsf/
  contrastive_resnet18_cufsf/
  standardized_ingestion_ood/
  multidataset_standardized_resnet18/
```

Important checkpoints:

```text
data/outputs/contrastive_resnet18_cufsf/models/best_resnet18_contrastive.pt
data/outputs/standardized_ingestion_ood/models/best_resnet18_standardized_contrastive.pt
data/outputs/multidataset_standardized_resnet18/models/best_multidataset_standardized_resnet18.pt
```

Important result CSVs:

```text
data/outputs/baseline_cufsf/baseline_results.csv
data/outputs/contrastive_resnet18_cufsf/fine_tuned_results.csv
data/outputs/contrastive_resnet18_cufsf/cufsf_test_vs_fs2k_ood_results.csv
data/outputs/standardized_ingestion_ood/standardized_vs_previous_results.csv
data/outputs/multidataset_standardized_resnet18/multidataset_cufsf_fs2k_test_results.csv
data/outputs/multidataset_standardized_resnet18/multidataset_with_fei_full_gallery_results.csv
```

Retrieval visualizations are saved in each experiment's `retrieval_examples/` folder.

## Reported Results

### CUFSF Baselines

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| Random | 0.008 | 0.029 | 0.042 | 0.027 | 126 |
| Pixel cosine | 0.013 | 0.033 | 0.075 | 0.040 | 108 |
| Pretrained ResNet18 | 0.025 | 0.084 | 0.113 | 0.066 | 57 |

### CUFSF-Only Contrastive ResNet18

| Split | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| CUFSF test | 0.272 | 0.565 | 0.678 | 0.406 | 4 |
| FS2K OOD | 0.004 | 0.015 | 0.021 | 0.013 | 767 |

### Standardized CUFSF-Only Training

| Split | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| CUFSF test | 0.159 | 0.448 | 0.636 | 0.313 | 6 |
| FS2K OOD | 0.008 | 0.030 | 0.049 | 0.024 | 465.5 |

### Multi-Dataset Standardized Training

| Split | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| CUFSF test | 0.389 | 0.745 | 0.879 | 0.544 | 2 |
| FS2K held-out test | 0.655 | 0.902 | 0.946 | 0.769 | 1 |

### FEI Third-Dataset Evaluation

| Split | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| FEI centralized 200-photo gallery | 0.100 | 0.400 | 0.500 | 0.219 | 11 |
| FEI controlled orientation-13 200-photo gallery | 0.250 | 0.400 | 0.600 | 0.343 | 8 |

With a 200-photo FEI gallery, chance Recall@1 is 0.005. The controlled orientation-13 gallery performs better, which suggests that pose consistency matters.

## Troubleshooting

- `FileNotFoundError` for `pairs.csv`: check that `data/processed/cufsf_224_gray/pairs.csv` exists and that the notebook path cell points to the correct parent folder.
- Missing FS2K pairs: check that FS2K is arranged as `data/FS2K/photo/photo*` and `data/FS2K/sketch/sketch*`.
- Missing FEI results: create `data/FEI/fei-selected-sketches/` and add one generated sketch for each selected FEI identity.
- Different results: keep the notebook seed values, run notebooks in order, and reuse the frozen split CSVs once created.
- Slow training: use a CUDA GPU if possible; otherwise reduce `EPOCHS` for a quick smoke test.

## Notes

`progress.md` contains the chronological experiment log and interpretation. It is useful if you want to understand why each notebook was added.
