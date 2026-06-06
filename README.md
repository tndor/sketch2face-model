# Sketch2Face

Sketch-to-photo face retrieval experiments for the semester 4 main challenge.

The project evaluates whether a model can use a sketch as a query and retrieve the matching face photo from a gallery.

## Task

```text
query sketch -> rank photo gallery -> measure where the correct photo appears
```

Metrics used:

- Recall@1
- Recall@5
- Recall@10
- MRR
- Mean rank
- Median rank

## Data

Main datasets used:

- **CUFSF / ColorFERET-style processed set**
  - `data/processed/cufsf_224_gray/pairs.csv`
  - 1,194 paired sketch-photo identities.
- **FS2K**
  - Raw FS2K photo/sketch folders.
  - 2,104 paired identities.
- **FEI**
  - Photos only.
  - AI-generated sketches were created for 20 selected FEI identities.
  - FEI is used as a third-dataset transfer test.

FEI gallery folders:

- `data/FEI/fei-selected-set`
  - 20 selected photos used to generate sketches.
- `data/FEI/fei-centralized-set`
  - 200 photos, one per FEI identity, sampled from orientations 06, 11, and 13.
- `data/FEI/fei-controlled-set`
  - 200 photos, one per FEI identity, all orientation 13.
- `data/FEI/fei-selected-sketches`
  - Expected folder for generated FEI sketches.

## Notebooks

Run order:

1. `notebooks/01_cufsf_baseline_retrieval.ipynb`
   - Random, pixel cosine, and pretrained ResNet18 baselines.
2. `notebooks/02_cufsf_contrastive_resnet18_training.ipynb`
   - Contrastive ResNet18 trained on CUFSF.
3. `notebooks/03_standardized_ingestion_ood_experiment.ipynb`
   - Standardized preprocessing experiment for CUFSF and FS2K.
4. `notebooks/04_multidataset_standardized_training_fei_eval.ipynb`
   - Multi-dataset standardized training on CUFSF + FS2K.
   - FEI full-gallery evaluation.

Notebook 04 can rerun CUFSF/FS2K/FEI evaluation without retraining, as long as the saved checkpoint exists.

## Best Model

Best current model:

```text
data/outputs/multidataset_standardized_resnet18/models/best_multidataset_standardized_resnet18.pt
```

Earlier checkpoints:

```text
data/outputs/contrastive_resnet18_cufsf/models/best_resnet18_contrastive.pt
data/outputs/standardized_ingestion_ood/models/best_resnet18_standardized_contrastive.pt
```

## Main Results

### Baselines on CUFSF

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

The model learned CUFSF retrieval, but generalized very poorly to FS2K.

### Standardized CUFSF-Only Training

| Split | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| CUFSF test | 0.159 | 0.448 | 0.636 | 0.313 | 6 |
| FS2K OOD | 0.008 | 0.030 | 0.049 | 0.024 | 465.5 |

Standardization helped FS2K slightly, but hurt CUFSF. Preprocessing alone did not solve OOD generalization.

### Multi-Dataset Standardized Training

Training data:

- CUFSF train: 764 pairs.
- FS2K train: 1,472 pairs.
- Combined train: 2,236 pairs.

| Split | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| CUFSF test | 0.389 | 0.745 | 0.879 | 0.544 | 2 |
| FS2K held-out test | 0.655 | 0.902 | 0.946 | 0.769 | 1 |

Multi-dataset training greatly improved the embedding and made the model much more transferable.

### FEI Third-Dataset Test

FEI evaluation setup:

- 20 generated sketch queries.
- 200-photo gallery.
- Correct match is same FEI person id.

| Split | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| FEI centralized 200-photo gallery | 0.100 | 0.400 | 0.500 | 0.219 | 11 |
| FEI controlled orientation-13 200-photo gallery | 0.250 | 0.400 | 0.600 | 0.343 | 8 |

FEI performance is above chance. With 200 gallery images, chance Recall@1 is 0.005. The controlled orientation-13 gallery performs better, suggesting pose consistency matters.

## Interpretation

The main experimental arc:

1. Pretrained features and pixel similarity are weak baselines.
2. Contrastive ResNet18 learns CUFSF well.
3. CUFSF-only training fails badly on FS2K OOD.
4. Standardized preprocessing helps OOD slightly but is not enough.
5. Training on both standardized CUFSF and FS2K produces a much stronger retrieval model.
6. FEI shows partial transfer to a third dataset with AI-generated sketches, but performance still depends on pose consistency and sketch quality.

## Important Outputs

Result tables:

```text
data/outputs/baseline_cufsf/baseline_results.csv
data/outputs/contrastive_resnet18_cufsf/fine_tuned_results.csv
data/outputs/contrastive_resnet18_cufsf/cufsf_test_vs_fs2k_ood_results.csv
data/outputs/standardized_ingestion_ood/standardized_vs_previous_results.csv
data/outputs/multidataset_standardized_resnet18/multidataset_cufsf_fs2k_test_results.csv
data/outputs/multidataset_standardized_resnet18/multidataset_with_fei_full_gallery_results.csv
```

Retrieval visualizations:

```text
data/outputs/contrastive_resnet18_cufsf/retrieval_examples/
data/outputs/standardized_ingestion_ood/retrieval_examples/
data/outputs/multidataset_standardized_resnet18/retrieval_examples/
```

Progress notes:

```text
progress.md
```

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Current dependency list:

```text
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
pillow
tqdm
faiss-cpu
```
