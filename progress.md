# Project Progress

Chronological notes for later documentation.

## Baseline Setup

- Set up the initial sketch-to-photo retrieval baseline on the processed CUFSF / ColorFERET dataset.
- Dataset used:
  - `data/processed/cufsf_224_gray/pairs.csv`
  - 1,194 paired identities.
  - Each row maps one sketch to one correct photo.
- Evaluation task:
  - Query: sketch.
  - Gallery: photos.
  - Goal: rank the matching photo as high as possible.
- Metrics used:
  - Recall@1
  - Recall@5
  - Recall@10
  - MRR
  - Mean rank
  - Median rank

## Baseline Results

Baseline results on CUFSF:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| Random | 0.008 | 0.029 | 0.042 | 0.027 | 126 |
| Pixel cosine | 0.013 | 0.033 | 0.075 | 0.040 | 108 |
| Pretrained ResNet18 | 0.025 | 0.084 | 0.113 | 0.066 | 57 |

Findings:

- Random and pixel matching perform very poorly, as expected.
- Pretrained ImageNet ResNet18 is better than random, but still weak.
- This confirmed that fine-tuning is needed for the sketch-photo retrieval task.

## Contrastive ResNet18 Notebook

- Created notebook:
  - `notebooks/02_cufsf_contrastive_resnet18_training.ipynb`
- Added frozen splits:
  - `train_pairs.csv`
  - `val_pairs.csv`
  - `test_pairs.csv`
- Split sizes:
  - Train: 764 pairs.
  - Validation: 191 pairs.
  - Test: 239 pairs.
- Implemented:
  - Paired sketch-photo Dataset.
  - DataLoaders.
  - ResNet18 encoder.
  - Projection head.
  - Symmetric contrastive loss.
  - Training loop.
  - Validation retrieval evaluation.
  - Best model checkpointing by validation MRR.
  - Loss and retrieval curves.
  - Top-5 retrieval visualizations.
  - Optional FS2K OOD evaluation.

## Training Run

- Training ran on CPU.
- Early stopping triggered after 15 epochs.
- Best checkpoint was epoch 7.
- Best validation metrics:

| Metric | Value |
|---|---:|
| Recall@1 | 0.277 |
| Recall@5 | 0.560 |
| Recall@10 | 0.712 |
| MRR | 0.421 |
| Median Rank | 4 |

Findings:

- Training loss decreased quickly.
- Validation performance improved strongly until around epoch 7.
- After epoch 7, training loss kept decreasing but validation performance did not improve much.
- This suggests the model started fitting the CUFSF training distribution more tightly rather than improving general retrieval ability.

## CUFSF Test Results

Final fine-tuned ResNet18 result on the frozen CUFSF test split:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| Fine-tuned ResNet18 | 0.272 | 0.565 | 0.678 | 0.406 | 4 |

Findings:

- Fine-tuning improved strongly over the pretrained ResNet18 baseline.
- Pretrained ResNet18 Recall@1 was 0.025.
- Fine-tuned ResNet18 Recall@1 was 0.272.
- This shows that contrastive training successfully adapted the model to CUFSF sketch-photo retrieval.
- The correct match is usually ranked near the top, with a median rank of 4.

## FS2K OOD Evaluation

- Built FS2K sketch-photo pairs from raw FS2K folders.
- Pairing rule:
  - `photo/photo1/image0001.jpg` maps to `sketch/sketch1/sketch0001.jpg`.
  - Same logic for groups 2 and 3.
- FS2K pairs:
  - 2,104 pairs.
- Evaluated the CUFSF-trained model on FS2K without fine-tuning.

FS2K OOD results:

| Split | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| CUFSF test | 0.272 | 0.565 | 0.678 | 0.406 | 4 |
| FS2K OOD | 0.004 | 0.015 | 0.021 | 0.013 | 767 |

Findings:

- OOD performance on FS2K is extremely poor.
- The model does not generalize well from CUFSF to FS2K.
- Correct FS2K matches are usually ranked hundreds of places down.
- The model likely learned CUFSF-specific cues rather than a robust general sketch-to-photo identity representation.

## Standardized Ingestion Experiment

- Created notebook:
  - `notebooks/03_standardized_ingestion_ood_experiment.ipynb`
- Goal:
  - Test whether standardizing all images and reducing background variation improves OOD transfer.
- Standardization used:
  - Grayscale conversion.
  - Consistent crop and resize to 224 x 224.
  - Contrast normalization.
  - Neutral-background oval mask for photos.
  - PIL fallback pipeline because OpenCV was not installed.
- Standardized datasets created:
  - CUFSF train/val/test using the same frozen split.
  - FS2K all pairs.

Diagnostic with the old CUFSF checkpoint on standardized images:

| Experiment | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| Old checkpoint, standardized CUFSF test | 0.151 | 0.393 | 0.519 | 0.270 | 9 |
| Old checkpoint, standardized FS2K OOD | 0.012 | 0.042 | 0.063 | 0.034 | 343 |

Findings:

- Standardization improved FS2K OOD performance for the old checkpoint.
- Original FS2K Recall@1 was 0.004; standardized FS2K Recall@1 became 0.012.
- Original FS2K median rank was 767; standardized FS2K median rank became 343.
- However, standardized CUFSF performance dropped for the old checkpoint because the model had not been trained on this new preprocessing.

## Standardized ResNet18 Training

- Retrained ResNet18 on standardized CUFSF train split.
- Best checkpoint was epoch 19.
- Best validation metrics:

| Metric | Value |
|---|---:|
| Recall@1 | 0.199 |
| Recall@5 | 0.518 |
| Recall@10 | 0.717 |
| MRR | 0.348 |
| Median Rank | 5 |

Final standardized pipeline comparison:

| Experiment | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| Previous pipeline, CUFSF test | 0.272 | 0.565 | 0.678 | 0.406 | 4 |
| Previous pipeline, FS2K OOD | 0.004 | 0.015 | 0.021 | 0.013 | 767 |
| Standardized pipeline, CUFSF test | 0.159 | 0.448 | 0.636 | 0.313 | 6 |
| Standardized pipeline, FS2K OOD | 0.008 | 0.030 | 0.049 | 0.024 | 465.5 |

Findings:

- Standardization helped FS2K OOD relative to the original pipeline.
- It roughly doubled FS2K Recall@1 and Recall@5.
- It improved FS2K median rank from 767 to 465.5 after retraining.
- However, CUFSF test performance dropped compared with the original pipeline.
- This suggests the standardization removes some nuisance variation, but may also remove or distort useful identity cues.
- The OOD problem is not solved by preprocessing alone.

## Multi-Dataset Standardized Training

- Created notebook:
  - `notebooks/04_multidataset_standardized_training_fei_eval.ipynb`
- Goal:
  - Train on both standardized CUFSF and standardized FS2K.
  - Test whether dataset diversity improves general retrieval.
  - Evaluate on a third dataset, FEI, using AI-generated sketches.
- Training data:
  - CUFSF train: 764 pairs.
  - FS2K train: 1,472 pairs.
  - Combined train: 2,236 pairs.
- Validation data:
  - CUFSF val: 191 pairs.
  - FS2K val: 316 pairs.
  - Best model selected by average CUFSF val MRR and FS2K val MRR.
- Best checkpoint:
  - Epoch 17.

Best validation metrics:

| Split | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| CUFSF val | 0.403 | 0.707 | 0.838 | 0.541 | 2 |
| FS2K val | 0.696 | 0.934 | 0.962 | 0.795 | 1 |
| Combined val | 0.568 | 0.828 | 0.890 | 0.679 | 1 |

Test results:

| Split | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| CUFSF test | 0.389 | 0.745 | 0.879 | 0.544 | 2 |
| FS2K held-out test | 0.655 | 0.902 | 0.946 | 0.769 | 1 |

Findings:

- Multi-dataset training strongly improved performance.
- CUFSF test Recall@1 improved from 0.272 in the original CUFSF-only model to 0.389.
- FS2K held-out test performance became very strong after FS2K was included in training.
- This suggests the earlier OOD failure was mainly caused by limited training diversity, not by the ResNet18 contrastive approach itself.
- Standardization alone was not enough, but standardization plus multi-dataset training worked much better.

## FEI Third-Dataset Evaluation

- Added FEI as a third dataset.
- FEI originally has photos only, so sketches were AI-generated for selected FEI identities.
- Selected FEI sketch query set:
  - 20 identities.
  - One selected face image per identity from orientation 11 or 13.
- FEI gallery sets:
  - `fei-centralized-set`: one image per identity, randomly sampled from orientations 06, 11, and 13.
  - `fei-controlled-set`: one image per identity, always orientation 13.
- Added `originalimages_part4`, expanding FEI gallery identities from 150 to 200.
- Current FEI evaluation:
  - 20 generated sketch queries.
  - 200 gallery photos.
  - Correct match is same FEI person id.

FEI full-gallery results:

| Split | Recall@1 | Recall@5 | Recall@10 | MRR | Median Rank |
|---|---:|---:|---:|---:|---:|
| FEI centralized 200-photo gallery | 0.100 | 0.400 | 0.500 | 0.219 | 11 |
| FEI controlled orientation-13 200-photo gallery | 0.250 | 0.400 | 0.600 | 0.343 | 8 |

Findings:

- FEI performance is above chance even with a 200-photo gallery.
- Chance Recall@1 with 200 gallery photos is 0.005.
- The controlled orientation-13 gallery performs better than the mixed-orientation centralized gallery.
- This suggests pose consistency matters for sketch-photo retrieval.
- FEI results are lower than CUFSF/FS2K because FEI is a true third-dataset test and uses AI-generated sketches.
- The result still shows partial transfer to unseen FEI identities and generated sketch inputs.

## Main Interpretation So Far

- The fine-tuned model works well in-distribution on CUFSF.
- It performs much better than random, pixel cosine, and pretrained ResNet18 baselines.
- The poor FS2K result shows a strong domain shift problem.
- CUFSF is controlled and consistent, while FS2K has different sketch styles, photo conditions, poses, backgrounds, lighting, and image quality.
- Standardization reduces part of the domain gap, but does not create robust cross-dataset identity matching by itself.
- Multi-dataset training on standardized CUFSF plus FS2K creates a much stronger and more transferable embedding.
- FEI shows that the model can transfer partially to a third dataset, but performance depends strongly on pose consistency and sketch generation quality.
- This is an important limitation to discuss in the report:
  - The method is effective on the dataset it was trained for.
  - Generalization to a different sketch-photo dataset remains weak.
  - Training on multiple sketch-photo datasets improves transfer, but does not fully solve third-dataset OOD retrieval.

## Useful Output Files

- Baseline notebook:
  - `notebooks/01_cufsf_baseline_retrieval.ipynb`
- Training notebook:
  - `notebooks/02_cufsf_contrastive_resnet18_training.ipynb`
- Standardized ingestion notebook:
  - `notebooks/03_standardized_ingestion_ood_experiment.ipynb`
- Multi-dataset training and FEI notebook:
  - `notebooks/04_multidataset_standardized_training_fei_eval.ipynb`
- Frozen splits:
  - `data/outputs/contrastive_resnet18_cufsf/splits/train_pairs.csv`
  - `data/outputs/contrastive_resnet18_cufsf/splits/val_pairs.csv`
  - `data/outputs/contrastive_resnet18_cufsf/splits/test_pairs.csv`
- Best model:
  - `data/outputs/contrastive_resnet18_cufsf/models/best_resnet18_contrastive.pt`
- Result tables:
  - `data/outputs/baseline_cufsf/baseline_results.csv`
  - `data/outputs/contrastive_resnet18_cufsf/fine_tuned_results.csv`
  - `data/outputs/contrastive_resnet18_cufsf/cufsf_test_vs_fs2k_ood_results.csv`
  - `data/outputs/standardized_ingestion_ood/old_checkpoint_standardized_eval.csv`
  - `data/outputs/standardized_ingestion_ood/standardized_vs_previous_results.csv`
  - `data/outputs/multidataset_standardized_resnet18/multidataset_cufsf_fs2k_test_results.csv`
  - `data/outputs/multidataset_standardized_resnet18/multidataset_with_fei_full_gallery_results.csv`
- Figures:
  - `data/outputs/contrastive_resnet18_cufsf/figures/training_curves.png`
  - `data/outputs/contrastive_resnet18_cufsf/retrieval_examples/fine_tuned_cufsf_test_top5.png`
  - `data/outputs/contrastive_resnet18_cufsf/retrieval_examples/fine_tuned_fs2k_ood_top5.png`
  - `data/outputs/standardized_ingestion_ood/figures/standardized_training_curves.png`
  - `data/outputs/standardized_ingestion_ood/retrieval_examples/standardized_cufsf_test_top5.png`
  - `data/outputs/standardized_ingestion_ood/retrieval_examples/standardized_fs2k_ood_top5.png`
  - `data/outputs/multidataset_standardized_resnet18/figures/multidataset_training_curves.png`
  - `data/outputs/multidataset_standardized_resnet18/retrieval_examples/multidataset_cufsf_test_top5.png`
  - `data/outputs/multidataset_standardized_resnet18/retrieval_examples/multidataset_fs2k_test_top5.png`
  - `data/outputs/multidataset_standardized_resnet18/retrieval_examples/fei_centralized_full150_gallery_top5.png`
  - `data/outputs/multidataset_standardized_resnet18/retrieval_examples/fei_controlled_full150_gallery_top5.png`
