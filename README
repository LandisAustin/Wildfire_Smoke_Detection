# Wildfire Smoke Detection Using YOLO11

A computer vision project focused on detecting wildfire smoke in images using deep learning and object detection.

Developed as part of **CS 458 — Data Mining at the University of Nevada, Reno**, this project explores how different YOLO architectures, image resolutions, training datasets, and negative examples affect wildfire smoke detection performance.

## Project Overview

Early detection of wildfire smoke can help identify potential fires before they spread. However, automated smoke detection presents several challenges, including distant or faint smoke plumes, changing lighting conditions, and visual similarities between smoke, clouds, fog, and haze.

This project uses **Ultralytics YOLO11** to train object detection models capable of identifying smoke and predicting its location within an image.

The project focuses on:

- Training pretrained YOLO models for a custom smoke detection task.
- Investigating the effects of image resolution and model architecture.
- Evaluating the impact of smoke-negative training examples.
- Expanding and refining a manually annotated dataset.
- Comparing detection performance using standard object detection metrics.

## Technologies Used

- **Python** — Dataset processing, training, and evaluation
- **Ultralytics YOLO11** — Object detection architecture
- **PyTorch** — Deep learning framework
- **OpenCV** — Image processing
- **Pandas** — Dataset analysis and evaluation
- **Matplotlib** — Training and evaluation visualizations
- **Jupyter Notebook / VS Code** — Experiment development
- **CUDA** — GPU-accelerated training

Training was performed locally using an NVIDIA GeForce GTX 1660 SUPER GPU.

## Dataset

The final dataset contains **3,758 images** from 103 recording groups, including smoke-positive and smoke-negative examples.

| Category | Images |
|---|---:|
| Smoke-positive | 2,465 |
| Smoke-negative | 1,293 |
| **Total** | **3,758** |

Images were annotated using the YOLO bounding-box format, with a single detection class: `smoke`.

### Dataset Preparation

To reduce data leakage, the dataset was split by recording rather than randomly assigning individual frames.

This ensures that frames from the same recording do not appear in both training and validation.

| Split | Images | Recordings | Positive | Negative |
|---|---:|---:|---:|---:|
| Training | 3,006 | 86 | 2,026 | 980 |
| Validation | 752 | 17 | 439 | 313 |

The dataset preparation script validates annotations, identifies recording groups, generates the train/validation split, and saves a manifest for reproducibility.

**Note:** The full image dataset is excluded from Git because of its size. Dataset preparation scripts are included in the repository.

## Experiments

Five experiments were conducted to investigate different training configurations.

| Experiment | Model | Resolution | Training configuration |
|---|---|---|---|
| Baseline | YOLO11n | 640 | Original annotated dataset |
| Experiment 1 | YOLO11n | 960 | Increased image resolution |
| Experiment 2 | YOLO11n | 640 | Added 180 negative images |
| Experiment 3 | YOLO11s | 960 | Larger model architecture |
| Experiment 4 | YOLO11n | 960 | Expanded negative training examples |
| Experiment 5 | YOLO11n | 960 | Full expanded dataset with new recording-based split |

The first four experiments used the original validation dataset of 191 images.

Experiment 5 introduced a larger dataset and a new validation split of 752 images. Its evaluation results are therefore reported separately rather than treated as a direct controlled comparison.

## Results

Model performance was evaluated using precision, recall, F1 score, mAP50, and mAP50:95.

### Experiments 1–4

| Experiment | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.6197 | 0.3413 | 0.4402 | 0.3689 | 0.1639 |
| Experiment 1 | 0.5661 | 0.3894 | 0.4615 | 0.3923 | **0.1826** |
| Experiment 2 | 0.6197 | 0.3508 | 0.4480 | 0.3313 | 0.1466 |
| Experiment 3 | 0.4974 | 0.3576 | 0.4161 | 0.3558 | 0.1604 |
| Experiment 4 | 0.5748 | **0.4921** | **0.5303** | **0.4543** | 0.1738 |

Experiment 1 achieved the highest mAP50:95, the project's primary evaluation metric, on the original validation set.

Experiment 5 achieved the highest recall, F1, and mAP50 on that same validation set.

### Experiment 5 — Expanded Dataset

| Metric | Result |
|---|---:|
| Precision | 0.6539 |
| Recall | 0.5164 |
| F1 | 0.5771 |
| mAP50 | 0.5793 |
| **mAP50:95** | **0.2937** |

Experiment 5 produced encouraging results on its expanded validation dataset. However, because the validation data changed, these results do not establish a controlled improvement over the earlier models.

## Key Findings

- Increasing image resolution from 640 to 960 improved mAP50:95 in the initial experiments.
- Adding more smoke-negative examples in Experiment 4 was associated with improved recall and F1, although bounding-box localization remained challenging.
- Increasing model size from YOLO11n to YOLO11s did not improve performance in the tested configuration.
- Training on the expanded dataset produced promising detection results, but further evaluation on a common held-out test set would be needed for a rigorous comparison.
- Detecting faint or distant smoke remains an important challenge.

## Repository Structure

```text
Wildfire_Smoke_Project/
├── notebooks/                 # Training and evaluation notebooks
├── prepare_experiment_5.py    # Dataset preparation and splitting
├── verify_smoke_dataset.py    # Image/annotation verification
├── generate_missing_labels.py # Empty annotation generation
├── organize_missing_annotations.py
├── Report.docx                # Technical report
├── README.md
└── .gitignore
```

Large datasets, model checkpoints, and generated training outputs are stored locally and excluded from version control.

## Reproducibility

The repository contains scripts and notebooks used to prepare the dataset, train models, and evaluate results.

To reproduce the experiments:

1. Install Python and the required dependencies, including Ultralytics and PyTorch.
2. Obtain and organize the wildfire smoke images and corresponding YOLO annotations.
3. Run `prepare_experiment_5.py` to validate the annotations and generate the recording-based dataset split.
4. Open `notebooks/CS458_Experiment_5.ipynb` and configure the project directory.
5. Execute the training and evaluation cells.

Experiment 5 used the following training configuration:

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Image size | 960 |
| Epochs | 50 |
| Batch size | 4 |
| Random seed | 42 |
| Optimizer | Ultralytics automatic selection |
| Data-loader workers | 0 |
| Dataset caching | Disabled |

The training run took approximately 271 minutes on an NVIDIA GTX 1660 SUPER.

## Future Improvements

Potential future work includes:

- Evaluating performance on additional unseen wildfire recordings.
- Improving annotations for faint and distant smoke.
- Investigating data augmentation and alternative model architectures.
- Testing detection performance on continuous video streams.
- Developing a real-time smoke detection interface.

## Academic Context

**Course:** CS 458 — Data Mining  
**Institution:** University of Nevada, Reno  
**Year:** 2026

This repository documents the implementation, experimentation, and evaluation of a machine-learning approach to wildfire smoke detection.