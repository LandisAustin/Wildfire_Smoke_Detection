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
- Investigating confidence thresholds and their effects on detection performance.
- Performing quantitative and visual error analysis to identify model limitations.

## Technologies Used

- **Python** — Dataset processing, training, and evaluation
- **Ultralytics YOLO11** — Object detection architecture
- **PyTorch** — Deep learning framework
- **OpenCV** — Image processing and visualization
- **Pandas** — Dataset analysis and evaluation
- **Matplotlib** — Training and evaluation visualizations
- **Jupyter Notebook / VS Code** — Experiment development
- **CUDA** — GPU-accelerated training

The final experiments were trained locally using an NVIDIA GeForce GTX 1660 SUPER GPU with 6 GB of dedicated VRAM.

## Dataset

The final dataset contains **3,758 images** from 103 recording groups, including smoke-positive and smoke-negative examples.

| Category | Images |
|---|---:|
| Smoke-positive | 2,465 |
| Smoke-negative | 1,293 |
| **Total** | **3,758** |

Images were annotated using the YOLO bounding-box format, with a single detection class: `smoke`.

Smoke-negative images contain empty annotation files, allowing the model to learn from images without visible wildfire smoke.

### Dataset Preparation

To reduce data leakage, the dataset was split by recording rather than randomly assigning individual frames.

This ensures that frames from the same recording do not appear in both training and validation.

| Split | Images | Recordings | Positive | Negative |
|---|---:|---:|---:|---:|
| Training | 3,006 | 86 | 2,026 | 980 |
| Validation | 752 | 17 | 439 | 313 |

Dataset verification confirmed that there were **zero overlapping recording groups** between the training and validation sets.

The dataset preparation script validates annotations, identifies recording groups, generates the train/validation split, and saves a manifest for reproducibility.

**Note:** The full image dataset is excluded from Git because of its size. Dataset preparation scripts are included in the repository.

## Experiments

Seven experiments were conducted following an initial baseline to investigate different training configurations.

| Experiment | Model | Resolution | Training configuration |
|---|---|---|---|
| Baseline | YOLO11n | 640 | Original annotated dataset |
| Experiment 1 | YOLO11n | 960 | Increased image resolution |
| Experiment 2 | YOLO11n | 640 | Added 180 negative images |
| Experiment 3 | YOLO11s | 960 | Larger model architecture |
| Experiment 4 | YOLO11n | 960 | Expanded negative training examples |
| Experiment 5 | YOLO11n | 960 | Full expanded dataset with new recording-based split |
| Experiment 6 | YOLO11s | 960 | Larger architecture trained on the expanded dataset |
| Experiment 7 | YOLO11n | 1280 | High-resolution fine-tuning of Experiment 5 |

The baseline and Experiments 1–4 used the original validation dataset of 191 images.

Experiments 5–7 used the expanded dataset and a new validation split of 752 images. Their evaluation results are therefore reported separately rather than treated as a direct controlled comparison against the earlier experiments.

An additional inference-only evaluation was performed using Experiment 5's existing weights at a resolution of 1280 to investigate whether increasing inference resolution alone could improve detection performance.

## Results

Model performance was evaluated using precision, recall, F1 score, mAP50, and mAP50:95.

**mAP50:95** was selected as the primary evaluation metric because it measures detection performance across multiple bounding-box overlap thresholds.

F1 scores in the experiment tables were calculated from the reported mean precision and recall.

### Baseline and Experiments 1–4 — Original Dataset

| Experiment | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.6197 | 0.3413 | 0.4402 | 0.3689 | 0.1639 |
| Experiment 1 | 0.5661 | 0.3894 | 0.4615 | 0.3923 | **0.1826** |
| Experiment 2 | 0.6197 | 0.3508 | 0.4480 | 0.3313 | 0.1466 |
| Experiment 3 | 0.4974 | 0.3576 | 0.4161 | 0.3558 | 0.1604 |
| Experiment 4 | 0.5748 | **0.4921** | **0.5303** | **0.4543** | 0.1738 |

Experiment 1 achieved the highest mAP50:95 on the original validation set, demonstrating the potential benefits of increasing image resolution from 640 to 960.

Experiment 4 achieved the highest recall, F1, and mAP50 on that same validation set.

### Experiments 5–7 — Expanded Dataset

| Experiment | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|---:|
| Experiment 5 — YOLO11n | 0.6539 | 0.5164 | 0.5771 | 0.5793 | 0.2937 |
| Experiment 6 — YOLO11s | **0.6565** | **0.5376** | **0.5911** | **0.5838** | **0.3091** |
| Experiment 7 — High-Resolution Fine-Tuning | 0.6209 | 0.5260 | 0.5695 | 0.5391 | 0.2816 |

Experiment 6 achieved the highest performance across all reported metrics on the expanded validation dataset.

Compared to Experiment 5, Experiment 6 achieved a **5.24% relative improvement in mAP50:95**. However, its training time increased from approximately 4.5 hours to over 48 hours on the available hardware.

Experiment 5 remains a practical alternative because of its smaller architecture and substantially lower training cost.

Because the validation dataset changed after Experiment 4, the results do not establish a controlled improvement over the earlier models.

### High-Resolution Inference and Fine-Tuning

Following an error analysis of Experiment 6, additional evaluations were conducted to investigate whether increasing image resolution could improve the detection of small and distant smoke plumes.

Experiment 5 was first evaluated at 1280 resolution without modifying its trained weights. Experiment 7 then fine-tuned those weights at 1280 resolution for an additional 20 epochs.

| Configuration | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|---:|
| Experiment 5 — Original 960 | **0.6539** | 0.5164 | **0.5771** | **0.5793** | **0.2937** |
| Experiment 5 — 1280 Inference | 0.6107 | 0.5011 | 0.5505 | 0.5397 | 0.2696 |
| Experiment 7 — 1280 Fine-Tuning | 0.6209 | **0.5260** | 0.5695 | 0.5391 | 0.2816 |

Increasing inference resolution without retraining resulted in an 8.21% relative decrease in mAP50:95.

High-resolution fine-tuning recovered some of this performance, improving mAP50:95 by 4.45% relative to the inference-only evaluation. However, it did not outperform the original Experiment 5 configuration.

These results demonstrate that increasing image resolution does not necessarily improve detection performance. Because Experiment 7 also changed the training duration, optimizer configuration, and batch size, the effect of resolution cannot be isolated completely.

## Confidence Threshold Analysis

Experiment 6 was evaluated at multiple confidence thresholds to investigate the trade-off between precision and recall.

The following metrics were calculated directly from the confusion matrix at each confidence threshold.

| Confidence | True Positives | False Positives | False Negatives | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0.10 | 278 | 328 | 161 | 0.4587 | 0.6333 | 0.5321 |
| 0.15 | 271 | 244 | 168 | 0.5262 | 0.6173 | 0.5681 |
| 0.20 | 263 | 196 | 176 | 0.5730 | 0.5991 | 0.5857 |
| **0.25** | **246** | **149** | **193** | **0.6228** | **0.5604** | **0.5899** |
| 0.30 | 228 | 110 | 211 | 0.6746 | 0.5194 | 0.5869 |
| 0.40 | 195 | 64 | 244 | 0.7529 | 0.4442 | 0.5587 |
| 0.50 | 154 | 28 | 285 | 0.8462 | 0.3508 | 0.4960 |

A confidence threshold of **0.25** achieved the highest F1 score among the tested thresholds.

Lower confidence thresholds increased recall but generated more false-positive detections. Higher thresholds improved precision but caused more smoke objects to be missed.

For wildfire detection, selecting an appropriate confidence threshold is particularly important because missed detections and false alarms have different practical consequences.

The optimal deployment threshold may differ depending on the relative importance of detection sensitivity and false-alarm prevention.

## Error Analysis

Additional quantitative and visual error analysis was performed on Experiment 6 using a confidence threshold of 0.25.

The analysis investigated whether incorrect predictions were primarily caused by background confusion, inaccurate bounding-box localization, or missed smoke detections.

### Smoke-Negative Images

The model was evaluated on all 313 smoke-negative validation images.

| Metric | Result |
|---|---:|
| Smoke-negative images evaluated | 313 |
| Images without false detections | 309 |
| Images with false detections | 4 |
| Total false-positive bounding boxes | 4 |
| Image-level false-positive rate | **1.28%** |
| Image-level specificity | **98.72%** |

The model successfully ignored the majority of images without annotated smoke.

Additionally, the four false-positive images originated from only two recording sequences, suggesting that certain environmental conditions may contribute disproportionately to background confusion.

### Smoke-Positive Images

The model was also evaluated on all 439 smoke-positive validation images.

Predicted bounding boxes were compared against ground-truth annotations using Intersection over Union (IoU), with a matching threshold of 0.50.

| Metric | Count |
|---|---:|
| Ground-truth smoke objects | 439 |
| Predicted bounding boxes | 345 |
| Correct detections | 242 |
| Poor localization | 83 |
| Duplicate detections | 5 |
| Background detections | 15 |
| False negatives | 197 |
| Completely missed smoke objects | 138 |

Of the 103 incorrect predicted bounding boxes, **80.6% were attributed to poor localization**.

Additionally, 138 of the 439 annotated smoke objects had no overlapping prediction at the selected confidence threshold.

Completely missed smoke objects are a subset of false negatives rather than an additional error category.

The custom error analysis uses its own IoU matching procedure and single-image inference configuration, so its counts may differ from those generated by Ultralytics' standard confusion matrix.

### Visual Error Analysis

Sampled false-negative and poorly localized predictions were visually inspected by displaying both the predicted bounding boxes and ground-truth annotations.

Several common challenges were identified:

- Extremely small or distant smoke plumes.
- Faint smoke blending into atmospheric haze.
- Low visual contrast between smoke and surrounding terrain.
- Difficulty defining the boundaries of diffuse smoke plumes.
- Small bounding-box positioning errors resulting in disproportionately large reductions in IoU.

Many sampled false-negative examples were difficult to distinguish visually even when the ground-truth annotations were provided.

The analysis suggested that improving small-object detection and bounding-box localization may be more valuable than simply increasing the number of smoke-negative training images.

These observations motivated the high-resolution inference evaluation and Experiment 7.

## Key Findings

- Increasing image resolution from 640 to 960 improved mAP50:95 in the initial experiments.
- Adding more smoke-negative examples in Experiment 4 was associated with improved recall and F1, although bounding-box localization remained challenging.
- Increasing model size from YOLO11n to YOLO11s did not improve performance on the original dataset but improved every reported metric on the expanded dataset.
- Experiment 6 achieved the highest overall validation performance, with an mAP50:95 of 0.3091.
- Experiment 5 provided a more computationally efficient alternative, achieving an mAP50:95 of 0.2937 with substantially less training time.
- Increasing inference resolution from 960 to 1280 reduced detection performance.
- High-resolution fine-tuning partially recovered performance but did not outperform the original 960-resolution configuration.
- A confidence threshold of 0.25 provided the highest F1 score among the tested thresholds.
- Experiment 6 achieved an image-level false-positive rate of only 1.28% on smoke-negative validation images at the selected threshold.
- Poor bounding-box localization and missed faint or distant smoke were more significant observed challenges than false detections on smoke-negative images.
- Training on the expanded dataset produced promising results, but further evaluation on a common held-out test set would be needed for a rigorous comparison with the earlier experiments.

## Repository Structure

```text
Wildfire_Smoke_Project/
├── notebooks/                  # Training and evaluation notebooks
├── image_scripts/              # Dataset preparation and verification
├── results/                    # Selected experiment results
├── requirements.txt            # Python dependencies
├── Report.docx                 # Technical report
├── README.md
└── .gitignore
```

Large datasets, model checkpoints, and unselected generated training outputs are stored locally and excluded from version control.

Selected experiment results and visualizations are organized separately for documentation and comparison.

## Reproducibility

The repository contains scripts and notebooks used to prepare the dataset, train models, and evaluate results.

To reproduce the expanded-dataset experiments:

1. Install Python and the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure that a compatible CUDA-enabled PyTorch installation is available if GPU acceleration is required.
3. Obtain and organize the wildfire smoke images and corresponding YOLO annotations.
4. Run `prepare_experiment_5.py` to validate the annotations and generate the recording-based dataset split.
5. Open `notebooks/CS458_Experiment_5.ipynb` and configure the project directory.
6. Execute the dataset verification, training, and evaluation cells.
7. Execute the confidence threshold and error analysis cells to reproduce the additional evaluations.

### Final Training Configurations

| Parameter | Experiment 5 | Experiment 6 | Experiment 7 |
|---|---|---|---|
| Model | YOLO11n | YOLO11s | YOLO11n |
| Starting weights | Pretrained | Pretrained | Experiment 5 |
| Image size | 960 | 960 | 1280 |
| Epochs | 50 | 50 | 20 |
| Batch size | 4 | 2 | 2 |
| Random seed | 42 | 42 | 42 |
| Data-loader workers | 0 | 0 | 0 |
| Dataset caching | Disabled | Disabled | Disabled |

Experiment 5 used Ultralytics' automatic optimizer selection.

Experiment 7 used the AdamW optimizer with an initial learning rate of 0.001 and initialized training from Experiment 5's best checkpoint.

### Computational Cost

| Experiment | Model | Training time |
|---|---|---|
| Experiment 5 | YOLO11n | Approximately 271 minutes |
| Experiment 6 | YOLO11s | Over 48 hours |
| Experiment 7 | YOLO11n | 20 additional fine-tuning epochs |

Training was performed on an NVIDIA GeForce GTX 1660 SUPER GPU.

The substantial difference in training time between Experiments 5 and 6 highlights the practical computational trade-offs associated with larger model architectures on resource-constrained hardware. Training time may also be affected by batch size and GPU memory limitations.

## Final Model Selection

Two models were identified as the strongest candidates for further development.

### Experiment 6 — Highest Detection Performance

- Architecture: YOLO11s
- Input resolution: 960
- Precision: 0.6565
- Recall: 0.5376
- F1: 0.5911
- mAP50: 0.5838
- mAP50:95: **0.3091**
- Selected confidence threshold: 0.25

Experiment 6 is the preferred model when detection performance is the primary consideration.

### Experiment 5 — Computationally Efficient Alternative

- Architecture: YOLO11n
- Input resolution: 960
- Precision: 0.6539
- Recall: 0.5164
- F1: 0.5771
- mAP50: 0.5793
- mAP50:95: **0.2937**

Experiment 5 is a promising alternative for resource-constrained applications because of its smaller architecture and lower training cost. Further inference-speed benchmarking would be necessary to establish its suitability for real-time deployment.

## Future Improvements

Potential future work includes:

- Evaluating performance on additional unseen wildfire recordings using an independent test set.
- Expanding the dataset with challenging smoke-positive examples.
- Improving annotation consistency for faint and diffuse smoke plumes.
- Investigating alternative approaches to small-object detection.
- Exploring image tiling and region-based inference for distant smoke.
- Investigating data augmentation and alternative model architectures.
- Evaluating model inference speed and resource consumption.
- Testing detection performance on continuous video streams.
- Incorporating temporal information to distinguish developing smoke from static background features.
- Developing a real-time smoke detection interface.

## Academic Context

**Course:** CS 458 — Data Mining  
**Institution:** University of Nevada, Reno  
**Year:** 2026

This repository documents the implementation, experimentation, and evaluation of a machine-learning approach to wildfire smoke detection.