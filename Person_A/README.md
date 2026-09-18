# Person 1 — Computer Vision & Embedding Module

## Overview

This module implements the computer vision pipeline for road damage classification using the RDD2022 dataset. It is designed as a standalone Python component that outputs image embeddings for downstream Spark integration.

## Features

* Image preprocessing (224×224 resize + ImageNet normalization)
* MobileNetV2 frozen-backbone feature extraction
* Multi-scale feature fusion (1312-dimensional embedding)
* ResNet18 embedding extractor (alternative implementation)
* Fine-tuned MobileNetV2 classifier
* Evaluation using Accuracy, Weighted F1, Per-class F1, and Confusion Matrix

## API

`extract_embedding(image_path) → numpy.ndarray`

Returns a 1312-dimensional embedding vector that can be consumed by the Spark pipeline.

## Results

| Metric              |  Score |
| ------------------- | -----: |
| 5-Class Accuracy    | 63.48% |
| 5-Class Weighted F1 | 62.60% |
| 4-Class Accuracy    | 65.29% |
| 4-Class Weighted F1 | 65.54% |

## Notes

The published paper reports **78% accuracy** on the India subset using a different experimental setup. The current implementation reproduces the feature-extraction pipeline and evaluation framework but is **not directly comparable** because the dataset split and training pipeline differ.
