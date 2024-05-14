# FCOS Object Detector

## Overview
This repository contains a PyTorch implementation of the Fully Convolutional One-Stage (FCOS) object detection system. FCOS is distinguished by its use of a fully convolutional network that predicts object boundaries directly from the feature maps without relying on predefined anchor boxes. This makes FCOS simpler and generally more flexible compared to traditional anchor-based detection systems.

## Features
- **Anchor-Free Detection**: FCOS eliminates the need for anchor generation, simplifying the training pipeline and reducing hyperparameter tuning.
- **Feature Pyramid Network (FPN)**: Incorporates FPN to leverage multi-scale information for robust detection across various object sizes.
- **Real-Time Performance**: Optimized for efficiency, suitable for real-time object detection tasks.

## Module Description
- `fcos.py`: Contains the main implementation of the FCOS detection system including the backbone network, prediction head, and post-processing steps.


## Installation
```bash
git clone https://github.com/AnmolGulati6/fcos-object-detector.git
cd fcos-object-detector
```

## Usage
### Training
```bash
python train.py --data_path /path/to/your/dataset --epochs 50
```

### Inference
```bash
python infer.py --model_path /path/to/saved/model --image_path /path/to/image.jpg
```

## Contributing
Contributions to improve the FCOS implementation are welcome. Please submit a pull request or open an issue to discuss potential changes or additions.

