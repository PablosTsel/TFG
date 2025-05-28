# Disaster Detection and Damage Assessment

This project focuses on analyzing satellite imagery to detect buildings and assess damage in disaster areas using the xView2 dataset.

## System Architecture

### Overall Two-Stage Pipeline

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Pre-disaster       │     │  Post-disaster       │     │                     │
│  Satellite Image    │     │  Satellite Image     │     │   Ground Truth      │
└──────────┬──────────┘     └──────────┬───────────┘     │   Damage Labels     │
           │                           │                   └─────────────────────┘
           │                           │                              
           ▼                           │                              
┌─────────────────────┐                │                              
│  STAGE 1:           │                │                              
│  Building Detector  │                │                              
│  (U-Net)           │                │                              
└──────────┬──────────┘                │                              
           │                           │                              
           ▼                           ▼                              
┌─────────────────────┐     ┌─────────────────────┐                 
│  Binary Building    │     │  Post-disaster      │                 
│  Mask               │     │  Image              │                 
└──────────┬──────────┘     └──────────┬──────────┘                 
           │                           │                              
           └───────────┬───────────────┘                              
                       ▼                                              
            ┌─────────────────────┐                                  
            │  STAGE 2:           │                                  
            │  Damage Classifier  │                                  
            │  (Dual U-Net +      │                                  
            │   Attention Fusion) │                                  
            └──────────┬──────────┘                                  
                       │                                              
                       ▼                                              
            ┌─────────────────────┐                                  
            │  5-Class Damage     │                                  
            │  Segmentation Map   │                                  
            │  • Background       │                                  
            │  • No Damage        │                                  
            │  • Minor Damage     │                                  
            │  • Major Damage     │                                  
            │  • Destroyed        │                                  
            └─────────────────────┘                                  
```

### Stage 1: Building Detection Architecture

```
                        Building Detector (Binary U-Net)
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  Input: Pre-disaster Image (3×256×256)                         │
    │                                                                 │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │
                    ENCODER       ▼        
    ┌─────────────────────────────────────────────────────────────────┐
    │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
    │  │Conv 3→64│───►│Conv     │───►│Conv     │───►│Conv     │     │
    │  │ ReLU    │    │128→128  │    │256→256  │    │512→512  │     │
    │  │BatchNorm│    │MaxPool2d│    │MaxPool2d│    │MaxPool2d│     │
    │  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘     │
    │       │              │              │              │            │
    │     256×256       128×128        64×64          32×32          │
    └───────┼──────────────┼──────────────┼──────────────┼────────────┘
            │              │              │              │
            │              │              │              ▼
            │              │              │      ┌──────────────┐
            │              │              │      │ Bottleneck   │
            │              │              │      │ Conv 512→1024│
            │              │              │      │   16×16      │
            │              │              │      └──────┬───────┘
            │              │              │              │
            │              │              │              ▼
            │              │              │         DECODER
    ┌───────┼──────────────┼──────────────┼────────────────────────────┐
    │       ▼              ▼              ▼                            │
    │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
    │  │UpConv + │◄───│UpConv + │◄───│UpConv + │◄───│UpConv   │     │
    │  │Skip Conn│    │Skip Conn│    │Skip Conn│    │1024→512  │     │
    │  │Conv 64  │    │Conv 128 │    │Conv 256 │    │          │     │
    │  └────┬────┘    └─────────┘    └─────────┘    └─────────┘     │
    │       │                                                         │
    │     256×256                                                     │
    └───────┼──────────────────────────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Output: Binary Mask (1×256×256)                               │
    │  • 0 = Background (No Building)                                │
    │  • 1 = Building                                                │
    │                                                                 │
    │  Loss: 0.5 × BCE + 0.5 × Dice                                 │
    └─────────────────────────────────────────────────────────────────┘
```

### Stage 2: Damage Classification Architecture

```
                    Improved Damage Classifier (Dual U-Net + Attention)
    ┌──────────────────────────────────┬──────────────────────────────────┐
    │   Pre-disaster Image (3×256×256) │  Post-disaster Image (3×256×256)  │
    └──────────────┬───────────────────┴───────────────┬──────────────────┘
                   │                                   │
                   ▼                                   ▼
    ┌─────────────────────────┐         ┌─────────────────────────┐
    │   Pre-disaster U-Net    │         │  Post-disaster U-Net    │
    │   Encoder → Decoder     │         │   Encoder → Decoder     │
    │   Output: 5×256×256     │         │   Output: 5×256×256     │
    └────────────┬────────────┘         └────────────┬────────────┘
                 │                                   │
                 ▼                                   ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    ATTENTION FUSION MODULE                     │
    │                                                                │
    │  ┌─────────────────────┐     ┌─────────────────────┐         │
    │  │  Spatial Attention  │     │  Channel Attention  │         │
    │  │  ┌─────────────┐   │     │  ┌─────────────┐   │         │
    │  │  │ Conv + Sigmoid│  │     │  │ AvgPool     │   │         │
    │  │  │ → Attention   │  │     │  │ MaxPool     │   │         │
    │  │  │   Map        │  │     │  │ Conv → Sigmoid│  │         │
    │  │  └─────────────┘   │     │  └─────────────┘   │         │
    │  └──────────┬──────────┘     └──────────┬──────────┘         │
    │             │                           │                     │
    │             ▼                           ▼                     │
    │  ┌──────────────────────────────────────────────┐           │
    │  │     Weighted Feature Combination +            │           │
    │  │     Residual Connection from Pre-features    │           │
    │  └──────────────────────┬───────────────────────┘           │
    │                         │                                    │
    └─────────────────────────┼─────────────────────────────────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Final Classifier   │
                   │  Conv 5→128→5       │
                   │  BatchNorm + ReLU   │
                   └──────────┬──────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Output: 5-Class Segmentation (5×256×256)                      │
    │  • Class 0: Background (No Building)                           │
    │  • Class 1: No Damage (Green)                                  │
    │  • Class 2: Minor Damage (Yellow)                               │
    │  • Class 3: Major Damage (Orange)                               │
    │  • Class 4: Destroyed (Red)                                     │
    │                                                                 │
    │  Loss: 0.6 × Focal Loss (γ=3) + 0.4 × Dice Loss              │
    │  Class Weights: [0.05, 2.0, 5.0, 3.0, 3.0]                    │
    └─────────────────────────────────────────────────────────────────┘
```

## Model Performance & Key Features

### Stage 1: Building Detection
- **Best IoU**: 0.85+ on validation set
- **Dynamic Threshold Optimization**: Tests thresholds from 0.3 to 0.6
- **Training Time**: ~30 epochs, 256×256 images
- **Batch Size**: 16

### Stage 2: Damage Classification
- **Mean IoU**: 0.73+ across all damage classes
- **Per-Class Performance**:
  - No Damage: ~0.85 accuracy
  - Minor Damage: ~0.68 accuracy (most challenging)
  - Major Damage: ~0.72 accuracy
  - Destroyed: ~0.78 accuracy
- **Training Time**: ~20 epochs with attention mechanism
- **Batch Size**: 4 (due to dual U-Net architecture)

### Key Innovations

1. **Two-Stage Approach**: Separates building detection from damage assessment for better accuracy
2. **Attention-Based Fusion**: Combines pre/post disaster features intelligently
3. **Class Balancing Strategy**:
   - Weighted sampling (minor damage: 25×, major/destroyed: 20×)
   - Custom loss weights focusing on underrepresented classes
   - Special polygon dilation for minor damage regions
4. **Combined Loss Functions**:
   - Stage 1: BCE + Dice for better boundary detection
   - Stage 2: Focal Loss (γ=3) + Dice for handling class imbalance

## Project Structure

```bash
.
├── data
│   ├── hold           
│   │   ├── images
│   │   │   └── ... (hold images)
│   │   └── labels
│   │       └── ... (hold JSONs)
│   ├── test         
│   │   ├── images
│   │   │   └── ... (test images: *_pre_disaster.png, *_post_disaster.png)
│   │   └── labels
│   │       └── ... (test JSON files)
│   └── xBD           
│       ├── guatemala-volcano
│       │   ├── images
│   │   │   ├── guatemala-volcano_00000000_pre_disaster.png
│   │   │   ├── guatemala-volcano_00000000_post_disaster.png
│   │   │   └── ... (other images)
│   │   └── labels
│   │       ├── guatemala-volcano_00000000_pre_disaster.json
│   │       ├── guatemala-volcano_00000000_post_disaster.json
│   │       └── ... (other JSONs)
│   │   └── hurricane-florence
│   │   └── ... etc.
├── output
│   ├── building_detector
│   │   └── detect_run2
│   │       ├── config_run2.txt
│   │       ├── models
│   │       │   └── best_threshold.txt
│   │       ├── training_metrics_run2.txt
│   │       └── visualizations
│   │           ├── learning_curves.png
│   │           ├── predictions_epoch_1.png
│   │           └── predictions_epoch_30.png
│   ├── dam_classifier
│   │   └── classifier_run4
│   │       ├── class_performance.txt
│   │       ├── config_run4.txt
│   │       ├── models
│   │       │   └── best_model_epoch_18.pt
│   │       ├── training_metrics_run4.txt
│   │       └── visualizations
│   │           ├── class_accuracy.png
│   │           ├── learning_curves.png
│   │           ├── predictions_epoch_1.png
│   │           └── predictions_epoch_18.png
│   └── experiments
│       ├── building_detector
│       │   ├── baseline_bce_only
│       │   │   ├── config.txt
│       │   │   ├── models
│       │   │   │   ├── best_model_epoch_9.pt
│       │   │   │   └── best_threshold.txt
│       │   │   ├── training_metrics.txt
│       │   │   └── visualizations
│       │   │       ├── learning_curves.png
│       │   │       ├── predictions_epoch_1.png
│       │   │       └── predictions_epoch_9.png
│       │   ├── dice_loss_only
│       │   │   ├── config.txt
│       │   │   ├── models
│       │   │   │   ├── best_model_epoch_9.pt
│       │   │   │   └── best_threshold.txt
│       │   │   ├── training_metrics.txt
│       │   │   └── visualizations
│       │   │       ├── learning_curves.png
│       │   │       ├── predictions_epoch_1.png
│       │   │       ├── predictions_epoch_3.png
│       │   │       └── ... (other epoch predictions)
│       │   └── no_augmentation
│       │       ├── config.txt
│       │       ├── models
│       │       │   ├── best_model_epoch_10.pt
│       │       │   └── best_threshold.txt
│       │       ├── training_metrics.txt
│       │       └── visualizations
│       │           ├── learning_curves.png
│       │           ├── predictions_epoch_10.png
│       │           ├── predictions_epoch_1.png
│       │           └── ... (other epoch predictions)
│       └── damage_classifier
│           ├── no_weighting
│           │   ├── class_performance.txt
│           │   ├── config.txt
│           │   ├── models
│           │   │   └── best_model_epoch_10.pt
│           │   ├── training_metrics.txt
│           │   └── visualizations
│           │       ├── class_accuracy.png
│           │       ├── learning_curves.png
│           │       └── ... (epoch predictions)
│           ├── simple_concat
│           │   ├── class_performance.txt
│           │   ├── config.txt
│           │   ├── models
│           │   │   └── best_model_epoch_9.pt
│           │   ├── training_metrics.txt
│           │   └── visualizations
│           │       ├── class_accuracy.png
│           │       ├── learning_curves.png
│           │       └── ... (epoch predictions)
│           └── single_input
│               ├── class_performance.txt
│               ├── config.txt
│               ├── models
│               │   └── best_model_epoch_4.pt
│               ├── training_metrics.txt
│               └── visualizations
│                   ├── class_accuracy.png
│                   ├── learning_curves.png
│                   └── ... (epoch predictions)
├── scripts
│   ├── inference
│   │   └── evaluate_two_stage.py
│   ├── preprocessing
│   │   └── prepr.ipynb
│   └── training
│       ├── experiments
│       │   ├── classifier_noWeighting.py
│       │   ├── classifier_simpleConcat.py
│       │   ├── classifier_singleInput.py
│       │   ├── detector_baseline.py
│       │   ├── detector_lossFunction.py
│       │   └── detector_noAugmentation.py
│       ├── individual
│       │   ├── evaluate_baseline.py
│       │   ├── localization_model.py
│       │   └── train_baseline.py
│       ├── train_building_detector.py
│       ├── train_dam_classifier.py
│       └── utils.py
├── requirements.txt
├── xview_geotransforms.json
└── xview_tile_centers.geojson
```

## About xView2 Dataset

### Our Mission: Humanitarian Assistance & Disaster Recovery

When a disaster strikes, quick and accurate situational information is critical to an effective response. Before responders can act in the affected area, they need to know the location, cause and severity of damage. But disasters can strike anywhere, disrupting local communication and transportation infrastructure, making the process of assessing specific local damage difficult, dangerous, and slow.

### Raw Imagery is Not Enough

Satellite imagery can provide unbiased overhead views, but raw imagery is not enough to inform recovery efforts. High-resolution imagery is required to see specific damage conditions, but because disasters cover a large ground area, analysts must search through huge swaths of pixel space to localize and score damage in the area of interest. Then annotated imagery must be summarized and communicated to the recovery team. It is a slow and laborious process.

### What is xView?

The xView2 Challenge focuses on automating the process of assessing building damage after a natural disaster, an analytical bottleneck in the post-disaster workflow. To enable the Challenge and stimulate applied research in the computer vision community, we are releasing one of the largest and highest-quality publicly available dataset of high-resolution satellite imagery annotated with building locations and damage scores before and after natural disasters.

## Project Overview

This project is focused on disaster detection using satellite imagery from the xView2 dataset. The goal is to develop a machine learning model to predict and classify disaster areas from satellite images.

## Environment Setup

- Python Version: 3.11.2
- CUDA Version: 12.8
- cuDNN Version: 9.6.0
- PyTorch version: 2.5.1+cu121
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
