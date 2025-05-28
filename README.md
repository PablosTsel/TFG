(myenv) pablos@debian:~/Documents/uc3m/DammageAssesment$ tree -I "__pycache__|data|Literature|txtFiles|myenv"
.
.
├── data
│   ├── hold           
│   │   ├── images
│   │   │   └── … (hold images)
│   │   └── labels
│   │       └── … (hold JSONs)
│   ├── test         
│   │   ├── images
│   │   │   └── … (test images: *_pre_disaster.png, *_post_disaster.png)
│   │   └── labels
│   │       └── … (test JSON files)
│   └── xBD           
│       ├── guatemala-volcano
│       │   ├── images
│       │   │   ├── guatemala-volcano_00000000_pre_disaster.png
│       │   │   ├── guatemala-volcano_00000000_post_disaster.png
│       │   │   └── … (other images)
│       │   └── labels
│       │       ├── guatemala-volcano_00000000_pre_disaster.json
│       │       ├── guatemala-volcano_00000000_post_disaster.json
│       │       └── … (other JSONs)
│       ├── hurricane-florence
│       ├── … etc.
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
│   │           ├── predictions_epoch_30.png
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
│   │           ├── predictions_epoch_18.png
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
│       │   │       ├── predictions_epoch_4.png
│       │   │       ├── predictions_epoch_5.png
│       │   │       ├── predictions_epoch_6.png
│       │   │       ├── predictions_epoch_7.png
│       │   │       └── predictions_epoch_9.png
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
│       │           ├── predictions_epoch_3.png
│       │           ├── predictions_epoch_4.png
│       │           ├── predictions_epoch_5.png
│       │           ├── predictions_epoch_6.png
│       │           ├── predictions_epoch_7.png
│       │           └── predictions_epoch_8.png
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
│           │       ├── predictions_epoch_10.png
│           │       ├── predictions_epoch_1.png
│           │       ├── predictions_epoch_2.png
│           │       ├── predictions_epoch_8.png
│           │       └── predictions_epoch_9.png
│           ├── simple_concat
│           │   ├── class_performance.txt
│           │   ├── config.txt
│           │   ├── models
│           │   │   └── best_model_epoch_9.pt
│           │   ├── training_metrics.txt
│           │   └── visualizations
│           │       ├── class_accuracy.png
│           │       ├── learning_curves.png
│           │       ├── predictions_epoch_1.png
│           │       ├── predictions_epoch_2.png
│           │       ├── predictions_epoch_7.png
│           │       ├── predictions_epoch_8.png
│           │       └── predictions_epoch_9.png
│           └── single_input
│               ├── class_performance.txt
│               ├── config.txt
│               ├── models
│               │   └── best_model_epoch_4.pt
│               ├── training_metrics.txt
│               └── visualizations
│                   ├── class_accuracy.png
│                   ├── learning_curves.png
│                   ├── predictions_epoch_1.png
│                   ├── predictions_epoch_2.png
│                   └── predictions_epoch_4.png
├── README.md
├── requirements.txt
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
├── xview_geotransforms.json
└── xview_tile_centers.geojson










Our Mission:
Humanitarian Assistance & Disaster Recovery
When a disaster strikes, quick and accurate situational information is critical to an effective response. Before responders can act in the affected area, they need to know the location, cause and severity of damage. But disasters can strike anywhere, disrupting local communication and transportation infrastructure, making the process of assessing specific local damage difficult, dangerous, and slow.

Raw Imagery is Not Enough
Satellite imagery can provide unbiased overhead views, but raw imagery is not enough to inform recovery efforts. High-resolution imagery is required to see specific damage conditions, but because disasters cover a large ground area, analysts must search through huge swaths of pixel space to localize and score damage in the area of interest. Then annotated imagery must be summarized and communicated to the recovery team. It is a slow and laborious process.

Solving a Common Problem
Recognizing an opportunity to solve a key analytical bottleneck, the Defense Innovation Unit, together with other Humanitarian Assistance and Disaster Recovery (HADR) organizations, is releasing a new labeled, high-resolution satellite dataset and a challenge to the computer vision community.

What is xView?
A Dataset and Challenge
The xView2 Challenge focuses on automating the process of assessing building damage after a natural disaster, an analytical bottleneck in the post-disaster workflow. To enable the Challenge and stimulate applied research in the computer vision community, we are releasing one of the largest and highest-quality publicly available dataset of high-resolution satellite imagery annotated with building locations and damage scores before and after natural disasters.

Building Capability Together
The 2018 xView Challenge focused on object detection---predicting bounding boxes for a wide variety of different types objects in high-resolution satellite imagery. Thousands of submissions were benchmarked, accelerating more than 70 years of computational evaluation into three months. A diverse community of AI/ML practitioners from around the world participated in the Challenge, producing models that were a significant improvement beyond the government baseline. xView has since been used in hackathons, tutorials, rapid prototyping, to inform disaster recovery efforts, and as an example for how artificial intelligence and machine learning (AI/ML) technology can be used to accelerate analytical capabilities in Humanitarian Assistance and Disaster Recovery.

1. Project Overview
This project is focused on disaster detection using satellite imagery from the xView2 dataset. The goal is to develop a machine learning model to predict and classify disaster areas from satellite images.

2. Environment Setup
Python Version: 3.11.2
CUDA Version: 12.8
cuDNN Version: 9.6.0
PyTorch version: 2.5.1+cu121
GPU name: NVIDIA GeForce RTX 4070 Laptop GPU
environment: myenv
