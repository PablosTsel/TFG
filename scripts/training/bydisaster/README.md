# Disaster-Conditioned Damage Classification Model

This directory contains the implementation of a multi-task disaster-conditioned damage classification model as proposed by your tutor. The model combines disaster type classification with disaster-specific attention mechanisms to improve damage assessment.

## Main Components

1. **Model Architecture** (`disaster_model.py`):
   - `DisasterAttentionHead`: Attention mechanism specialized for a specific disaster type
   - `MultiHeadDisasterAttention`: Combines 9 disaster-specific attention heads and a disaster classifier
   - `DisasterConditionedDamageClassifier`: Complete model with pre/post UNets, multi-head attention, and damage decoder

2. **Dataset Implementation** (`disaster_dataset.py`):
   - `DisasterAwareDataset`: Extends the existing dataset to include disaster type labels
   - Supports balancing the dataset by disaster type
   - Maps specific disasters (e.g., "hurricane-harvey") to broader categories (e.g., "hurricane")

3. **Training Script** (`train_disaster_model.py`):
   - Multi-task training with weighted loss for damage segmentation and disaster classification
   - Comprehensive visualization and evaluation of both tasks
   - Saves confusion matrices, classification reports, and learning curves

## Key Features

- **Disaster-Specific Attention**: Each disaster type has its own specialized attention mechanism
- **Multi-Task Learning**: Simultaneously learns to classify disaster types and assess damage
- **Attention Selection**: Uses predicted disaster type to select the most appropriate attention head
- **Disaster Type Balancing**: Option to balance training data across disaster types

## How to Use

### Training the Model

To train the disaster-conditioned model with default parameters:

```bash
cd /path/to/project/root
python scripts/training/bydisaster/train_disaster_model.py
```

#### Important Parameters:

- `--data_dir`: Path to the xBD dataset directory (default: "data/xBD")
- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 30)
- `--lr`: Learning rate (default: 0.0002)
- `--damage_weight`: Weight for damage segmentation loss (default: 0.8)
- `--disaster_weight`: Weight for disaster classification loss (default: 0.2)
- `--balance_disasters`: Enable to balance dataset by disaster type

Example with custom parameters:

```bash
python scripts/training/bydisaster/train_disaster_model.py \
  --data_dir data/xBD \
  --batch_size 2 \
  --epochs 50 \
  --lr 0.0001 \
  --damage_weight 0.7 \
  --disaster_weight 0.3 \
  --balance_disasters
```

### Evaluation and Visualization

The training script automatically generates:

1. **Confusion matrices** for disaster type classification
2. **Classification reports** with precision, recall, and F1-score for each disaster type
3. **Visualizations** of model predictions showing both disaster classification and damage assessment
4. **Learning curves** for both tasks

Output files are saved to `output/disaster_model/disaster_runX/` where X is the run number.

## Extending the Model

### Using Weighted Combination of Heads

The current implementation uses the argmax approach (selecting the head with highest probability). To implement a weighted combination of heads, modify the `forward` method in `MultiHeadDisasterAttention` class:

```python
# Instead of selecting just the max head:
selected_outputs = torch.zeros_like(pre_feat)
for b in range(batch_size):
    # Use probabilities as weights for a weighted sum of all heads
    for head_idx in range(len(self.attention_heads)):
        weight = disaster_probs[b, head_idx]
        selected_outputs[b] += all_head_outputs[b, head_idx] * weight
```

### Adding More Disaster Types

To add new disaster types:

1. Update the `DISASTER_TYPES` list in `disaster_model.py`
2. Add the new disasters to the `DISASTER_MAPPING` dictionary in `disaster_dataset.py`

## Performance Expectations

The disaster-conditioned model is expected to provide several benefits:

1. **Improved damage assessment** by specializing attention for each disaster type
2. **Better generalization** to new examples of the same disaster type
3. **Interpretability** of which disaster patterns the model recognizes
4. **Auxiliary disaster type classification** as an additional model output

## References

This implementation is based on the approach proposed by your tutor, similar to the method described in:
- "Driving Under Challenging Conditions: Multi-task Attention-based Visual Environment Classification and Driver Gaze Estimation" (2020) 