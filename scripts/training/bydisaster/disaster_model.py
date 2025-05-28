#!/usr/bin/env python3
# Disaster-Conditioned Damage Classification Model
# This model uses disaster-specific attention heads to improve damage assessment

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

# Import from project modules
from scripts.training.utils import UNet

# List of disaster types
DISASTER_TYPES = [
    "fire",
    "tsunami",
    "tornado",
    "wildfire",
    "bushfire",
    "flooding",
    "earthquake",
    "volcano", 
    "hurricane"
]

class DisasterAttentionHead(nn.Module):
    """
    Individual attention head for a specific disaster type
    """
    def __init__(self, in_channels):
        super(DisasterAttentionHead, self).__init__()
        
        # Spatial attention path
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel attention path
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, pre_feat, post_feat):
        # Concatenate features for spatial attention
        concat_feat = torch.cat([pre_feat, post_feat], dim=1)
        
        # Spatial attention
        spatial_attn = self.spatial_attention(concat_feat)
        weighted_post_spatial = post_feat * spatial_attn
        
        # Channel attention
        avg_pool = self.avg_pool(concat_feat)
        max_pool = self.max_pool(concat_feat)
        channel_attn = self.channel_attention(torch.cat([avg_pool, max_pool], dim=1))
        weighted_post_channel = post_feat * channel_attn
        
        # Combine weighted features
        combined = torch.cat([weighted_post_spatial, weighted_post_channel], dim=1)
        fused_feat = self.fusion_conv(combined)
        
        # Add residual connection from pre_feat
        fused_feat = fused_feat + pre_feat
        
        return fused_feat

class MultiHeadDisasterAttention(nn.Module):
    """
    Multi-head attention module with one head per disaster type
    """
    def __init__(self, in_channels, num_disaster_types=9):
        super(MultiHeadDisasterAttention, self).__init__()
        
        # Create one attention head per disaster type
        self.attention_heads = nn.ModuleList([
            DisasterAttentionHead(in_channels) for _ in range(num_disaster_types)
        ])
        
        # Disaster classifier module (takes concatenated features, outputs disaster probabilities)
        self.disaster_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_disaster_types)
        )
        
    def forward(self, pre_feat, post_feat):
        # Get disaster type prediction
        concat_feat = torch.cat([pre_feat, post_feat], dim=1)
        disaster_logits = self.disaster_classifier(concat_feat)
        disaster_probs = F.softmax(disaster_logits, dim=1)
        
        # In training mode, process all attention heads
        outputs_from_all_heads = []
        for head in self.attention_heads:
            head_output = head(pre_feat, post_feat)
            outputs_from_all_heads.append(head_output.unsqueeze(1))
            
        # Stack outputs along a new dimension
        all_head_outputs = torch.cat(outputs_from_all_heads, dim=1)  # [B, num_heads, C, H, W]
        
        # Option 1: Select the head with highest probability (argmax approach)
        # In inference, we'll only use the top predicted disaster head
        batch_size = disaster_probs.size(0)
        
        # Get the index of highest probability for each sample in batch
        _, max_indices = disaster_probs.max(dim=1)  # [B]
        
        # Select the corresponding head output for each sample
        selected_outputs = torch.zeros_like(pre_feat)
        for b in range(batch_size):
            selected_head_idx = max_indices[b].item()
            selected_outputs[b] = all_head_outputs[b, selected_head_idx]
        
        return selected_outputs, disaster_logits

class DisasterConditionedDamageClassifier(nn.Module):
    """
    Multi-task damage classifier that conditions on disaster type
    """
    def __init__(self, in_channels=3, out_channels=5, num_disaster_types=9):
        super(DisasterConditionedDamageClassifier, self).__init__()
        
        # Pre-disaster branch - full UNet
        self.pre_encoder = UNet(in_channels=in_channels, out_channels=out_channels)
        
        # Post-disaster branch - full UNet
        self.post_encoder = UNet(in_channels=in_channels, out_channels=out_channels)
        
        # Store the output channels for later use
        self.out_channels = out_channels
        self.num_disaster_types = num_disaster_types
        
        # Multi-head attention fusion module for combining pre and post features
        self.attention_fusion = MultiHeadDisasterAttention(
            in_channels=out_channels,
            num_disaster_types=num_disaster_types
        )
        
        # Final convolution layers to produce class predictions
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )
        
    def forward(self, pre_img, post_img):
        # Get features from both UNets
        pre_features = self.pre_encoder(pre_img)
        post_features = self.post_encoder(post_img)
        
        # Fuse features with multi-head attention mechanism
        fused_features, disaster_logits = self.attention_fusion(pre_features, post_features)
        
        # Final prediction
        damage_logits = self.final_conv(fused_features)
        
        # Return both damage and disaster predictions
        return damage_logits, disaster_logits 