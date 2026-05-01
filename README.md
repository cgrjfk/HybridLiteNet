## HybridLiteNetV2: A Lightweight Multi-Branch Hybrid Network with Transformer Attention

> 🚀 Achieved **95.95% Accuracy** on CIFAR-10 with **only 0.99M parameters if use TTA the model can get 96.18%**.

HybridLiteNet is a compact yet powerful deep learning architecture that integrates multi-branch convolution, ASFF-based feature fusion, and lightweight Transformer blocks. Designed for efficiency and high accuracy, it is suitable for small-scale image recognition tasks like CIFAR-10, CIFAR-100, and TinyImageNet.

---

## 🧱  HybridLiteNetV1

```text
Input (3×32×32)
│
├── Multi-Branch Conv: conv1x1 / conv3x3 / conv5x5 (SE attention)
│   └── Output: 32+32+32 channels
│
├── ASFF Fusion (96 channels → 64)
│
├── Stem Conv (3×3 stride=2)
│
├── MBConv Backbone
│   ├── MBConv(64→32)
│   ├── MBConv(32→32)
│   ├── MBConv(32→48)
│   ├── MBConv(48→96)
│   ├── MBConv(96→128)
│   └── MBConv(128→192)
│
├── Feature Fusion (concat ASFF & backbone → 288)
│   └── Conv1×1 → 160 channels
│
├── Lightweight Transformer Block
│   └── RMSNorm + PerformerAttention + Depthwise MLP
│
├── Classification Head
│   └── Conv1×1 → BN → Swish → AvgPool → Dropout(0.5) → Linear(320→10)
│
└── Output (Logits, shape: [batch_size, 10])
 ```
## 🚀HybridLiteNetV2
# HybridLiteNetV2 Model Architecture 
## Overall Overview
**HybridLiteNetV2** is a lightweight, efficient hybrid vision model that combines **ConvNeXt convolutional blocks**, **ECA attention**, **Linear Attention Transformer**, and modern CNN design patterns. It’s optimized for small-scale image inputs (e.g., 32×32) and classification tasks, with a lightweight parameter count (~0.99M).

---

## 1. Core Component Layers
### ECA (Efficient Channel Attention)
A lightweight channel attention mechanism for enhancing critical feature channels without heavy computation:
- Takes input feature maps and applies global average pooling to compress spatial dimensions
- Uses a 1D convolution to capture local cross-channel interactions
- Generates channel-wise attention weights and multiplies them with input features (feature re-calibration)

### ConvNeXt Block
A modern, efficient convolutional building block inspired by Transformers:
- **Depthwise Conv (dwconv)**: Spatial convolution applied to each channel independently
- **BatchNorm**: Normalizes feature distributions
- **Pointwise Convolutions (pwconv1/pwconv2)**: Expands channels to 4× dimension then projects back to original
- **GELU Activation**: Smooth non-linearity
- **Residual Connection**: Preserves gradient flow and stabilizes training

### LinearAttention
Efficient linear-time self-attention (replaces standard quadratic attention):
- Uses ELU-based feature normalization (`phi(x) = elu(x)+1`)
- Computes attention via **matrix dot products** instead of softmax, reducing complexity from $O(N^2)$ to $O(N)$
- Multi-head design for parallel feature learning
- Final linear projection for output refinement

### TransBlock (Transformer Block with Linear Attention)
Transformer encoder layer optimized for efficiency:
- **LayerNorm**: Pre-normalization for stable training
- **LinearAttention**: Efficient global context modeling
- **MLP**: 2-layer feed-forward network (expands to 2× dimension, GELU activation)
- **Residual Connections**: For both attention and MLP branches

---

## 2. Full Model Structure (HybridLiteNetV2)
The model follows a **stem → 3 feature stages → 2 downsampling layers → classification head** pipeline, integrating CNN and Transformer hybrid features.

### (1) Stem Layer (Input Embedding)
First feature extraction for raw RGB images:
- `Conv2d(3→48)`: 3×3 convolution, stride=1, padding=1
- `BatchNorm2d` + `GELU`
- **Purpose**: Converts 3-channel input to 48-channel base features

### (2) Stage 1
Pure convolutional feature learning:
- 2× stacked ConvNeXtBlock (48 channels)
- ECA attention (48 channels)
- **Purpose**: Local feature extraction with channel enhancement

### (3) Downsample 1
Spatial downsampling + channel expansion:
- `Conv2d(48→96)`: 3×3 convolution, stride=2, padding=1
- **Output**: 96 channels, 16×16 spatial size (from 32×32)

### (4) Stage 2 + Linear Transformer Hybrid
**Convolution + Transformer hybrid design**:
- 2× stacked ConvNeXtBlock (96 channels)
- **Linear Transformer Branch**:
  1. Flatten spatial dimensions → `(B, N, C)` sequence
  2. TransBlock (LinearAttention + MLP) models global context
  3. Reshape back to 2D feature maps
- **Purpose**: Combines local CNN features and global Transformer context

### (5) Downsample 2
Second downsampling + channel expansion:
- `Conv2d(96→160)`: 3×3 convolution, stride=2, padding=1
- **Output**: 160 channels, 8×8 spatial size

### (6) Stage 3
Final high-level feature learning:
- 2× stacked ConvNeXtBlock (160 channels)
- ECA attention (160 channels)
- **Purpose**: Refines high-level semantic features

### (7) Classification Head
Maps final features to class scores:
1. 1×1 conv (160→256) + BatchNorm + GELU
2. Depthwise conv (256) + 1×1 conv (256→320)
3. Global Average Pooling (compresses to 1×1)
4. Flatten + Dropout(0.4) + Linear(320→num_classes)
- **Purpose**: Final classification with regularization

---

## 3. Forward Pass Flow
```
Input (1,3,32,32)
→ Stem (48,32,32)
→ Stage1 (ConvNeXt×2 + ECA)
→ Down1 (96,16,16)
→ Stage2 (ConvNeXt×2)
→ [Flatten → TransBlock → Reshape] (96,16,16)
→ Down2 (160,8,8)
→ Stage3 (ConvNeXt×2 + ECA)
→ Head → Output (num_classes)
```

## 4. Key Specifications
- Input size: 3×32×32 (RGB small images)
- Total parameters: ~**0.99M** (lightweight)
- Hybrid design: ConvNeXt (local) + Linear Transformer (global) + ECA (channel attention)
- Efficiency: Low computation via depthwise conv and linear attention

---
## TTA-result
```
TTA Accuracy (Enhanced for weak classes): 96.21% 
Baseline Accuracy with tta: 96.18% 
Improvement: +0.03% 
Weak classes (extra augmentations): ['bird', 'cat', 'dog'] 

              precision    recall  f1-score   support 

    airplane       0.97      0.96      0.96      1000 
  automobile       0.98      0.98      0.98      1000 
        bird       0.95      0.95      0.95      1000 
         cat       0.92      0.91      0.92      1000 
        deer       0.97      0.97      0.97      1000 
         dog       0.93      0.93      0.93      1000 
        frog       0.98      0.98      0.98      1000 
       horse       0.98      0.97      0.98      1000 
        ship       0.97      0.98      0.97      1000 
       truck       0.97      0.97      0.97      1000 

    accuracy                           0.96     10000 
   macro avg       0.96      0.96      0.96     10000 
weighted avg       0.96      0.96      0.96     10000
```

### Summary
- **HybridLiteNetV2** is a new lightweight hybrid CNN-Transformer model
- Core components: ECA, ConvNeXt, LinearAttention, TransBlock
- Architecture: Stem → 3 Stages → 2 Downsampling → Classification Head
- Strengths: High efficiency, small parameters, strong local-global feature fusion
- Output: Class logits for image classification
