## HybridLiteNet: A Lightweight Multi-Branch Hybrid Network with Transformer Attention

> 🚀 Achieved **93.11% Accuracy** on CIFAR-10 with **only 0.8M parameters**.

HybridLiteNet is a compact yet powerful deep learning architecture that integrates multi-branch convolution, ASFF-based feature fusion, and lightweight Transformer blocks. Designed for efficiency and high accuracy, it is suitable for small-scale image recognition tasks like CIFAR-10, CIFAR-100, and TinyImageNet.

---

## 🌟 Highlights

- ✅ **Multi-Scale Convolution Branches**: Parallel 1×1, 3×3, and 5×5 convolution branches capture diverse receptive fields.
- ✅ **ASFF (Adaptive Spatial Feature Fusion)**: Learnable fusion weights across branches for optimal spatial representation.
- ✅ **MBConv Backbone**: Depthwise separable inverted residual blocks with SE attention for lightweight efficiency.
- ✅ **Performer-style Transformer Block**: Low-rank attention + Depthwise MLP for global feature modeling with minimal overhead.
- ✅ **Swish/Mish Activations + SE Attention**: Nonlinear expressiveness and channel reweighting enhance discrimination.
- ✅ **RMSNorm**: Lightweight normalization replacing BatchNorm in Transformer blocks.
- ✅ **Only ~0.8M parameters** with **93.11% accuracy on CIFAR-10**.

---

## 🧱 Model Architecture

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
