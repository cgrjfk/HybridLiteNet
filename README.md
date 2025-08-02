

# HybridLiteNet (0.8M parameters)

A **lightweight hybrid CNN + Transformer model** designed for CIFAR-10.

## Key Techniques

* Multi-branch CNN stem (1×1 / 3×3 / 5×5 convolutions with SE attention)
* MBConv backbone (MobileNetV2 style)
* ASFF (Adaptive Spatial Feature Fusion) for multi-scale feature fusion
* Lightweight Performer-style attention block with RMSNorm
* Compact classification head

## Results on CIFAR-10

* **Parameters:** \~0.8M
* **Training:** 350 epochs
* **Accuracy:**

  * Train Acc: 89.68%
  * Test Acc: 93.06%
  * Best Acc: **93.11%**
```
Epoch [350/350]
Train Loss: 0.7386 | Train Acc: 89.68%
Test  Loss: 0.6776 | Test  Acc: 93.06%
```



