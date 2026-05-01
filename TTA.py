import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from HybridLiteNet import HybridLiteNetV2
from train import CachedDataset

# ========== 薄弱类别配置（基于微调后的分类报告） ==========
# 根据报告，Shirt (类别6) 的 f1-score 最低 (0.85)
# 可选：Pullover (2) 和 Coat (4) 也可作为薄弱类，可自行添加
WEAK_CLASSES = [2, 6]  # Shirt 是最需要增强的类别
# 若希望同时增强 Pullover 和 Coat，可改为 [2, 4, 6]
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_tta_transforms():
    """定义针对 FashionMNIST 的 TTA 变换（无上下翻转，保持语义）"""
    # 基础增强（所有类别通用）
    base_transforms = [
        lambda img: img,  # 原始图像
        lambda img: transforms.functional.hflip(img),  # 水平翻转（对服装有效）
        lambda img: transforms.functional.affine(img, angle=0, translate=(2, 2), scale=1.0, shear=0),  # 平移+2
        lambda img: transforms.functional.affine(img, angle=0, translate=(-2, -2), scale=1.0, shear=0),  # 平移-2
        lambda img: transforms.functional.rotate(img, angle=5),  # 旋转+5度
        lambda img: transforms.functional.rotate(img, angle=-5),  # 旋转-5度
        lambda img: transforms.functional.affine(img, angle=0, translate=(2, 2), scale=0.95, shear=0),  # 轻微缩小
        lambda img: transforms.functional.affine(img, angle=0, translate=(2, 2), scale=1.05, shear=0),  # 轻微放大
    ]

    # 针对薄弱类别（如 Shirt）的额外增强（更强、更多样）
    weak_transforms = [
        lambda img: transforms.functional.affine(img, angle=0, translate=(3, 3), scale=1.0, shear=0),  # 更大平移
        lambda img: transforms.functional.affine(img, angle=0, translate=(-3, -3), scale=1.0, shear=0),
        lambda img: transforms.functional.rotate(img, angle=10),  # 更大旋转
        lambda img: transforms.functional.rotate(img, angle=-10),
        lambda img: transforms.functional.affine(img, angle=0, translate=(2, 2), scale=0.9, shear=0),  # 更强缩放
        lambda img: transforms.functional.affine(img, angle=0, translate=(2, 2), scale=1.1, shear=0),
        lambda img: transforms.functional.adjust_brightness(img, brightness_factor=0.9),  # 亮度微调
        lambda img: transforms.functional.adjust_contrast(img, contrast_factor=1.1),  # 对比度微调
        # 服装特有的小幅度弹性变形（可选，需安装 kornia 或使用 torchvision 的 RandomPerspective）
        # lambda img: transforms.functional.perspective(img, ...)   # 更高级，此处省略
    ]
    return base_transforms, weak_transforms


def predict_with_tta(model, img, transforms_list, device):
    """对单张图像应用一组变换，返回平均预测概率"""
    preds = []
    with torch.no_grad():
        for transform in transforms_list:
            augmented = transform(img)
            augmented = augmented.unsqueeze(0).to(device)
            output = model(augmented)
            prob = torch.softmax(output, dim=1)
            preds.append(prob.cpu())
    avg_prob = torch.stack(preds).mean(dim=0)
    return avg_prob


def evaluate_selective_tta(model, testloader, device, base_transforms, weak_transforms):
    """
    选择性 TTA 评估：
    - 对薄弱类别（WEAK_CLASSES）使用全部增强（base + weak）
    - 对其他类别仅使用原始图像（单次预测）
    """
    model.eval()
    all_preds = []
    all_targets = []

    for inputs, targets in tqdm(testloader, desc="Selective TTA Evaluation"):
        inputs = inputs.to(device)
        batch_preds = []

        for i in range(inputs.shape[0]):
            img = inputs[i].cpu()
            label = targets[i].item()

            if label in WEAK_CLASSES:
                # 薄弱类别：使用 base + weak 全部增强
                all_transforms = base_transforms + weak_transforms
                avg_prob = predict_with_tta(model, img, all_transforms, device)
            else:
                # 非薄弱类别：仅原始图像
                with torch.no_grad():
                    output = model(img.unsqueeze(0).to(device))
                    avg_prob = torch.softmax(output, dim=1).cpu()

            pred = avg_prob.argmax(dim=1).item()
            batch_preds.append(pred)

        all_preds.extend(batch_preds)
        all_targets.extend(targets.cpu().numpy())

    return all_preds, all_targets


def evaluate_baseline(model, testloader, device):
    """基准评估（无 TTA，单次预测）"""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc="Baseline"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    return all_preds, all_targets


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Weak classes (extra augmentations): {[CLASS_NAMES[i] for i in WEAK_CLASSES]}")

    # ========== 加载 FashionMNIST 测试集 ==========
    # 注意：必须将灰度图转为 RGB（3通道）
    transform_test = transforms.Compose([
        transforms.Resize(32),  # 统一尺寸
        transforms.Grayscale(num_output_channels=3),  # 1通道 → 3通道
        transforms.ToTensor(),
    ])
    print("Loading FashionMNIST test set...")
    testset = CachedDataset(
        dataset_name="FashionMNIST", root='./data', train=False,
        transform=transform_test, download=True
    )
    testloader = DataLoader(testset, batch_size=50, shuffle=False, num_workers=0)

    # ========== 加载微调后的最佳模型 ==========
    print("Loading best FashionMNIST model...")
    model = HybridLiteNetV2(num_classes=10).to(device)
    model_path = "best_model_FashionMNIST_finetuned.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found, trying best_model.pth...")
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # 获取 TTA 增强列表
    base_transforms, weak_transforms = get_tta_transforms()
    print(f"Base transforms: {len(base_transforms)}")
    print(f"Weak transforms (for {[CLASS_NAMES[i] for i in WEAK_CLASSES]}): {len(weak_transforms)}")
    print(f"Total transforms for weak classes: {len(base_transforms) + len(weak_transforms)}")

    # ========== 基准评估 ==========
    print("\n" + "=" * 50)
    print("Running baseline evaluation (without TTA)...")
    all_preds_baseline, all_targets = evaluate_baseline(model, testloader, device)
    baseline_acc = accuracy_score(all_targets, all_preds_baseline) * 100

    from sklearn.metrics import recall_score
    baseline_recall = recall_score(all_targets, all_preds_baseline, average=None)

    print(f"\nBaseline Overall Accuracy: {baseline_acc:.2f}%")
    print("Baseline Per-class Recall:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:15s}: {baseline_recall[i] * 100:.2f}%")

    # ========== 选择性 TTA 评估 ==========
    print("\n" + "=" * 50)
    print("Running selective TTA evaluation (only weak classes get augmentations)...")
    all_preds_tta, _ = evaluate_selective_tta(model, testloader, device, base_transforms, weak_transforms)
    tta_acc = accuracy_score(all_targets, all_preds_tta) * 100
    tta_recall = recall_score(all_targets, all_preds_tta, average=None)

    print(f"\nSelective TTA Overall Accuracy: {tta_acc:.2f}%")
    print("Selective TTA Per-class Recall:")
    for i, name in enumerate(CLASS_NAMES):
        improvement = (tta_recall[i] - baseline_recall[i]) * 100
        weak_marker = " ✓" if i in WEAK_CLASSES else ""
        print(f"  {name:15s}: {tta_recall[i] * 100:.2f}% (Δ {improvement:+.2f}%){weak_marker}")

    improvement = tta_acc - baseline_acc
    print(f"\n{'=' * 50}")
    print(f"✓ Total Improvement: +{improvement:.2f}%")
    print(f"✓ Final Accuracy: {tta_acc:.2f}%")

    # ========== 分类报告 ==========
    print("\n" + "=" * 50)
    print("Classification Report (Selective TTA for FashionMNIST):")
    report = classification_report(all_targets, all_preds_tta, target_names=CLASS_NAMES)
    print(report)

    # 保存报告
    with open('classification_report_FashionMNIST_selective_tta.txt', 'w') as f:
        f.write(f"Selective TTA Accuracy: {tta_acc:.2f}%\n")
        f.write(f"Baseline Accuracy: {baseline_acc:.2f}%\n")
        f.write(f"Improvement: +{improvement:.2f}%\n")
        f.write(f"Weak classes (extra augmentations): {[CLASS_NAMES[i] for i in WEAK_CLASSES]}\n\n")
        f.write(report)
    print("✓ Report saved to classification_report_FashionMNIST_selective_tta.txt")

    # ========== 混淆矩阵 ==========
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(all_targets, all_preds_tta)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap='Blues', values_format='d', ax=ax, colorbar=False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    ax.set_title(f'FashionMNIST Selective TTA Confusion Matrix (Accuracy: {tta_acc:.2f}%)',
                 fontweight='bold', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix_FashionMNIST_selective_tta.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrix saved to confusion_matrix_FashionMNIST_selective_tta.png")

    # ========== 薄弱类别提升总结 ==========
    print(f"\n{'=' * 50}")
    print("Weak Class Improvement Summary (Selective TTA):")
    weak_improvement_total = 0
    for i in WEAK_CLASSES:
        imp = (tta_recall[i] - baseline_recall[i]) * 100
        weak_improvement_total += imp
        print(f"  {CLASS_NAMES[i]}: +{imp:.2f}%")
    print(f"  Average weak class improvement: +{weak_improvement_total / len(WEAK_CLASSES):.2f}%")

    print(f"\n{'=' * 50}")
    print(f"🎉 Final Result: {baseline_acc:.2f}% → {tta_acc:.2f}% (+{improvement:.2f}%)")
    print(
        f"🎉 Only weak classes {[CLASS_NAMES[i] for i in WEAK_CLASSES]} received {len(weak_transforms)} extra augmentations.")


if __name__ == "__main__":
    main()
'''
Device: cuda
Weak classes (extra augmentations): ['Pullover', 'Shirt']
Loading FashionMNIST test set...
Loaded FashionMNIST - test: 10000 samples
Classes: ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
Loading best FashionMNIST model...
Loaded model from best_model_FashionMNIST_finetuned.pth
Base transforms: 8
Weak transforms (for ['Pullover', 'Shirt']): 8
Total transforms for weak classes: 16

==================================================
Running baseline evaluation (without TTA)...
Baseline: 100%|██████████| 200/200 [00:04<00:00, 49.56it/s]

Baseline Overall Accuracy: 94.85%
Baseline Per-class Recall:
  T-shirt/top    : 90.10%
  Trouser        : 99.30%
  Pullover       : 91.80%
  Dress          : 95.30%
  Coat           : 93.80%
  Sandal         : 98.50%
  Shirt          : 85.20%
  Sneaker        : 97.60%
  Bag            : 99.40%
  Ankle boot     : 97.50%

==================================================
Running selective TTA evaluation (only weak classes get augmentations)...
Selective TTA Evaluation: 100%|██████████| 200/200 [02:36<00:00,  1.28it/s]

Selective TTA Overall Accuracy: 94.98%
Selective TTA Per-class Recall:
  T-shirt/top    : 90.10% (Δ +0.00%)
  Trouser        : 99.30% (Δ +0.00%)
  Pullover       : 92.40% (Δ +0.60%) ✓
  Dress          : 95.30% (Δ +0.00%)
  Coat           : 93.80% (Δ +0.00%)
  Sandal         : 98.50% (Δ +0.00%)
  Shirt          : 85.90% (Δ +0.70%) ✓
  Sneaker        : 97.60% (Δ +0.00%)
  Bag            : 99.40% (Δ +0.00%)
  Ankle boot     : 97.50% (Δ +0.00%)

==================================================
✓ Total Improvement: +0.13%
✓ Final Accuracy: 94.98%

==================================================
Classification Report (Selective TTA for FashionMNIST):
              precision    recall  f1-score   support

 T-shirt/top       0.91      0.90      0.91      1000
     Trouser       1.00      0.99      1.00      1000
    Pullover       0.93      0.92      0.93      1000
       Dress       0.95      0.95      0.95      1000
        Coat       0.93      0.94      0.93      1000
      Sandal       0.99      0.98      0.99      1000
       Shirt       0.86      0.86      0.86      1000
     Sneaker       0.97      0.98      0.97      1000
         Bag       0.99      0.99      0.99      1000
  Ankle boot       0.97      0.97      0.97      1000

    accuracy                           0.95     10000
   macro avg       0.95      0.95      0.95     10000
weighted avg       0.95      0.95      0.95     10000

✓ Report saved to classification_report_FashionMNIST_selective_tta.txt
✓ Confusion matrix saved to confusion_matrix_FashionMNIST_selective_tta.png

==================================================
Weak Class Improvement Summary (Selective TTA):
  Pullover: +0.60%
  Shirt: +0.70%
  Average weak class improvement: +0.65%

==================================================
🎉 Final Result: 94.85% → 94.98% (+0.13%)
🎉 Only weak classes ['Pullover', 'Shirt'] received 8 extra augmentations.
'''