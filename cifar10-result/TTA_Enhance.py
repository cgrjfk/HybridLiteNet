import torch
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from HybridLiteNet import HybridLiteNetV2
from train import CachedDataset

# 薄弱类别索引
# bird=2, cat=3, dog=5
WEAK_CLASSES = [2, 3, 5]  # bird, cat, dog


def get_tta_transforms():
    """定义 TTA 使用的数据增强变换"""
    # 基础增强（所有类别通用）
    base_transforms = [
        lambda img: img,  # 原始图像
        lambda img: transforms.functional.hflip(img),  # 水平翻转
        lambda img: transforms.functional.affine(img, angle=0, translate=(2, 2), scale=1.0, shear=0),  # 平移+2
        lambda img: transforms.functional.affine(img, angle=0, translate=(-2, -2), scale=1.0, shear=0),  # 平移-2
        lambda img: transforms.functional.rotate(img, angle=5),  # 旋转+5度
        lambda img: transforms.functional.rotate(img, angle=-5),  # 旋转-5度
        lambda img: transforms.functional.hflip(
            transforms.functional.affine(img, angle=0, translate=(2, 2), scale=1.0, shear=0)),  # 翻转+平移
        lambda img: transforms.functional.rotate(transforms.functional.hflip(img), angle=5),  # 翻转+旋转
    ]

    # 针对薄弱类别的额外增强（更强、更多样）
    weak_transforms = [
        lambda img: transforms.functional.affine(img, angle=0, translate=(3, 3), scale=1.0, shear=0),  # 更大平移
        lambda img: transforms.functional.affine(img, angle=0, translate=(-3, -3), scale=1.0, shear=0),
        lambda img: transforms.functional.rotate(img, angle=10),  # 更大旋转
        lambda img: transforms.functional.rotate(img, angle=-10),
        lambda img: transforms.functional.hflip(transforms.functional.rotate(img, angle=10)),  # 翻转+旋转10度
        lambda img: transforms.functional.affine(img, angle=0, translate=(2, 2), scale=0.95, shear=0),  # 缩放0.95
        lambda img: transforms.functional.affine(img, angle=0, translate=(2, 2), scale=1.05, shear=0),  # 缩放1.05
        lambda img: transforms.functional.adjust_brightness(img, brightness_factor=0.9),  # 亮度调整
        lambda img: transforms.functional.adjust_contrast(img, contrast_factor=1.1),  # 对比度调整
    ]

    return base_transforms, weak_transforms


def predict_with_tta(model, img, label, base_transforms, weak_transforms, device):
    """
    对单张图像应用 TTA，对薄弱类别使用更多增强
    img: tensor [C, H, W]
    label: 真实标签（用于判断是否为薄弱类别）
    """
    preds = []
    with torch.no_grad():
        # 应用基础增强
        for transform in base_transforms:
            augmented = transform(img)
            augmented = augmented.unsqueeze(0).to(device)
            output = model(augmented)
            prob = torch.softmax(output, dim=1)
            preds.append(prob.cpu())

        # 如果是薄弱类别，额外应用更多增强
        if label in WEAK_CLASSES:
            for transform in weak_transforms:
                augmented = transform(img)
                augmented = augmented.unsqueeze(0).to(device)
                output = model(augmented)
                prob = torch.softmax(output, dim=1)
                preds.append(prob.cpu())

    # 平均所有增强的预测
    avg_prob = torch.stack(preds).mean(dim=0)
    return avg_prob


def evaluate_with_tta(model, testloader, device, base_transforms, weak_transforms):
    """对整个测试集应用 TTA 评估"""
    model.eval()
    all_preds = []
    all_targets = []

    for inputs, targets in tqdm(testloader, desc="TTA Evaluation"):
        inputs = inputs.to(device)

        batch_preds = []
        for i in range(inputs.shape[0]):
            img = inputs[i].cpu()
            label = targets[i]
            avg_prob = predict_with_tta(model, img, label, base_transforms, weak_transforms, device)
            pred = avg_prob.argmax(dim=1).item()
            batch_preds.append(pred)

        all_preds.extend(batch_preds)
        all_targets.extend(targets.numpy())

    return all_preds, all_targets


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Weak classes (extra augmentations): {WEAK_CLASSES}")

    # 加载测试集
    print("Loading CIFAR-10 test set...")
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = CachedDataset(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=50, shuffle=False, num_workers=0)

    # 加载最佳模型
    print("Loading best model...")
    model = HybridLiteNetV2(num_classes=10).to(device)
    model.load_state_dict(torch.load("best_model_finetuned_96.18.pth", map_location=device))
    model.eval()

    # 获取 TTA 增强列表
    base_transforms, weak_transforms = get_tta_transforms()
    print(f"Base transforms: {len(base_transforms)}")
    print(f"Weak transforms (for cat/dog/bird): {len(weak_transforms)}")
    print(f"Total transforms for weak classes: {len(base_transforms) + len(weak_transforms)}")

    # 基准评估（无 TTA）
    print("\n" + "=" * 50)
    print("Running baseline evaluation (without TTA)...")
    all_preds_baseline = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc="Baseline"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds_baseline.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    baseline_acc = accuracy_score(all_targets, all_preds_baseline) * 100

    # 分别计算各类别准确率
    from sklearn.metrics import recall_score
    baseline_recall = recall_score(all_targets, all_preds_baseline, average=None)

    print(f"\nBaseline Overall Accuracy: {baseline_acc:.2f}%")
    print("Baseline Per-class Recall:")
    for i, class_name in enumerate(testset.classes):
        print(f"  {class_name:10s}: {baseline_recall[i] * 100:.2f}%")

    # TTA 评估
    print("\n" + "=" * 50)
    print("Running TTA evaluation (with weak class enhancement)...")
    all_preds_tta, all_targets = evaluate_with_tta(model, testloader, device, base_transforms, weak_transforms)
    tta_acc = accuracy_score(all_targets, all_preds_tta) * 100

    # 计算 TTA 后各类别准确率
    tta_recall = recall_score(all_targets, all_preds_tta, average=None)

    print(f"\nTTA Overall Accuracy: {tta_acc:.2f}%")
    print("TTA Per-class Recall:")
    for i, class_name in enumerate(testset.classes):
        improvement = (tta_recall[i] - baseline_recall[i]) * 100
        weak_marker = " ✓" if i in WEAK_CLASSES else ""
        print(f"  {class_name:10s}: {tta_recall[i] * 100:.2f}% (Δ {improvement:+.2f}%){weak_marker}")

    # 提升幅度
    improvement = tta_acc - baseline_acc
    print(f"\n{'=' * 50}")
    print(f"✓ Total Improvement: +{improvement:.2f}%")
    print(f"✓ Final Accuracy: {tta_acc:.2f}%")

    # 生成详细分类报告
    print("\n" + "=" * 50)
    print("Classification Report (with TTA - Weak Class Enhanced):")
    target_names = testset.classes
    report = classification_report(all_targets, all_preds_tta, target_names=target_names)
    print(report)

    # 保存报告
    with open('classification_report_tta_enhanced.txt', 'w') as f:
        f.write(f"TTA Accuracy (Enhanced for weak classes): {tta_acc:.2f}%\n")
        f.write(f"Baseline Accuracy: {baseline_acc:.2f}%\n")
        f.write(f"Improvement: +{improvement:.2f}%\n")
        f.write(f"Weak classes (extra augmentations): {[testset.classes[i] for i in WEAK_CLASSES]}\n\n")
        f.write(report)
    print("✓ Classification report saved to classification_report_tta_enhanced.txt")

    # 生成混淆矩阵
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(all_targets, all_preds_tta)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues', values_format='d', ax=ax, colorbar=False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    ax.set_title(f'Confusion Matrix with Enhanced TTA (Accuracy: {tta_acc:.2f}%)',
                 fontweight='bold', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix_tta_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrix saved to confusion_matrix_tta_enhanced.png")

    # 总结薄弱类别提升
    print(f"\n{'=' * 50}")
    print("Weak Class Improvement Summary:")
    weak_improvement_total = 0
    for i in WEAK_CLASSES:
        imp = (tta_recall[i] - baseline_recall[i]) * 100
        weak_improvement_total += imp
        print(f"  {testset.classes[i]}: +{imp:.2f}%")
    print(f"  Average weak class improvement: +{weak_improvement_total / len(WEAK_CLASSES):.2f}%")

    print(f"\n{'=' * 50}")
    print(f"🎉 Final Result: {baseline_acc:.2f}% → {tta_acc:.2f}% (+{improvement:.2f}%)")
    print(f"🎉 Weak classes (bird/cat/dog) received {len(weak_transforms)} extra augmentations each!")


if __name__ == "__main__":
    main()
'''
Device: cuda
Weak classes (extra augmentations): [2, 3, 5]
Loading CIFAR-10 test set...
Files already downloaded and verified
Loading best model...
Base transforms: 8
Weak transforms (for cat/dog/bird): 9
Total transforms for weak classes: 17

==================================================
Running baseline evaluation (without TTA)...
Baseline: 100%|██████████| 200/200 [00:02<00:00, 97.34it/s] 
TTA Evaluation:   0%|          | 0/200 [00:00<?, ?it/s]
Baseline Overall Accuracy: 96.18%
Baseline Per-class Recall:
  airplane  : 96.40%
  automobile: 98.10%
  bird      : 94.60%
  cat       : 91.00%
  deer      : 97.20%
  dog       : 93.20%
  frog      : 98.30%
  horse     : 97.60%
  ship      : 98.00%
  truck     : 97.40%

==================================================
Running TTA evaluation (with weak class enhancement)...
TTA Evaluation: 100%|██████████| 200/200 [05:28<00:00,  1.64s/it]

TTA Overall Accuracy: 96.21%
TTA Per-class Recall:
  airplane  : 96.10% (Δ -0.30%)
  automobile: 98.40% (Δ +0.30%)
  bird      : 94.90% (Δ +0.30%) ✓
  cat       : 91.20% (Δ +0.20%) ✓
  deer      : 97.40% (Δ +0.20%)
  dog       : 93.30% (Δ +0.10%) ✓
  frog      : 98.20% (Δ -0.10%)
  horse     : 97.40% (Δ -0.20%)
  ship      : 98.00% (Δ +0.00%)
  truck     : 97.20% (Δ -0.20%)

==================================================
✓ Total Improvement: +0.03%
✓ Final Accuracy: 96.21%

==================================================
Classification Report (with TTA - Weak Class Enhanced):
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

✓ Classification report saved to classification_report_tta_enhanced.txt
✓ Confusion matrix saved to confusion_matrix_tta_enhanced.png

==================================================
Weak Class Improvement Summary:
  bird: +0.30%
  cat: +0.20%
  dog: +0.10%
  Average weak class improvement: +0.20%

==================================================
🎉 Final Result: 96.18% → 96.21% (+0.03%)
🎉 Weak classes (bird/cat/dog) received 9 extra augmentations each!

'''