import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from HybridLiteNet import HybridLiteNet
from tqdm import tqdm
from multiprocessing import freeze_support
import math


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # image enhance
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    # CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # model、loss function、optimizer
    model = HybridLiteNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)  # 0.05

    # warmup + cosine
    total_epochs = 350
    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos((epoch - warmup_epochs) /
                                       (total_epochs - warmup_epochs) * math.pi))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # record
    best_acc = 0.0
    train_acc_list, test_acc_list = [], []
    for epoch in range(total_epochs):

        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, targets in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{total_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_acc = 100. * correct / total
        train_acc_list.append(train_acc)

        # eval
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_acc = 100. * correct / total
        test_acc_list.append(test_acc)
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{total_epochs}] "
              f"Train Loss: {train_loss / len(trainloader):.4f} | Train Acc: {train_acc:.2f}% || "
              f"Test Loss: {test_loss / len(testloader):.4f} | Test Acc: {test_acc:.2f}%")
        # best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

        # draw acc curve
    plt.figure()
    plt.plot(range(1, total_epochs + 1), train_acc_list, label='Train Acc')
    plt.plot(range(1, total_epochs + 1), test_acc_list, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train/Test Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png', dpi=300)
    plt.close()

    # Matrix
    print("Generating confusion matrix for best model...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu())
            all_targets.append(targets.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=testset.classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()

    print(f"Training complete. Best Test Accuracy: {best_acc:.2f}%")
    print("Accuracy curve saved as accuracy_curve.png")
    print("Confusion matrix saved as confusion_matrix.png")
    print("Best model saved as best_model.pth")


if __name__ == "__main__":
    freeze_support()
    main()
#  Train Loss: 0.8138 | Train Acc: 86.35% || Test Loss: 0.7124 | Test Acc: 92.01%
"""
Epoch [350/350] Train Loss: 0.7386 | Train Acc: 89.68% || Test Loss: 0.6776 | Test Acc: 93.06%
Generating confusion matrix for best model...
Training complete. Best Test Accuracy: 93.11%
Accuracy curve saved as accuracy_curve.png
Confusion matrix saved as confusion_matrix.png
Best model saved as best_model.pth
"""