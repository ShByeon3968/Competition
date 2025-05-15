import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import confusion_matrix

from custom_dataset import ClasswiseTorchvisionAugImageFolder  # ← 사용자 정의 클래스
from utils import rand_bbox  # ← CutMix용 함수

# 설정
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME = True
class_names = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']

# Transform 정의
transform_strong = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
transform_weak = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
transform_val = transform_weak

classwise_transforms = {
    'Andesite': transform_strong,
    'Basalt': transform_strong,
    'Etc': transform_strong,
    'default': transform_weak
}

# Dataset & Loader
train_dataset = ClasswiseTorchvisionAugImageFolder('./data/train', classwise_transforms)
val_dataset = datasets.ImageFolder('./data/val', transform=transform_val)

labels = [label for _, label in train_dataset.samples]
class_counts = np.bincount(labels)
class_weights = 1. / class_counts
sample_weights = [class_weights[l] for l in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델
model = timm.create_model('convnextv2_base.fcmae_ft_in1k', pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

# 손실함수
class_counts_tensor = torch.tensor(class_counts, dtype=torch.float)
weights = 1. / class_counts_tensor
weights = weights / weights.sum()
criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# 평가 지표
metrics = MetricCollection({
    'acc': MulticlassAccuracy(num_classes=NUM_CLASSES, average='macro')
}).to(DEVICE)

# 체크포인트
start_epoch = 0
best_val_acc = 0.0
if RESUME:
    ckpt_path = sorted(glob.glob('checkpoints/convnextv2/best_convnextv2_base_model_8901.pt'))[-1]
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    print(f"✅ Resumed from {ckpt_path}")

# 학습 루프
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        r = np.random.rand(1)
        if r < 0.5:
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(imgs.size(0)).to(DEVICE)
            target_a, target_b = labels, labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2)))
        else:
            lam = 1.0
            target_a = target_b = labels

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    train_acc = correct / len(train_loader.dataset)
    print(f"[Train] Epoch {epoch+1} Loss: {total_loss:.4f} Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    metrics.reset()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"[Val Epoch {epoch+1}]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            metrics.update(preds, labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    result = metrics.compute()
    val_acc = result['acc'].item()
    print(f"[Val Accuracy] Macro: {val_acc:.4f}")

    # 혼동행렬 저장
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Epoch {epoch+1})")
    os.makedirs("./confusion_matrices", exist_ok=True)
    plt.savefig(f"./confusion_matrices/epoch_{epoch+1}_cm.png")
    plt.close()

    # 모델 저장
    save_path = f"./checkpoints/convnextv2_base_epoch_{epoch+1}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc
    }, save_path)
    print(f"✅ Saved model to {save_path}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_path = f"./checkpoints/best_convnextv2_base_model.pt"
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc
        }, best_path)
        print(f"🏆 Best model updated: {best_path} (Val Acc: {val_acc:.4f})")
