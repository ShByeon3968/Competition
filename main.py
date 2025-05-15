import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
import os
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import AlbumentationsDataset
from utils import load_config

# config.yaml 로드
config = load_config()
# 하이퍼파라미터
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Albumentations 증강 정의
albumentations_train = A.Compose([
    A.RandomResizedCrop((384, 384), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=15, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
albumentations_val = A.Compose([
    A.Resize(400, 400),                
    A.CenterCrop(384, 384),            
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


train_dataset = AlbumentationsDataset(config['train_path'], albumentations_train)
val_dataset = AlbumentationsDataset(config['valid_path'], albumentations_val)

train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

# 클래스 가중치 (Etc 강조)
# 클래스별 샘플 수 → [31904, 21448, 12748, 59132, 74339, 71574, 29736]
class_counts = torch.tensor(config['class_samples'], dtype=torch.float32)
class_weights = 1.0 / class_counts
class_weights[2] *= 3.0  # Etc 클래스 가중치 강화
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(DEVICE)

# 평가 지표
metrics = MetricCollection({
    'acc': MulticlassAccuracy(num_classes=config['NUM_CLASSES'], average='macro')
}).to(DEVICE)

# 모델 정의
model = timm.create_model(config['tf_efficientnet_b7'], pretrained=True, num_classes=config['NUM_CLASSES'])
model.to(DEVICE)

# 손실함수, 옵티마이저
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# 체크포인트 불러오기
start_epoch = 0
best_val_acc = 0.0
if config['RESUME']:
    ckpt_path = sorted(glob.glob('checkpoints/best_conv2_cut_train_2_model_*.pt'))
    if ckpt_path:
        checkpoint = torch.load(ckpt_path[-1], map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ Resumed from checkpoint: {ckpt_path[-1]} (Epoch {start_epoch})")
    else:
        print("⚠️ No checkpoint found. Starting from scratch.")

# 학습 루프
for epoch in range(start_epoch, config['EPOCHS']):
    model.train()
    total_loss, correct = 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct / ((pbar.n + 1) * config['BATCH_SIZE']):.4f}"})

    avg_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_loader.dataset)
    print(f"\n[Train] Epoch {epoch+1}, Loss: {avg_loss:.6f}, Acc: {train_acc:.4f}")

    # 검증
    model.eval()
    metrics.reset()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)

            metrics.update(preds, labels)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    result = metrics.compute()
    val_acc = result['acc'].item()
    print(f"[Val Accuracy] Macro: {val_acc:.4f}")

    # Confusion Matrix 저장
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config['class_names'])
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    os.makedirs("./confusion_matrices", exist_ok=True)
    cm_path = f"./confusion_matrices/confusion_matrix_epoch_conv2_clean_{epoch+1}.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # Best Model 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = f"./checkpoints/best_conv2_clean_model_{val_acc:.4f}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc
        }, save_path)
        print(f"🏆 Best model saved to {save_path}")
