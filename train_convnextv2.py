import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
import glob
from torch.optim.lr_scheduler import CosineAnnealingLR

# 하이퍼파라미터
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME = True  # ← 체크포인트에서 이어서 학습

# 데이터 전처리
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),      
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변화                   
    transforms.RandomRotation(30),  # 회전
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 평가 지표
metrics = MetricCollection({
    'acc': MulticlassAccuracy(num_classes=NUM_CLASSES, average='macro')
}).to(DEVICE)

# 데이터 로딩
train_dataset = datasets.ImageFolder(root='./data/train', transform=transform_train)
val_dataset = datasets.ImageFolder(root='./data/val', transform=transform_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 정의
model = timm.create_model('convnextv2_base.fcmae_ft_in1k', pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)

# 손실함수, 옵티마이저
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

best_val_acc = 0.0  # 🔧 초기화

# 체크포인트 불러오기
start_epoch = 0
if RESUME:
    checkpoint_files = sorted(glob.glob('./checkpoints/best_convnextv2_model.pt'))
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ Resumed from checkpoint: {latest_ckpt} (Epoch {start_epoch})")
    else:
        print("⚠️ No checkpoint found. Starting from scratch.")
# ---------------------학습 루프---------------------
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss, correct = 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_loader.dataset)
    print(f"[Train] Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}")

    # ---------------------Validation---------------------
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            metrics.update(preds, labels)

    result = metrics.compute()
    val_acc = result['acc'].item()
    print(f"[Val Accuracy] Macro: {val_acc:.4f}")

    # 모델 저장
    save_path = f"./checkpoints/convnextv2_epoch_{epoch+1}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': result['acc'].item()
    }, save_path)
    print(f"✅ Model saved to {save_path}")

    # ---------------------best 모델 저장---------------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_path = f"./checkpoints/best_convnextv2_model.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc
        }, best_path)
        print(f"🏆 Best model updated and saved to {best_path} (Val Acc: {val_acc:.4f})")
