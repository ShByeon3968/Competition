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
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME = False  # 이어서 학습하려면 True

# 데이터 전처리 (ImageNet 기준)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

# ✅ EfficientNet 모델 정의
model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)

# 손실함수, 옵티마이저, 스케쥴러러
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 체크포인트 불러오기
start_epoch = 0
if RESUME:
    checkpoint_files = sorted(glob.glob('./checkpoints/efficientnet_epoch_*.pt'))
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ Resumed from checkpoint: {latest_ckpt} (Epoch {start_epoch})")
    else:
        print("⚠️ No checkpoint found. Starting from scratch.")

# 학습 루프
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

    # ✅ scheduler step 호출
    scheduler.step()
    # 검증
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            metrics.update(preds, labels)

    result = metrics.compute()
    print(f"[Val Accuracy] Macro: {result['acc']:.4f}")

    # 모델 저장
    save_path = f"./checkpoints/efficientnet_epoch_{epoch+1}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': result['acc'].item()
    }, save_path)
    print(f"✅ Model saved to {save_path}")
