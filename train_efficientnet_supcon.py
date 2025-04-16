import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from loss import SupConLoss
from model import ContrastiveClassifier
from sklearn.metrics import f1_score, classification_report
from torchvision.datasets import ImageFolder
from glob import glob

# =====================
# 학습 파라미터 설정
# =====================
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 20
ALPHA = 0.3  # SupConLoss 비중
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESUME = False

# 데이터 전처리 (ImageNet 기준)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================
# 데이터셋 로딩
# =====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder('./data/train', transform=train_transform)
val_dataset = ImageFolder('./data/val', transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# 모델, 손실함수, 옵티마이저
# =====================
model = ContrastiveClassifier(num_classes=NUM_CLASSES).to(DEVICE)
ce_loss_fn = nn.CrossEntropyLoss()
con_loss_fn = SupConLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 🔁 Resume 체크포인트 로딩
start_epoch = 0
best_f1 = 0.0
os.makedirs('./supcon_checkpoints', exist_ok=True)

if RESUME:
    checkpoint_files = sorted(glob('./supcon_checkpoints/epoch_*.pt'))
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        checkpoint_epoch = int(latest_ckpt.split('_')[-1].split('.')[0])
        model.load_state_dict(torch.load(latest_ckpt, map_location=DEVICE))
        start_epoch = checkpoint_epoch
        print(f"✅ Resumed from checkpoint: {latest_ckpt} (Epoch {start_epoch})")

# =====================
# 학습 루프
# =====================
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_ce_loss, total_con_loss = 0.0, 0.0
    correct, total = 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        feat, proj, logits = model(imgs)

        ce_loss = ce_loss_fn(logits, labels)
        con_loss = con_loss_fn(proj, labels)
        loss = ce_loss + ALPHA * con_loss

        loss.backward()
        optimizer.step()

        total_ce_loss += ce_loss.item()
        total_con_loss += con_loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"[Train Epoch {epoch+1}] CE Loss: {total_ce_loss:.4f} | SupCon Loss: {total_con_loss:.4f} | Acc: {acc:.4f}")

    # ============== Validation ==============
    model.eval()
    val_correct, val_total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            _, _, logits = model(imgs)
            preds = logits.argmax(1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"[Val Epoch {epoch+1}] Acc: {val_acc:.4f} | Macro F1: {val_f1:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))

    # 🔒 모델 저장
    torch.save(model.state_dict(), f'./supcon_checkpoints/epoch_{epoch+1}.pt')
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), './supcon_checkpoints/best_model.pt')
        print(f"✅ Best model updated at Epoch {epoch+1} with Macro F1: {best_f1:.4f}")