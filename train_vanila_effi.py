# train with cutmix
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup

# 하이퍼파라미터
class_names = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']
NUM_CLASSES = 7
BATCH_SIZE = 16
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME = True

# 데이터 전처리
transform_train = transforms.Compose([
    transforms.RandomResizedCrop((384, 384), scale=(0.7, 1.0), ratio=(0.9, 1.1)),  # 크롭 추가
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.RandomResizedCrop((384, 384), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mixup_fn = Mixup(
    mixup_alpha=0.0,      # mixup 사용 안함
    cutmix_alpha=1.0,     # cutmix만 사용 (1.0 권장, 실험적으로 0.5~1.0 사이 사용)
    prob=0.7,             # 항상 적용
    switch_prob=0.0,      # switch prob 0 (CutMix only)
    mode='batch',
    label_smoothing=0.1,
    num_classes=7
)

# 평가 지표
metrics = MetricCollection({
    'acc': MulticlassAccuracy(num_classes=NUM_CLASSES, average='macro')
}).to(DEVICE)

# 데이터 로딩
train_dataset = datasets.ImageFolder(root='./data/train', transform=transform_train)
val_dataset = datasets.ImageFolder(root='./data/val', transform=transform_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ CoAtNet-2 모델 정의
model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=NUM_CLASSES)
# convnextv2_base.fcmae_ft_in22k_in1k_384
model.to(DEVICE)

# 손실함수, 옵티마이저
criterion = SoftTargetCrossEntropy()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# 체크포인트 불러오기
start_epoch = 0
if RESUME:
    checkpoint_files = sorted(glob.glob('checkpoints/0429/best_effi_cut_model_0.8920.pt'))
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ Resumed from checkpoint: {latest_ckpt} (Epoch {start_epoch})")
    else:
        print("⚠️ No checkpoint found. Starting from scratch.")

# 최고 성능 초기화
best_val_acc = 0.0

# 학습 루프
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss, correct = 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, labels = mixup_fn(imgs, labels)  # CutMix 적용

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 정확도 계산 (soft label → hard label 변환)
        if labels.ndim == 2:
            labels_for_acc = labels.argmax(dim=1)
        else:
            labels_for_acc = labels
        correct += (outputs.argmax(1) == labels_for_acc).sum().item()

        # ✅ 실시간 loss 출력
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{correct / ((pbar.n + 1) * BATCH_SIZE):.4f}"
        })

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

    # ✅ Confusion Matrix 저장
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    os.makedirs("./confusion_matrices", exist_ok=True)
    cm_path = f"./confusion_matrices/confusion_matrix_epoch_effi_cut_{epoch+1}.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"📊 Confusion matrix saved: {cm_path}")

    # ✅ best 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_path = f"./checkpoints/best_effi_cut_model_{val_acc:.4f}.pt"
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc
        }, best_path)
        print(f"🏆 Best model updated (Epoch {epoch+1}, Val Acc: {val_acc:.4f}) → saved to {best_path}")
