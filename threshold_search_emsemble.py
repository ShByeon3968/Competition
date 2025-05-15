import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from utils import load_model
# ✅ val_loader 재사용 또는 재정의
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_dataset = ImageFolder('./data/val', transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']
NUM_CLASSES = len(class_names)

# ConvNeXt, EfficientNet 불러오기
convnext = load_model("convnextv2_base.fcmae_ft_in22k_in1k_384", "checkpoints/0423/best_conv2_model_9294.pt")
effnet = load_model("convnext_base.fb_in22k_ft_in1k_384", "checkpoints/0423/best_conv_b_model.pt")

# 앙상블 추론 결과 수집
y_true, y_probs, y_preds = [], [], []

convnext.eval(), effnet.eval()
with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Validating with Ensemble"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out1 = torch.softmax(convnext(imgs), dim=1)
        out2 = torch.softmax(effnet(imgs), dim=1)
        probs = (0.6 * out1 + 0.4 * out2).cpu().numpy()

        y_true.extend(labels.cpu().numpy())
        y_probs.extend(probs)
        y_preds.extend(np.argmax(probs, axis=1))

y_true = np.array(y_true)
y_probs = np.array(y_probs)
y_preds = np.array(y_preds)

# ✅ 클래스별 threshold 및 fallback 대상 탐색
threshold_range = np.arange(0.3, 0.91, 0.01)
best_thresholds = np.zeros(NUM_CLASSES)
best_fallbacks = np.zeros(NUM_CLASSES, dtype=int)

for cls in range(NUM_CLASSES):
    best_f1 = 0
    best_thresh = 0.5
    best_fallback = cls  # fallback 없이 자기 자신

    for t in threshold_range:
        for fallback in range(NUM_CLASSES):
            if fallback == cls:
                continue  # 자기 자신 fallback은 무의미

            adjusted_preds = []
            for i in range(len(y_probs)):
                pred = np.argmax(y_probs[i])
                max_prob = np.max(y_probs[i])

                if pred == cls and max_prob < t:
                    pred = fallback  # fallback 적용

                adjusted_preds.append(pred)

            f1 = f1_score(y_true, adjusted_preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                best_fallback = fallback

    best_thresholds[cls] = best_thresh
    best_fallbacks[cls] = best_fallback
    print(f"Class {class_names[cls]:<16} → Threshold: {best_thresh:.2f} | Fallback: {class_names[best_fallback]:<16} | Macro F1: {best_f1:.4f}")
