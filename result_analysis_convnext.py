import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

# ✅ 경로 및 클래스 정보
CHECKPOINT_PATH = 'convnext_epoch_10.pt'
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']

# ✅ 모델 정의
import timm
model = timm.create_model('convnext_large', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)['model_state_dict'])
model.to(DEVICE)
model.eval()

# ✅ val_loader 재사용 또는 재정의
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_dataset = ImageFolder('./data/val', transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ✅ 추론 및 예측값 수집
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in tqdm(val_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# ✅ Confusion Matrix 저장
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
cm_path = "efficientnet_confusion_matrix.png"
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

# ✅ Classification Report 저장
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

# ✅ Confidence Histogram 저장
correct_confs = [p[l] for p, l, pred in zip(all_probs, all_labels, all_preds) if l == pred]
wrong_confs = [p[pred] for p, l, pred in zip(all_probs, all_labels, all_preds) if l != pred]

plt.hist(correct_confs, bins=20, alpha=0.7, label='Correct', color='green')
plt.hist(wrong_confs, bins=20, alpha=0.7, label='Wrong', color='red')
plt.xlabel('Predicted Class Confidence')
plt.ylabel('Number of Samples')
plt.title('Prediction Confidence Distribution')
plt.legend()
hist_path = "efficientnet_confidence_hist.png"
plt.tight_layout()
plt.savefig(hist_path)
plt.close()

# ✅ PDF 리포트 생성
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=14)
pdf.cell(200, 10, txt="EfficientNet Inference Report", ln=True, align='C')

# Confusion matrix
pdf.ln(10)
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Confusion Matrix", ln=True)
pdf.image(cm_path, w=180)

# Classification report
pdf.ln(5)
pdf.set_font("Courier", size=10)
for line in report.splitlines():
    pdf.cell(200, 6, txt=line, ln=True)

# Histogram
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Prediction Confidence Histogram", ln=True)
pdf.image(hist_path, w=180)

# 저장
output_path = "convnext_inference_report.pdf"
pdf.output(output_path)
print(f"✅ PDF 리포트 저장 완료: {output_path}")
