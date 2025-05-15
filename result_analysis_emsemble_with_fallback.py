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
ensemble_model_configs = [
    # (모델명, 체크포인트 경로, soft voting 가중치)
    ('convnextv2_base.fcmae_ft_in1k', 'checkpoints/0423/best_conv2_model_9294.pt', 0.6),
    ('convnext_base.fb_in22k_ft_in1k_384', 'checkpoints/0423/best_conv_b_model.pt', 0.4),
    # 필요시 더 추가
]
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']

# ✅ 모델 로딩 함수
import timm
def load_model(model_name, checkpoint_path, num_classes):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE)['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

# ✅ 모델 앙상블 준비
ensemble_models = []
ensemble_weights = []
for model_name, ckpt, weight in ensemble_model_configs:
    model = load_model(model_name, ckpt, NUM_CLASSES)
    ensemble_models.append(model)
    ensemble_weights.append(weight)
ensemble_weights = np.array(ensemble_weights)
ensemble_weights = ensemble_weights / ensemble_weights.sum()  # 정규화

# ✅ val_loader 재사용 또는 재정의
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

transform_val = transforms.Compose([
    transforms.Resize((384, 384)),  # 모든 모델 입력 크기에 맞춰 resize (최대 모델에 맞춰서 통일 추천)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_dataset = ImageFolder('./data/val', transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ✅ Fallback 전략 세팅 (예시, 꼭 validation에서 조정/최적화)
fallback_map = {
    0: 5,  # Andesite → Mud_Sandstone
    1: 1,  # Basalt   → 없음 (자기 자신 유지)
    2: 3,  # Etc      → Gneiss
    3: 5,  # Gneiss   → Mud_Sandstone
    4: 3,  # Granite  → Gneiss
    5: 3,  # Mud_Sandstone → Gneiss
    6: 3   # Weathered_Rock → Gneiss
}
best_thresholds = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]  # 예시 threshold

# ✅ 추론 및 예측값 수집 (Soft Voting + Fallback)
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in tqdm(val_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        model_probs = []
        for model in ensemble_models:
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            model_probs.append(probs.cpu().numpy())
        model_probs = np.stack(model_probs, axis=0)
        weighted_probs = np.average(model_probs, axis=0, weights=ensemble_weights)

        batch_preds = []
        for i, probs in enumerate(weighted_probs):
            pred = np.argmax(probs)
            max_prob = probs[pred]
            # Fallback 적용
            if max_prob < best_thresholds[pred]:
                pred = fallback_map[pred]
            batch_preds.append(pred)
        all_preds.extend(batch_preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(weighted_probs)

# confusion matrix, classification_report, 히스토그램, PDF 
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (with Fallback)")
cm_path = "ensemble_confusion_matrix_with_fallback.png"
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

correct_confs = [p[l] for p, l, pred in zip(all_probs, all_labels, all_preds) if l == pred]
wrong_confs = [p[pred] for p, l, pred in zip(all_probs, all_labels, all_preds) if l != pred]

plt.hist(correct_confs, bins=20, alpha=0.7, label='Correct', color='green')
plt.hist(wrong_confs, bins=20, alpha=0.7, label='Wrong', color='red')
plt.xlabel('Predicted Class Confidence')
plt.ylabel('Number of Samples')
plt.title('Prediction Confidence Distribution (with Fallback)')
plt.legend()
hist_path = "ensemble_confidence_hist_with_fallback.png"
plt.tight_layout()
plt.savefig(hist_path)
plt.close()

# PDF 리포트
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=14)
pdf.cell(200, 10, txt="Ensemble Inference Report (with Fallback)", ln=True, align='C')

pdf.ln(10)
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Confusion Matrix (with Fallback)", ln=True)
pdf.image(cm_path, w=180)

pdf.ln(5)
pdf.set_font("Courier", size=10)
for line in report.splitlines():
    pdf.cell(200, 6, txt=line, ln=True)

pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Prediction Confidence Histogram (with Fallback)", ln=True)
pdf.image(hist_path, w=180)

output_path = "Ensemble_inference_report_with_fallback.pdf"
pdf.output(output_path)
print(f"✅ PDF 리포트 저장 완료: {output_path}")