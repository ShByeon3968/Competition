import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from utils import load_model
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# 환경 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7

# 클래스 이름 리스트 (옵션: train_dataset.classes)
class_names = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']

# ConvNeXt, EfficientNet 불러오기
convnext_model = load_model("convnextv2_base.fcmae_ft_in22k_in1k_384", "checkpoints/0423/best_conv2_model_8966.pt")
efficientnet_model = load_model("tf_efficientnetv2_s.in21k", "checkpoints/0423/best_efficient_s_model_9069.pt")

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_dataset = ImageFolder('./data/val', transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 1. Validation 데이터에서 softmax 결과 저장 (여기서는 두 모델 모두 미리 불러온 상태)
convnext_model.eval()
efficientnet_model.eval()
probs1_list, probs2_list, y_true_list = [], [], []

with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Collect ensemble validation logits"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        probs1 = torch.softmax(convnext_model(imgs), dim=1)
        probs2 = torch.softmax(efficientnet_model(imgs), dim=1)
        probs1_list.append(probs1.cpu().numpy())
        probs2_list.append(probs2.cpu().numpy())
        y_true_list.append(labels.cpu().numpy())

probs1 = np.concatenate(probs1_list, axis=0)  # (N, C)
probs2 = np.concatenate(probs2_list, axis=0)
y_true = np.concatenate(y_true_list, axis=0)  # (N,)

# 2. Grid Search (0.0 ~ 1.0 step 0.01)
best_f1, best_w = 0, 0.5
results = []

for w in np.arange(0.0, 1.01, 0.01):
    ens_probs = w * probs1 + (1 - w) * probs2
    preds = np.argmax(ens_probs, axis=1)
    f1 = f1_score(y_true, preds, average='macro')
    results.append((w, f1))
    if f1 > best_f1:
        best_f1 = f1
        best_w = w

print(f"\n🏆 Best Ensemble Weight: ConvNeXt={best_w:.2f}, EfficientNet={1-best_w:.2f}")
print(f"    Validation Macro F1: {best_f1:.4f}")

# (선택) 가중치별 F1 변화 시각화
import matplotlib.pyplot as plt
ws, f1s = zip(*results)
plt.plot(ws, f1s, marker='o')
plt.xlabel('ConvNeXt Weight')
plt.ylabel('Macro F1')
plt.title('Grid Search for Ensemble Weight')
plt.grid()
plt.show()
