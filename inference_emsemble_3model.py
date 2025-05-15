import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import os
import csv
from tqdm import tqdm
import numpy as np

# 환경 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7
class_names = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']
IMAGE_DIR = './data/test'

# 모델 로딩 함수
def load_model(model_name, checkpoint_path, num_classes):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE)['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

# 모델 목록: (모델명, 체크포인트경로, soft voting 가중치)
ensemble_model_configs = [
    ('convnextv2_base.fcmae_ft_in22k_in1k_384', "checkpoints/0512/best_conv2_cut_model_0.8845.pt", 0.4),
    ('maxvit_small_tf_384.in1k', 'checkpoints/0507/best_mvit_cut_model_0.8997.pt', 0.3),
    ('tf_efficientnet_b4_ns', 'checkpoints/0507/best_effi_cut_model_0.9094.pt', 0.3)
]

# 모델 로드 및 가중치 정규화
ensemble_models, ensemble_weights = [], []
for model_name, ckpt, weight in ensemble_model_configs:
    model = load_model(model_name, ckpt, NUM_CLASSES)
    ensemble_models.append(model)
    ensemble_weights.append(weight)
ensemble_weights = np.array(ensemble_weights)
ensemble_weights = ensemble_weights / ensemble_weights.sum()  # 정규화

# 384x384 전처리
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 추론 및 결과 저장
inference_results = []

with torch.no_grad():
    for fname in tqdm(os.listdir(IMAGE_DIR), desc="3-Model Ensemble Predict 384x384"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(IMAGE_DIR, fname)
            image = Image.open(path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            model_probs = []
            for model in ensemble_models:
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                model_probs.append(probs.cpu().numpy())
            model_probs = np.stack(model_probs, axis=0)  # (num_models, batch, num_classes)
            weighted_probs = np.average(model_probs, axis=0, weights=ensemble_weights)  # (batch, num_classes)
            pred = np.argmax(weighted_probs, axis=1)[0]
            pred_name = class_names[pred] if pred < len(class_names) else str(pred)

            inference_results.append({'ID': fname.split('.')[0], 'rock_type': pred_name})
            print(f"{fname.split('.')[0]} → {pred_name}")

# CSV로 저장
output_path = './ensemble_3model_rock_predictions.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ID', 'rock_type'])
    writer.writeheader()
    writer.writerows(inference_results)

print(f"\n✅ All predictions saved to {output_path}")
