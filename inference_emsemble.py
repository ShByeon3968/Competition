import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import csv
from utils import load_model

# 환경 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7

# 클래스 이름 리스트 (옵션: train_dataset.classes)
class_names = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']

# ConvNeXt, EfficientNet 불러오기
convnext_model = load_model("convnextv2_base.fcmae_ft_in22k_in1k_384", "checkpoints/0423/best_conv2_model_9294.pt")
efficientnet_model = load_model("convnext_base.fb_in22k_ft_in1k_384", "checkpoints/0425/best_conv_b_model_0.8918242454528809.pt")

# 모델 가중치 (총합이 1이 되도록 설정)
WEIGHT_CONVNEXT = 0.45
WEIGHT_EFFICIENTNET = 0.55

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 이미지 폴더 경로
IMAGE_DIR = './data/test'

# 결과 저장 리스트
inference_results = []

# 이미지 폴더 내 전체 추론
with torch.no_grad():
    for fname in tqdm(os.listdir(IMAGE_DIR), desc="Ensembling Predict"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(IMAGE_DIR, fname)
            image = Image.open(path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            # 각 모델로 softmax 예측
            out_conv = F.softmax(convnext_model(input_tensor), dim=1)
            out_eff = F.softmax(efficientnet_model(input_tensor), dim=1)

            # 가중 평균 (soft voting)
            ensemble_output = WEIGHT_CONVNEXT * out_conv + WEIGHT_EFFICIENTNET * out_eff
            pred = torch.argmax(ensemble_output, dim=1).item()

            pred_name = class_names[pred] if pred < len(class_names) else str(pred)

            inference_results.append({'ID': fname.split('.')[0], 'rock_type': pred_name})
            print(f"{fname.split('.')[0]} → {pred_name}")

# CSV로 저장
output_path = './ensemble_rock_predictions_0425_v4.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ID', 'rock_type'])
    writer.writeheader()
    writer.writerows(inference_results)

print(f"\n✅ All predictions saved to {output_path}")

