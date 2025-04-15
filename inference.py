import torch
import timm
from torchvision import transforms
from PIL import Image
import os
import json  # 결과 저장용
import pandas as pd

csv_file = pd.read_csv('./data/sample_submission.csv')

# ✅ 설정
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = './data/test'  # 추론할 이미지 폴더
MODEL_PATH = './checkpoints/convnext_epoch_7.pt'

# ✅ 모델 로드
model = timm.create_model('convnext_large', pretrained=False, num_classes=NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(DEVICE)

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 클래스 이름 리스트 (옵션: train_dataset.classes)
class_names = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']

# 결과 저장 리스트
inference_results = []

# 이미지 폴더 내 전체 추론
with torch.no_grad():
    for fname in os.listdir(IMAGE_DIR):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(IMAGE_DIR, fname)
            image = Image.open(path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            pred_name = class_names[pred] if pred < len(class_names) else str(pred)

            inference_results.append({'ID': fname.split('.')[0], 'rock_type': pred_name})
            print(f"{fname.split('.')[0]} → {pred_name}")

# CSV로 저장
df = pd.DataFrame(inference_results)
df.to_csv('inference_results.csv', index=False, encoding='utf-8-sig')
print("✅ inference_results.csv 저장 완료!")

