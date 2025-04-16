import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import os
import csv
from tqdm import tqdm

# 환경 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7

# 클래스 이름
class_names = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']

# 모델 로딩
def load_model(model_name, checkpoint_path):
    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE)['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

convnext_model = load_model("convnext_large", "./checkpoints/convnext_epoch_10.pt")
efficientnet_model = load_model("tf_efficientnet_b4_ns", "./checkpoints/efficientnet_epoch_9.pt")

# 모델 가중치 (soft voting 비율)
WEIGHT_CONVNEXT = 0.6
WEIGHT_EFFICIENTNET = 0.4

# TTA용 다양한 크기 설정
tta_resize_sizes = [224, 256, 288]

# 이미지 경로
IMAGE_DIR = './data/test'

# 결과 저장 리스트
inference_results = []

# TTA 적용 전처리 함수 생성
def get_tta_transforms(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# 추론 시작
with torch.no_grad():
    for fname in tqdm(os.listdir(IMAGE_DIR), desc="Multi-Scale TTA + Ensemble Predict"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(IMAGE_DIR, fname)
            image = Image.open(path).convert("RGB")

            # 예측 누적값 초기화
            ensemble_output = torch.zeros(NUM_CLASSES, device=DEVICE)

            # 각 스케일 TTA 수행
            for size in tta_resize_sizes:
                transform = get_tta_transforms(size)
                input_tensor = transform(image).unsqueeze(0).to(DEVICE)

                # 모델별 softmax 출력
                out_conv = F.softmax(convnext_model(input_tensor), dim=1)
                out_eff = F.softmax(efficientnet_model(input_tensor), dim=1)

                # soft voting 후 누적
                combined = WEIGHT_CONVNEXT * out_conv + WEIGHT_EFFICIENTNET * out_eff
                ensemble_output += combined.squeeze(0)

            # 평균 내기
            ensemble_output /= len(tta_resize_sizes)
            pred = torch.argmax(ensemble_output).item()
            pred_name = class_names[pred] if pred < len(class_names) else str(pred)

            inference_results.append({'ID': fname.split('.')[0], 'rock_type': pred_name})
            print(f"{fname.split('.')[0]} → {pred_name}")

# CSV로 저장
output_path = './ensemble_multiscale_tta_predictions_v2.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ID', 'rock_type'])
    writer.writeheader()
    writer.writerows(inference_results)

print(f"\n✅ All predictions saved to {output_path}")
