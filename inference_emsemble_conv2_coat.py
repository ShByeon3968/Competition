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

# 클래스 이름 리스트
class_names = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']

# 모델 로딩 함수
def load_model(model_name, checkpoint_path):
    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE)['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

# ✅ 앙상블 모델 불러오기
convnextv2_model = load_model("convnextv2_base.fcmae_ft_in1k", "./checkpoints/convnextv2_epoch_9.pt")
coatnet_model = load_model("coatnet_2_rw_224.sw_in12k_ft_in1k", "./checkpoints/coatnet_epoch_10.pt")

# ✅ 모델별 soft voting 가중치
WEIGHT_CONVNEXTV2 = 0.6
WEIGHT_COATNET = 0.4

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 이미지 폴더
IMAGE_DIR = './data/test'
inference_results = []

# 추론 루프
with torch.no_grad():
    for fname in tqdm(os.listdir(IMAGE_DIR), desc="ConvNeXtV2 + CoAtNet Ensemble Predict"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(IMAGE_DIR, fname)
            image = Image.open(path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            out_convnextv2 = F.softmax(convnextv2_model(input_tensor), dim=1)
            out_coatnet = F.softmax(coatnet_model(input_tensor), dim=1)

            # 앙상블 (soft voting)
            ensemble_output = WEIGHT_CONVNEXTV2 * out_convnextv2 + WEIGHT_COATNET * out_coatnet
            pred = torch.argmax(ensemble_output, dim=1).item()
            pred_name = class_names[pred] if pred < len(class_names) else str(pred)

            inference_results.append({'ID': fname.split('.')[0], 'rock_type': pred_name})
            print(f"{fname.split('.')[0]} → {pred_name}")

# 결과 저장
output_path = './ensemble_rock_predictions_convnextv2_coatnet.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['ID', 'rock_type'])
    writer.writeheader()
    writer.writerows(inference_results)

print(f"\n✅ All predictions saved to {output_path}")
