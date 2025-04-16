import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import deprocess_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from tqdm import tqdm

# 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7
BATCH_SIZE = 32
CHECKPOINT_PATH = './checkpoints/convnext_epoch_7.pt'

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 데이터셋 불러오기
val_dataset = ImageFolder('./data/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = val_dataset.classes

# 모델 불러오기
model = timm.create_model('convnext_large', pretrained=False, num_classes=NUM_CLASSES)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(DEVICE)


# Grad-CAM 타겟 레이어 지정
target_layers = [model.stages[-1].blocks[-1].conv_dw]  # ConvNeXt에서 마지막 Block 사용

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

# 결과 수집
y_true, y_pred = [], []
misclassified = []

with torch.no_grad():
    for imgs, labels in tqdm(val_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        for i in range(len(preds)):
            if preds[i] != labels[i]:
                misclassified.append({
                    'image_tensor': imgs[i].cpu(),
                    'true_label': labels[i].item(),
                    'pred_label': preds[i].item()
                })

# 1. Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# 2. 오예측 이미지 + Grad-CAM 시각화
def show_misclassified_with_gradcam(misclassified, class_names, max_images=5):
    plt.figure(figsize=(15, 5))
    count = 0
    for sample in misclassified[:max_images]:
        image_tensor = sample['image_tensor']
        true_label = sample['true_label']
        pred_label = sample['pred_label']

        # De-normalize
        img = image_tensor.clone()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        img = img.permute(1, 2, 0).numpy().clip(0, 1)

        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        targets = [ClassifierOutputTarget(pred_label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        cam_result = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        plt.subplot(1, max_images, count+1)
        plt.imshow(cam_result)
        plt.title(f"GT: {class_names[true_label]}\nPred: {class_names[pred_label]}")
        plt.axis('off')
        count += 1

    plt.tight_layout()
    plt.show()

# 실행
plot_confusion_matrix(y_true, y_pred, class_names)
show_misclassified_with_gradcam(misclassified, class_names, max_images=5)
