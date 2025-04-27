import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import torch
import timm

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# 상위 k개 혼동 페어 추출 함수
def get_topk_confusing_pairs(y_true, y_pred, class_names, topk=5) -> list:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    np.fill_diagonal(cm, 0)  # 정답 맞춘 항목은 제외
    pair_scores = []

    for i, j in itertools.product(range(len(class_names)), repeat=2):
        if i != j and cm[i][j] > 0:
            pair_scores.append(((class_names[i], class_names[j]), cm[i][j]))

    # 출현 빈도순 정렬
    pair_scores.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pair_scores[:topk]
    return top_pairs

# 모델 로딩 함수
def load_model(model_name, checkpoint_path, num_classes=7):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda')['model_state_dict'])
    model.to('cuda')
    model.eval()
    return model