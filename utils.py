import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

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