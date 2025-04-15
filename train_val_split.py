import os
import random
import shutil

def split_train_val(train_dir, val_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    os.makedirs(val_dir, exist_ok=True)

    classes = os.listdir(train_dir)
    for cls in classes:
        cls_train_path = os.path.join(train_dir, cls)
        cls_val_path = os.path.join(val_dir, cls)
        os.makedirs(cls_val_path, exist_ok=True)

        images = os.listdir(cls_train_path)
        num_val = int(len(images) * val_ratio)
        val_images = random.sample(images, num_val)

        for img in val_images:
            src_path = os.path.join(cls_train_path, img)
            dst_path = os.path.join(cls_val_path, img)
            shutil.move(src_path, dst_path)

        print(f"[{cls}] moved {num_val} images to validation set.")

# 사용 예시
split_train_val(train_dir='./data/train', val_dir='./data/val', val_ratio=0.2)
