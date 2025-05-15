from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
# Custom Dataset
class AlbumentationsDataset(Dataset):
    def __init__(self, root, transform):
        self.dataset = ImageFolder(root=root)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        image = self.transform(image=image)['image']
        return image, label