import torchvision
import torch
from torch.utils.data import Dataset
from PIL import Image



class weatherData(Dataset):
    def __init__(self,image_path,label_path,transform=None):
        self.image_path=image_path
        self.label_path=label_path
        self.transform=transform
    def __len__(self):
        return len(self.label_path)
    def __getitem__(self, index):
        img=Image.open(self.image_path[index])
        label=self.label_path[index]
        if self.transform is not None:
            img=self.transform(img)
        return img,label

