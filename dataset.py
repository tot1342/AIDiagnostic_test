from skimage import exposure
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import os

np.random.seed(42)
torch.manual_seed(42)


class XRaySet(Dataset):
    def __init__(self, path_img, path_mask, data_path, type_="train"):
        self.path_img = path_img
        self.path_mask = path_mask
        
        self.type_ = type_
        
        self.data_path = list(data_path)

        self.transform = transforms.Compose([
            self.equalize,
            transforms.ToTensor(),
            transforms.Normalize((0.49), (0.25)),
            transforms.Resize((512, 512))
        ])
               
    def equalize(self, image):
        """ 
            Предобработка изображения
        """
        image = exposure.equalize_hist(image)
        image = exposure.equalize_adapthist(image / np.max(image))
        return image
        
    def __len__(self):
        return len(self.data_path)
    
    def augm(self, image, mask):
        """
            Аугментация выходных данным с помощью добавления небольшого шума
        """
        image = image + np.random.randn(*image.shape) * image.var()**0.1
        return image, mask

    def __getitem__(self, idx):
        """
            Подгружаем данные, в трейновую выборку аугментируем, затем обрабатываем
        """
        image = np.load(os.path.join(self.path_img, self.data_path[idx]), allow_pickle=True).astype("int16")
        mask = np.load(os.path.join(self.path_mask, self.data_path[idx]), allow_pickle=True).astype(float)
        
        if self.type_ == "train":
            image, mask = self.augm(image, mask)
            
        image = self.transform(image).float()  
        mask = torch.tensor((mask > 0.001).astype(float))
        
        return image, mask

