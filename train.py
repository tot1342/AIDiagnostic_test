import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from collections import OrderedDict
import random
import glob
import os

from model import UNet
from dice import DiceLoss
from dataset import XRaySet

random.seed(42)
torch.manual_seed(42)

#configs (можно вынести в yalm фаил)
BATCH_SIZE = 3
TRAIN_PERSENT = 0.8
PATH_IMAGE = "./subset_data/subset_img"
PATH_MASK = "./subset_data/subset_masks"
LR = 1e-4

class ObjectDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        
    def setup(self, stage=None):
        """
            Подгружаем данные, перемешиваем и разбиваем на обучающую и валидационную выборку
        """
        data_path = list(map(lambda x: os.sep.join(x.split(os.sep)[-2:]), glob.glob(f"{PATH_IMAGE}{os.sep}*{os.sep}*")))
        random.shuffle(data_path)
        
        self.train_dataset = XRaySet(path_img=PATH_IMAGE, 
                                     path_mask=PATH_MASK, 
                                     data_path=data_path[:int(len(data_path) * TRAIN_PERSENT)], 
                                     type_="train")
        self.val_dataset = XRaySet(path_img=PATH_IMAGE, 
                                     path_mask=PATH_MASK, 
                                     data_path=data_path[int(len(data_path) * TRAIN_PERSENT):], 
                                     type_="val")

    def train_dataloader(self):
        """
            Метод для создания обучающего даталоадера
        """
        train_dataloader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
        return train_dataloader

    def val_dataloader(self):
        """
            Метод для создания валидационного даталоадера
        """
        val_dataloader = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)
        return val_dataloader
    
#     def test_dataloader(self):
#         """
#             При наличии данных можно сделать тестовый даталоадер
#         """
#         test_dataloader = DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)
#         return test_dataloader


class Segmenter(LightningModule):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()    
        self.model = UNet(1, 1, bilinear=True)

    def forward(self, z):
        return self.model(z)

    def training_step(self, batch, batch_idx):
        """
            Шаг обучения
        """
        img, target = batch
        mask_out = self(img)
        loss = self.dice(mask_out, target)
        tqdm_dict = {"train_loss": loss}
        # результат сохраняется для tensorboard
        self.log("train_loss", loss)
        output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output
    
    def validation_step(self, batch, batch_idx):
        """
            Шаг валидации
        """
        img, target = batch
        mask_out = self(img)
        # Добавляем в tensorboard примеры выхода модели
        for i in range(mask_out.size(0)):
            im = make_grid([img[i]*0.25 + 0.49, mask_out[i], torch.unsqueeze(target[i], 0)])
            self.logger.experiment.add_image(f"im_{batch_idx}_{i}", im) 
        
        metric = 1 - self.dice(mask_out, target)
        tqdm_dict = {"val": metric}
        # результат сохраняется для tensorboard
        self.log("val", metric)
        output = OrderedDict({"val": metric, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output

    def configure_optimizers(self):
        """
            Оптимайзер с уменьшением градиента
        """
        opt_g = torch.optim.Adam(self.model.parameters(), lr=LR)
        sch = torch.optim.lr_scheduler.StepLR(opt_g, step_size=20, gamma=0.5)
        return {
            "optimizer":opt_g,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss",
                
            }
        }


if __name__ == "__main__":
    model = Segmenter()

    dm = ObjectDataModule()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val",
        mode="max",
        dirpath="./output",
        filename="Unet-{epoch:02d}-{val:.4f}",
    )
    
    logger = TensorBoardLogger("tb_logs", name="UNet")

    trainer = Trainer(gpus=1, accumulate_grad_batches=2, max_epochs=80, default_root_dir="./", logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model, dm)
    
    # При наличии данных можно создать тестовую выборку и проверить результат
    #trainer.test(model, dm)
    
    
