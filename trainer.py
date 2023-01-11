from pytorch_lightning import LightningModule, Trainer, seed_everything, LightningDataModule
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torchmetrics
import torchvision
import torch
from torch.utils.data import DataLoader

from model import Classifier

class plData(LightningDataModule):
    def __init__(self, train_path, test_path, val_path, batch_size=64):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop((536, 5)),
            torchvision.transforms.Grayscale(), ])

        self.batch_size = batch_size

    def setup(self, stage):
        self.train_dataset = torchvision.datasets.ImageFolder(self.train_path, transform=self.transform)
        self.val_dataset = torchvision.datasets.ImageFolder(self.val_path, transform=self.transform)
        self.test_dataset = torchvision.datasets.ImageFolder(self.test_path, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    NUM_EPOCHS = 20
    batch_size = 128

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath='C:/work/freelance/wood-detection-segmentation/weights',
        filename="Classifier_A-{epoch:02d}-{val_loss:.3f}",

    )

    trainer = Trainer(
        accelerator="auto",
        max_epochs=NUM_EPOCHS,
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback],
        logger=[TensorBoardLogger("logs/", name="Classifier"), CSVLogger(save_dir="logs/")],
    )

    model = Classifier()
    data = plData(train_path="../data/annotated/train/", val_path="../data/annotated/val/",
                  test_path="../data/annotated/test")

    trainer.fit(model, data)
