
from typing import Any
import pytorch_lightning as L
import torchvision
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch
import argparse
import random
import pandas as pd
import glob
import os
import numpy as np
from itertools import chain
from dataset import CustomDataset

class ImageClassificationPytochLightning(L.LightningModule):
    def __init__(self, classes, lr) -> None:
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained="True")
        for param in self.model.parameters():
            param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        self.num_ftrs = self.model.fc.in_features
        self.num_classes = len(classes)
        self.model.fc = torch.nn.Linear(self.num_ftrs, self.num_classes)
        print("self.num_classes)", self.num_classes)
        self.lr = lr
        self.criterion = CrossEntropyLoss()
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        x,y = batch
        y = y.view(-1).type(torch.long)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    # # print loss after each epoch
    # def training_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} loss: {avg_loss:.2f}")
    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=self.lr)  

class ConstructionDataset(L.LightningDataModule):
    def __init__(self, batch_size, path_dataset, data, classes):
        self.batch_size = batch_size
        self.df = data
        self.path_dataset = path_dataset
        self.classes = classes
        super().__init__()
    def setup(self, stage=None):
        np.random.seed(0)
        torch.manual_seed(0)
        numbers_random = list(range(10))
        random.shuffle(numbers_random)
        ## load and sLit dataset in ratio 8,1,1
        train_sequences = [] 
        val_sequences = [] 
        test_sequences = [] 
        for camera_folder in glob.glob(os.path.join(self.path_dataset, "cam_*")):
            sequences = [] 
            for i in range(10):
                sequence = sorted(glob.glob(os.path.join(camera_folder, '*.jpg_'+str(i)+'.png')))
                sequences.append(sequence) # list of 10 sequences

            selected_sequences = [sequences[num] for num in numbers_random[:8]]
            train_sequences.append(selected_sequences)
            val_sequences.append(sequences[numbers_random[8]])
            test_sequences.append(sequences[numbers_random[9]])

        # train_sequences list of 4 lists of 8 sequences
        flattened_train = list(chain.from_iterable(list(chain.from_iterable(train_sequences)))) # 32 sequences
        flattened_val = list(chain.from_iterable(val_sequences)) # 4 sequences 
        flattened_test = list(chain.from_iterable(test_sequences))# 4 sequences

        # remove images without label
        flattened_train = [train for train in flattened_train if not self.df.loc[self.df['images'] == os.path.basename(train), 'labels'].empty]
        flattened_val = [val for val in flattened_val if not self.df.loc[self.df['images'] == os.path.basename(val), 'labels'].empty]
        flattened_test = [test for test in flattened_test if not self.df.loc[self.df['images'] == os.path.basename(test), 'labels'].empty]

        self.train_set = CustomDataset(images_path = flattened_train, classes = self.classes, data_frame=self.df)
        self.val_set = CustomDataset(images_path = flattened_val, classes = self.classes,data_frame=self.df)
        self.test_set = CustomDataset(images_path = flattened_test, classes = self.classes,data_frame=self.df)
    def train_dataloader(self) -> Any:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self) -> Any:
        return DataLoader(self.val_set, batch_size=self.batch_size)
    def test_dataloader(self) -> Any:
        return DataLoader(self.test_set, batch_size=self.batch_size)

def main():
    parser = argparse.ArgumentParser(description='Image classfication')
    parser.add_argument('-l', '--lr', default=0.0005)
    parser.add_argument('-b', '--batch_size', default=4)
    parser.add_argument('-e', '--num_epoch', default=50)
    parser.add_argument('-tr','--train', action='store_true')
    parser.add_argument('-t','--test', action='store_true')
    args = parser.parse_args()

    N_EPOCHS = args.num_epoch
    LR = args.lr
    batch_size = args.batch_size
    path_dataset = "data/foundation_images"
    path_labels = "data/updated_stage_labels.csv"
    df = pd.read_csv(path_labels)
    classes = df['labels'].unique().astype(str)


    # Initialize LightningModule and LightningDataModule
    model = ImageClassificationPytochLightning(classes=classes, lr=LR)
    data_module = ConstructionDataset(batch_size=batch_size, path_dataset=path_dataset, data=df, classes=classes)

    # Initialize the Trainer
    trainer = L.Trainer(gpus=1, max_epochs=N_EPOCHS)  # Change gpus to None for CPU

    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
