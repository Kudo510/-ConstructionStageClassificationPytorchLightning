
from typing import Any
import pytorch_lightning as L
from torch.utils.data import DataLoader
import argparse
import random
import pandas as pd
import glob
import os
from dataset import CustomDataset
from model import ImageClassificationPytochLightning
from sklearn.model_selection import train_test_split
import logging
from pytorch_lightning.loggers import WandbLogger


logging.basicConfig(filename="training_log.txt", level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)
wandb_logger = WandbLogger(project="ImageClassificationPytochLightning", name="training", log_model="all")

class ConstructionDataset(L.LightningDataModule):
    def __init__(self, batch_size, path_dataset, data, classes):
        self.batch_size = batch_size
        self.df = data
        self.path_dataset = path_dataset
        self.classes = classes
        super().__init__()

    def setup(self, stage=None): # need to put stage = None here. error occurs otherwise
        random.seed(0)
        sequences = [] 
        for camera_folder in glob.glob(os.path.join(self.path_dataset, "cam_*")):
            for i in range(10):
                files = sorted(glob.glob(os.path.join(camera_folder, '*.jpg_'+str(i)+'.png')))
                sequences.extend(files)
        train_sequences, remaining_sequences = train_test_split(sequences, test_size=0.2, random_state=0)
        val_sequences, test_sequences = train_test_split(remaining_sequences, test_size=0.5, random_state=0)
        # remove images without label
        train_sequences = [train for train in train_sequences if not self.df.loc[self.df['images'] == os.path.basename(train), 'labels'].empty]
        val_sequences = [val for val in val_sequences if not self.df.loc[self.df['images'] == os.path.basename(val), 'labels'].empty]
        test_sequences = [test for test in test_sequences if not self.df.loc[self.df['images'] == os.path.basename(test), 'labels'].empty]

        self.train_set = CustomDataset(images_path = train_sequences, classes = self.classes, data_frame=self.df)
        self.val_set = CustomDataset(images_path = val_sequences, classes = self.classes,data_frame=self.df)
        self.test_set = CustomDataset(images_path = test_sequences, classes = self.classes,data_frame=self.df) 

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
    log.info(f"Initialize Model with learning rate {LR}")
    model = ImageClassificationPytochLightning(classes=classes, lr=LR)
    data_module = ConstructionDataset(batch_size=batch_size, path_dataset=path_dataset, data=df, classes=classes)

    # Initialize the Trainer
    log.info(f"Initialize the Trainer")
    trainer = L.Trainer(gpus=1, max_epochs=N_EPOCHS, logger=wandb_logger)  # Change gpus to None for CPU

    # Train the model
    trainer.fit(model, data_module)
    # # Save trained model as a checkpoint-
    # trainer.save_checkpoint("NCF_Trained.ckpt")

    if args.train:
        log.info("Training the model")
        trainer.fit(model, data_module)
    else:
        log.warning("No training performed. Use --train flag to train the model.")


if __name__ == "__main__":
    main()
