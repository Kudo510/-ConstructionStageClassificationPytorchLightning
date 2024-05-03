
from typing import Any
import pytorch_lightning as L
import torchvision
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
from sklearn.metrics import accuracy_score


class ImageClassificationPytochLightning(L.LightningModule):
    def __init__(self, classes, lr=0.0005) -> None:
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
        _,loss,_ = self._get_preds_loss_accuracy(batch)
        self.log('training_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    # # print loss after each epoch
    # def training_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     print(f"Epoch {self.current_epoch + 1}/{self.trainer.max_epochs} loss: {avg_loss:.2f}")

    def validation_step(self, batch, batch_idx):
        _,loss, acc = self._get_preds_loss_accuracy(batch)
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, logger=True)
        return loss, acc
    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar - so should use the function so that we dont have to repeat for val and train step - and this function will output prediction, accuracy, loss at the same time - really convinient"""
        x,y = batch
        y_gt= y.view(-1).type(torch.long)
        y_pred = self(x)
        loss = self.criterion(y_pred, y_gt)
        predictions = torch.argmax(y_pred, dim=-1)
        acc = accuracy_score(y_gt.detach().cpu().numpy(), predictions.detach().cpu().numpy()) # can set normalize=False to cal the acc , without dividing by batch size. sklearn can only precoess array so in numpy not torch- so need to convert to numpy
        return y_pred, loss, acc
    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=self.lr)  