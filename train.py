import torch 
import os
import random 
import numpy as np 
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import pandas as pd
from itertools import chain
import argparse

from model import ImageClassification
from dataset import CustomDataset

parser = argparse.ArgumentParser(description='Image classfication')
parser.add_argument('-l', '--lr', default=0.0005)
parser.add_argument('-b', '--batch_size', default=4)
parser.add_argument('-e', '--num_epoch', default=50)
parser.add_argument('-tr','--train', action='store_false')
parser.add_argument('-t','--test', action='store_true')
args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)
numbers_random = list(range(10))
random.shuffle(numbers_random)

path_dataset = "data/foundation_images"
path_labels = "data/updated_stage_labels.csv"
df = pd.read_csv(path_labels)
classes = df['labels'].unique().astype(str)

# cam1_f2023-05-30T10:30:13.705Z.jpg_0.png

## load and split dataset in ratio 8,1,1
train_sequences = [] 
val_sequences = [] 
test_sequences = [] 
for camera_folder in glob.glob(os.path.join(path_dataset, "cam_*")):
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
flattened_train = [train for train in flattened_train if not df.loc[df['images'] == os.path.basename(train), 'labels'].empty]
flattened_val = [val for val in flattened_val if not df.loc[df['images'] == os.path.basename(val), 'labels'].empty]
flattened_test = [test for test in flattened_test if not df.loc[df['images'] == os.path.basename(test), 'labels'].empty]


train_set = CustomDataset(images_path = flattened_train, classes = classes, data_frame=df)
val_set = CustomDataset(images_path = flattened_val, classes = classes,data_frame=df)
test_set = CustomDataset(images_path = flattened_test, classes = classes,data_frame=df)

batch_size = args.batch_size
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

# model = MyViT(n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
model = ImageClassification(classes=classes).to(device)
model.load_state_dict(torch.load("saved_checkpoints/best_checkpoint.pth"))

N_EPOCHS = args.num_epoch
LR = args.lr
optimizer = Adam(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()
# Training 
best_val_loss = float('inf')
if args.train:
    for epoch in trange(int(N_EPOCHS), desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device).view(-1).type(torch.long)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.detach().cpu().item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

        if epoch %10 == 0:
            torch.save(model.state_dict(), f'saved_checkpoints/model_checkpoint'+str(epoch)+'.pth')

        if epoch %10 == 0:
            # Validation
            val_loss = 0.0
            model.eval()  #
            with torch.no_grad():
                for batch_val in tqdm(val_loader, desc=f"Epoch {epoch + 1} in validation", leave=False):
                    x_val, y_val = batch_val
                    x_val, y_val = x_val.to(device), y_val.to(device).view(-1).type(torch.long)
                    y_hat_val = model(x_val)
                    loss = criterion(y_hat_val, y_val)
                    val_loss += loss.detach().cpu().item() / len(val_loader)
            print(f"Epoch {epoch + 1}/{N_EPOCHS} Validation Loss: {val_loss:.2f}")
            # Check if the validation loss is better than the best validation loss so far
            if val_loss < best_val_loss:
                # Update the best validation loss and save the model state
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                print("best_val_loss: ", best_val_loss)
                print("saving best model at epoch: ", epoch)
                torch.save(best_model_state, 'saved_checkpoints/best_model_checkpoint.pth')

# Testing            
if args.test: 
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x_test, y_test = batch
            x_test, y_test = x_test.to(device), y_test.to(device).view(-1).type(torch.long)
            y_hat_test = model(x_test)
            loss = criterion(y_hat_test, y_test)
            test_loss += loss.detach().cpu().item() / len(test_loader)
            correct += torch.sum(torch.argmax(y_hat_test, dim=1) == y_test).detach().cpu().item()
            total += len(x_test)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
print("Finish.")




