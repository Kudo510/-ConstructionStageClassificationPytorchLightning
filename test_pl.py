import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import os
import argparse
import pytorch_lightning as L
from model import ImageClassificationPytochLightning

parser = argparse.ArgumentParser(description='Test and Visualization')
parser.add_argument('-i', '--idx', default=0, type=int)
args = parser.parse_args()

img_path =  np.loadtxt('test_list.txt', dtype=str).tolist()

transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])


path_labels = "data/updated_stage_labels.csv"
df = pd.read_csv(path_labels)
classes = df['labels'].unique().astype(str)

# model = ImageClassificationPytochLightning.load_from_checkpoint("lightning_logs/version_11/checkpoints/epoch=49-step=26100.ckpt")
# model = ImageClassificationPytochLightning(classes=classes)

checkpoint_model = ImageClassificationPytochLightning.load_from_checkpoint("lightning_logs/version_12/checkpoints/epoch=49-step=26100.ckpt")
model = ImageClassificationPytochLightning(classes=classes, **checkpoint_model.init_model_args())


# Test for single image in the test set list. e.g here the first image
image = datasets.folder.default_loader(img_path[args.idx])
resized_image = transform(image).unsqueeze(dim = 0)
model.eval()  
with torch.no_grad():
    predictions = model(resized_image)
    print('predictions', predictions)
    predicted_label = torch.argmax(predictions, 1)
    print(f"predicted_label for image: {os.path.basename(img_path[args.idx])}: ",classes[predicted_label] )
    predicted_class = classes[predicted_label.item()] 
    print(f"GT label for {os.path.basename(img_path[args.idx])}: ", df.loc[df['images'] == str(os.path.basename(img_path[args.idx])), 'labels'].values[0])

unnormalized_img = (resized_image.squeeze().cpu() / 2) + 0.5  # Unnormalize for visualizatioin
npimg = unnormalized_img.numpy()
fig, axs = plt.subplots(1, figsize=(15, 5))
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()


# # Test for all the images in the test set
# img_path =  np.loadtxt('test_list.txt', dtype=str).tolist()
# for i in range(len(img_path)):
#     image = datasets.folder.default_loader(img_path[i])
#     resized_image = transform(image).unsqueeze(dim = 0)
#     #print("resize image", resized_image)
#     model.eval()  
#     with torch.no_grad():
#         predictions = model(resized_image)
#         # print('predictions', predictions)
#         predicted_label = torch.argmax(predictions, 1)
#         print(f"predicted_label for image: {os.path.basename(img_path[i])}: ",classes[predicted_label] )
#         predicted_class = classes[predicted_label.item()] 
#         print(f"GT label for {os.path.basename(img_path[i])}: ", df.loc[df['images'] == str(os.path.basename(img_path[i])), 'labels'].values[0])
