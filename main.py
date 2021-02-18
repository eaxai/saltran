import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from  torchvision import datasets, models, transforms
import numpy as np
import torchvision
from models import SalTran

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
		transforms.Resize((240, 320)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])


# data_dir = '/Users/eax/worx/deeplearning/phd/labsimdata/05'
data_dir = './BDDA/'
trainds = datasets.ImageFolder("./BDDA/training/acl/0.5_0.1", transform=data_transform)
testds = datasets.ImageFolder("./BDDA/test/acl/0.5_0.1", transform=data_transform)
image_datasets = {}
image_datasets['train'] = trainds
image_datasets['val'] = testds
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
											shuffle=True, num_workers=4)
			  for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes


# dataset setup

# model

# feature extraction
resnet = models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False


saltran = SalTran()

def train_epoch():
    pass

def eval_epoch()

def train():
    pass

def eval():
    pass

def main():
    pass

if __name__ == "__main__":
    main()
