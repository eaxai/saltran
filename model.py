import time
import os
import copy

from tqdm import tqdm
from tqdm.auto import trange

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np

from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from self_attention_cv.vit import ResNet50ViT
from mit1003 import MIT1003Data
from utils import corr_coeff, kld_loss, nss, train_val_dataset
from saltran import SalTran
from torch.optim import lr_scheduler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

	model = SalTran()
	model.to(device)
	dataset = MIT1003Data('../saldata/MIT1003/', 'ALLSTIMULI', 'ALLFIXATIONMAPS')
	ds = train_val_dataset(dataset)
	dataloaders = {x: torch.utils.data.DataLoader(ds[x], batch_size=2,
												  shuffle=True, num_workers=0)
					for x in ['train', 'val']}

	dataset_sizes = {x: len(ds[x]) for x in ['train', 'val']}
	print(dataset_sizes)
	# Observe that all parameters are being optimized
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	# Decay LR by a factor of 0.1 every 7 epochs
	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	best_f1 = 0.0

	for epoch in trange(2, desc='epochs'):
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0

			for data in tqdm(dataloaders[phase]):
				imgs = data['img'].float().to(device)
				fiks = data['fix'].to(device)
				maps = data['map'].float().to(device)
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					outs = model(imgs)
					losses = []
					losses.append(corr_coeff(outs.exp(), maps))
					losses.append(kld_loss(outs, maps))
					losses.append(nss(outs.exp(), fiks))
					losses = [l.mean(1).mean(0) for l in losses]
					loss = sum(losses)
					if phase == 'train':
						loss.backward()
						optimizer.step()

				running_loss += loss * imgs.size(0)

			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / dataset_sizes[phase]
			print(epoch_loss)

	time_elapsed = time.time() - since
	print(time_elapsed)