import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def train_val_dataset(dataset, val_split=0.5):
	train_idx, val_idx = train_test_split(list(range(len(dataset))),
										test_size=val_split)
	datasets = {}
	datasets['train'] = Subset(dataset, train_idx)
	datasets['val'] = Subset(dataset, val_idx)
	return datasets

def normalize_tensor(tensor, rescale=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin
    tsum = tensor.sum()
    if tsum > 0:
        return tensor / tsum
    print("Zero tensor")
    tensor.fill_(1. / tensor.numel())
    return tensor

def log_softmax(x):
    x_size = x.size()
    x = x.view(x.size(0), -1)
    x = F.log_softmax(x, dim=1)
    return x.view(x_size)

def nss(pred, fixations):
	size = pred.size()
	new_size = (-1, size[-1] * size[-2])
	pred = pred.reshape(new_size)
	fixations = fixations.reshape(new_size)

	pred_normed = (pred - pred.mean(-1, True)) / pred.std(-1, keepdim=True)
	results = []
	for this_pred_normed, mask in zip(torch.unbind(pred_normed, 0),
									  torch.unbind(fixations, 0)):
		if mask.sum() == 0:
			results.append(torch.ones([]).float().to(fixations.device))
			continue
		nss_ = torch.masked_select(this_pred_normed, mask)
		nss_ = nss_.mean(-1)
		results.append(nss_)
	results = torch.stack(results)
	results = results.reshape(size[:2])
	return results


def corr_coeff(pred, target):
	size = pred.size()
	new_size = (-1, size[-1] * size[-2])
	pred = pred.reshape(new_size)
	target = target.reshape(new_size)

	cc = []
	for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
		xm, ym = x - x.mean(), y - y.mean()
		r_num = torch.mean(xm * ym)
		r_den = torch.sqrt(
			torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
		r = r_num / r_den
		cc.append(r)

	cc = torch.stack(cc)
	cc = cc.reshape(size[:2])
	return cc  # 1 - torch.square(r)


def kld_loss(pred, target):
	loss = F.kl_div(pred, target, reduction='none')
	loss = loss.sum(-1).sum(-1)
	return loss
