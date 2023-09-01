import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision

def ordinary_step(net, batch, batch_idx, **kw):
	images, targets = batch
	images = [x.to(kw['device']) for x in images]
	targets = [{k: v.to(kw['device']) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
	loss_dict = net(images, targets)
	loss = sum(loss for loss in loss_dict.values())
	return {'loss':loss * len(images)}


def iou_step(net, batch, batch_idx, **kw):
	images, targets = batch
	images = [x.to(kw['device']) for x in images]
	targets = [{k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

	outputs = net(images)
	outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

	for p in outputs:
		ind = p['scores'] > .5
		p['boxes'] = p['boxes'][ind]
		p['scores'] = p['scores'][ind]
		p['labels'] = p['labels'][ind]

	from torchmetrics.detection import MeanAveragePrecision
	metric = MeanAveragePrecision(iou_type="bbox")
	metric.update(outputs, targets)
	# iou = sum([torchvision.ops.box_iou(o['boxes'], t['boxes']).max(dim=0).values.mean() for o, t in zip(outputs, targets)])

	return {k:v*len(images) for k, v in metric.compute().items()}



def rand_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	scores = net(inputs + kw['noise_level'] * torch.randn_like(inputs))
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def augmented_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	scores_, inputs_ = forward_with_sampling(net, inputs, **kw)
	loss_ = F.cross_entropy(scores_.permute(1, 2, 0), labels.unsqueeze(1).expand(-1, kw['num'] + 1), reduction = 'none')
	
	sigma = torch.std(loss_[:, 1:], dim = 1)
	mu = torch.mean(loss_[:, 1:], dim = 1)
	loss = loss_[:, 0].sum()

	with torch.no_grad():
		_, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float()

		correct = correct_[0].sum()
		hypothesis_accuracy	= {f'hypothesis/{v}-{alpha}':sprt(correct_, v, alpha).sum() for v in (0.9, ) for alpha in (0.01,)}
		correct_ = correct_.mean(dim = 0)		
		augmented_accuracy = correct_.sum()
		quantile_accuracy = {f'quantile/{v}':(correct_ > v).sum().float() for v in (0.9, 0.95, 0.975, 0.99, 0.999)}


	return {'loss':loss, 'correct':correct, **hypothesis_accuracy, 'augmented':augmented_accuracy, **quantile_accuracy, 'mu':mu.sum(), 'sigma':sigma.sum()}


def certify_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	scores_adv = net(kw['atk'](inputs, labels))

	scores_, inputs_ = forward_with_sampling(net, inputs, **kw)
	loss_ = F.cross_entropy(scores_.permute(1, 2, 0), labels.unsqueeze(1).expand(-1, kw['num'] + 1), reduction = 'none')
	
	sigma = torch.std(loss_[:, 1:], dim = 1)
	mu = torch.mean(loss_[:, 1:], dim = 1)
	loss = loss_[:, 0].sum()

	with torch.no_grad():
		_, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float()

		_, max_labels = scores_adv.max(1)
		correct_adv = (max_labels == labels).float().cpu()

		correct = correct_[0].sum()
		hypothesis_accuracy	= {f'hypothesis/{v}-{alpha}':torch.min(correct_adv, sprt(correct_, v, alpha)).sum() for v in (0.9,) for alpha in (0.01,)}
		correct_ = correct_.mean(dim = 0)		
		augmented_accuracy = correct_.sum()
		quantile_accuracy = {f'quantile/{v}':(correct_ > v).sum().float() for v in (0.9, 0.95, 0.975, 0.99, 0.999)}


	return {'loss':loss, 'correct':correct, **hypothesis_accuracy, 'augmented':augmented_accuracy, **quantile_accuracy, 'mu':mu.sum(), 'sigma':sigma.sum(), 'attacked':correct_adv.sum()}


def binom_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	outputs = kw['atk'](inputs, labels)
	scores = net(outputs.adv_images)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()

	certified = (outputs.certified).sum().cpu()

	return {'loss':loss, 'correct':correct, 'certified':certified}



def prl_step(net, batch, batch_idx, **kw):# dimension problem
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	scores_, inputs_ = forward_with_sampling(net, inputs, **kw)
	loss_ = F.cross_entropy(scores_.permute(1, 2, 0), labels.unsqueeze(1).expand(-1, kw['num'] + 1), reduction = 'none')

	alpha = torch.quantile(loss_.detach(), kw['threshold'], dim = 1)

	loss = F.relu(loss_ - alpha).mean(dim = 1).sum() / (1 - kw['threshold'])

	with torch.no_grad():
		_, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float()

		correct = correct_[0].sum()	
		correct_ = correct_.mean(dim = 0)
		
		augmented_accuracy = correct_.sum()
		quantile_accuracy = (correct_ > kw['threshold']).sum().float()

	return {'loss':loss, 'correct':correct, 'augmented':augmented_accuracy, 'quantile':quantile_accuracy, 'alpha':alpha}


def our_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	scores_ = forward_with_sampling(net, inputs, **{k:v for k, v in kw.items() if k in ['eps', 'sample_', 'microbatch_size', 'num']}).all_logits

	loss_ = F.cross_entropy(scores_.permute(1, 2, 0), labels.unsqueeze(1).expand(-1, kw['num'] + 1), reduction = 'none')

	sigma = torch.std(loss_[:, 1:], dim = 1)
	mu = torch.mean(loss_[:, 1:], dim = 1)

	loss = (mu + kw['z'] * sigma).sum()

	with torch.no_grad():
		_, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float()

		correct = correct_[0].sum()	
		correct_ = correct_.mean(dim = 0)
		
		augmented_accuracy = correct_.sum()
		quantile_accuracy = (correct_ > kw['threshold']).sum().float()

	return {'loss':loss, 'correct':correct, 'augmented':augmented_accuracy, 'quantile':quantile_accuracy, 'mu':mu.sum(), 'sigma':sigma.sum()}

def tradesplus_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	scores_, inputs_ = forward_with_sampling(net, inputs, **kw)

	loss_ = F.cross_entropy(scores_.permute(1, 2, 0), labels.unsqueeze(1).expand(-1, kw['num'] + 1), reduction = 'none')

	sigma = torch.std(loss_[:, 1:], dim = 1)
	mu = torch.mean(loss_[:, 1:], dim = 1)

	loss = (mu + kw['z'] * sigma).sum()

	inputs_ = kw['atk'](inputs, labels)

	scores = net(inputs_)
	loss += F.cross_entropy(scores, labels, reduction = 'sum') + kw['z'] * F.kl_div(torch.log_softmax(scores, dim=1), F.softmax(net(inputs), dim=1), reduction='batchmean') * inputs.shape[0]


	with torch.no_grad():
		_, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float()

		correct = correct_[0].sum()	
		correct_ = correct_.mean(dim = 0)
		
		augmented_accuracy = correct_.sum()
		quantile_accuracy = (correct_ > kw['threshold']).sum().float()

	return {'loss':loss, 'correct':correct, 'augmented':augmented_accuracy, 'quantile':quantile_accuracy, 'mu':mu.sum(), 'sigma':sigma.sum()}


def attacked_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	inputs_ = kw['atk'](inputs, labels)

	scores = net(inputs_)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def trades_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	inputs_ = kw['atk'](inputs, labels)

	scores = net(inputs_)
	loss = F.cross_entropy(scores, labels, reduction = 'sum') + kw['z'] * F.kl_div(torch.log_softmax(scores, dim=1), F.softmax(net(inputs), dim=1), reduction='batchmean') * inputs.shape[0]

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}


def predict_step(net, batch, batch_idx, **kw):
	inputs, _ = batch
	inputs = inputs.to(kw['device'])
	scores = net(inputs)

	max_scores, max_labels = scores.max(1)
	return {'predictions':max_labels}

def perturbation_estimate_step(net, batch, batch_idx, **kw): # num = 40, batch_size = 10000
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])

	eps = torch.ones_like(labels).view(1, -1, 1, 1, 1) * 0.5

	for _ in range(num_estimates):
		scores_, inputs_ = forward_with_sampling(net, inputs, **kw)
		_, max_labels_ = scores_.max(-1)
		correct_ = (max_labels_ == labels).float().mean(dim = 0).view(-1, 1, 1, 1)
		eps += (correct_ - 0.5)# * ((correct_ < 0.5).float() * 30 + 1)
		eps = torch.clamp(eps, lb, ub)

	return {'eps':eps.squeeze(), 'correct':correct_.squeeze()}#, 'samples':inputs_}
