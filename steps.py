import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision

def d_step(net, batch, batch_idx, **kw):
	images, targets = batch
	images = [x.to(kw['device']) for x in images]
	targets = [{k: v.to(kw['device']) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
	loss_dict = net(images, targets)
	loss = sum(loss for loss in loss_dict.values())
	return {'loss':loss * len(images)}

from torchmetrics.detection import MeanAveragePrecision

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

	metric = MeanAveragePrecision(iou_type="bbox", box_format='xyxy', iou_thresholds=[0.5], rec_thresholds=[0.5])
	mAP = metric.forward(outputs, targets)['map_50'] * len(images)

	return {'mAP50':mAP}

def attacked_iou_step(net, batch, batch_idx, **kw):
	images, targets = batch
	images = torch.stack(images).to(kw['device'])
	targets = [{k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

	labels = torch.nn.utils.rnn.pad_sequence([t['boxes'] for t in targets], batch_first = True)
	inputs_ = kw['atk'](images, labels).detach()
	
	outputs = net(inputs_)
	outputs = [{k: v.detach().to('cpu') for k, v in t.items()} for t in outputs]

	for p in outputs:
		ind = p['scores'] > .5
		p['boxes'] = p['boxes'][ind]
		p['scores'] = p['scores'][ind]
		p['labels'] = p['labels'][ind]

	with torch.no_grad():
		metric = MeanAveragePrecision(iou_type="bbox", box_format='xyxy', iou_thresholds=[0.5], rec_thresholds=[0.5])
		mAP = metric.forward(outputs, targets)['map_50'] * len(images)

	return {'mAP50':mAP}


def c_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	scores = net(inputs)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}

def attacked_c_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	inputs_ = kw['atk'](inputs, labels)

	scores = net(inputs_)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}

from soft_editdistance import similarity_from_edit_distance, soft_similarity_from_edit_distance

def transcribe_eval_step(net, batch, batch_idx, **kw):
	images, targets = batch
	images = [x.to(kw['device']) for x in images]
	targets = [{k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

	outputs = net(images)

	preds = [''.join(map(str, t.tolist())) for t in outputs.predictions]
	labels = [''.join(map(str, t['string'].tolist())) for t in targets]

	# soft = [soft_similarity_from_edit_distance(s1_probs.softmax(-1), s2['string']) for s1_probs, s2 in zip(outputs.logits, targets)]

	edit = torch.tensor([similarity_from_edit_distance(s1, s2) for s1, s2 in zip(preds, labels)]).sum(0)
	
	return {'distance':edit[0], 'similarity':edit[1]}

def attacked_transcribe_step(net, batch, batch_idx, **kw):
	images, targets = batch
	images = torch.stack(images).to(kw['device'])
	targets = [{k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

	labels = torch.nn.utils.rnn.pad_sequence([t['string'] for t in targets], batch_first = True, padding_value=10000)
	inputs_ = kw['atk'](images, labels).detach()
	
	outputs = net(inputs_)

	preds = [''.join(map(str, t.tolist())) for t in outputs.predictions]
	labels = [''.join(map(str, t['string'].tolist())) for t in targets]

	# soft = [soft_similarity_from_edit_distance(s1_probs.softmax(-1), s2['string']) for s1_probs, s2 in zip(outputs.logits, targets)]

	edit = torch.tensor([similarity_from_edit_distance(s1, s2) for s1, s2 in zip(preds, labels)]).sum(0)
	
	return {'distance':edit[0], 'similarity':edit[1]}