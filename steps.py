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

def map_step(net, batch, batch_idx, **kw):
	images, targets = batch
	images = [x.to(kw['device']) for x in images]
	# targets = [{k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

	outputs = net(images)
	outputs = [{k: v.detach().to('cpu') for k, v in t.items()} for t in outputs]

	# for p in outputs:
	# 	ind = p['scores'] > .5
	# 	p['boxes'] = p['boxes'][ind]
	# 	p['scores'] = p['scores'][ind]
	# 	p['labels'] = p['labels'][ind]

	metric = MeanAveragePrecision(iou_type="bbox", box_format='xyxy', iou_thresholds=[0.5, 0.75], rec_thresholds = [0.3, 0.5, 0.7])
	mAP = metric.forward(outputs, targets)

	return {k:v * len(images) for k, v in mAP.items() if k in {'map', 'map_75', 'map_50'}}

def _iou_calculate_step(images, model, labels):
	outputs = model(images)
	boxes = [p['boxes'][p['scores'] > .5] for p in outputs]

	ious = []
	for b, l in zip(boxes, labels):
		iou = torchvision.ops.box_iou(b, l)
		if iou.numel() > 0:
			ious.append(iou.max(dim = 1)[0].mean().item())
		else:
			ious.append(0)

	return torch.tensor(ious, device = images.device) - 0.4

def attacked_map_step(net, batch, batch_idx, **kw):
	images, targets = batch
	images_ = kw['attack'](torch.stack(images).to(kw['device']), _iou_calculate_step, labels = [t['boxes'].to(kw['device']) for t in targets], model = net, **kw['attack_config']).detach()
	
	outputs = net(images_)
	outputs = [{k: v.detach().to('cpu') for k, v in t.items()} for t in outputs]

	# for p in outputs:
	# 	ind = p['scores'] > .5
	# 	p['boxes'] = p['boxes'][ind]
	# 	p['scores'] = p['scores'][ind]
	# 	p['labels'] = p['labels'][ind]

	with torch.no_grad():
		metric = MeanAveragePrecision(iou_type="bbox", box_format='xyxy', iou_thresholds=[0.5, 0.75])
		mAP = metric.forward(outputs, targets)
		
	return {k:v * len(images) for k, v in mAP.items() if k in {'map', 'map_75', 'map_50'}}


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

#from soft_editdistance import similarity_from_edit_distance, soft_similarity_from_edit_distance

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
