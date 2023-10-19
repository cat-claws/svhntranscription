import torch
torch.multiprocessing.set_sharing_strategy('file_system')

config = {
	'dataset':'svhnfull',
	'training_step':'steps.d_step',
	'batch_size':64,
	'optimizer':'SGD',
	'optimizer_config':{
		'lr':0.005,
		'momentum':0.9,
		'weight_decay':5e-4,
	},
	'scheduler':'StepLR',
	'scheduler_config':{
		'step_size':5,
		'gamma':0.1
	},
	'attack':'square_attack',
	'attack_config':{
		'eps':50/255,
		# 'loss':'iou_',
		'n_queries':5
		# 'alpha':0.2,
		# 'steps':40,
		# 'random_start':True,
	},
	'device':'cuda' if torch.cuda.is_available() else 'cpu',
	'validation_step':'steps.map_step',
	'attacked_step':'steps.attacked_map_step'
}

def main(config):

	from torchiteration import train, attack, validate
	m = torch.hub.load('cat-claws/nn', 'fasterrcnn', num_classes = 2).to(config['device'])

	import steps
	from torch.utils.tensorboard import SummaryWriter
	writer = SummaryWriter(comment = f"_{config['dataset']}_{m._get_name()}_{config['training_step']}", flush_secs=10)

	import sys
	sys.path.insert(0, '..')#adversarial-attacks-pytorch/')
	from square_attack import square_attack

	for k, v in config.items():
		if k.endswith('_step'):
			config[k] = eval(v)
		elif k == 'optimizer':
			config[k] = vars(torch.optim)[v]([p for p in m.parameters() if p.requires_grad], **config[k+'_config'])
			config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['scheduler_config'])
		elif k == 'adversarial' or k == 'attack':
			config[k] = eval(v)

	from datasets import load_dataset
	svhn_full = load_dataset('svhn', 'full_numbers')

	import numpy as np
	import torchvision.transforms.v2 as T
	from torchvision.ops import box_convert

	T_ = T.Compose([
		T.Resize((112, 112)),
		T.ToImageTensor(),
		T.ConvertImageDtype(),
	])

	def transforms(e):
		for d, x in zip(e['digits'], e['image']):
			d['bbox'] = d['bbox'] * np.tile(np.array((112, 112)) / x.size, 2)
			d['bbox'] = box_convert(torch.tensor(d['bbox']), 'xywh', 'xyxy')
		e['image'] = T_(e['image'])
		return e

	full_set = svhn_full.map(transforms, batched = True)
	full_set.set_format('torch')

	def collate(e):
		images, targets = [], []
		for d in e:
			images.append(d['image'])
			targets.append({
				'boxes':d['digits']['bbox'].float(),
				'labels':d['digits']['label'].fill_(1).long(), 
				'string':d['digits']['label'].long()
			})
		return images, targets

	train_loader = torch.utils.data.DataLoader(full_set['train'], batch_size = config['batch_size'], collate_fn = collate, num_workers = 6, shuffle = True)
	test_loader = torch.utils.data.DataLoader(full_set['test'], batch_size = config['batch_size'], collate_fn = collate, num_workers = 6)



	for epoch in range(20):
		if epoch > 0:
			train(m,
				train_loader = train_loader,
				epoch = epoch,
				writer = writer,
				**config
			)

		if epoch < 2:
			validate(m,
				val_loader = test_loader,
				epoch = epoch,
				writer = writer,
				**config
			)

		else:
			# m.load_state_dict(torch.load(f"checkpoints/Oct17_12-46-49_vm1_svhnfull_FasterRCNN_steps.d_step_{epoch:03}.pt"))
			validate(m,
				val_loader = test_loader,
				epoch = epoch,
				writer = writer,
				**config
			)

			# torch.save(m.state_dict(), "checkpoints/" + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

	print(m)

	writer.flush()
	writer.close()

if __name__ == "__main__":
	main(config)