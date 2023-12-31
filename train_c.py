import torch

import steps
from load_svhn import c_train_loader, c_test_loader
from utils import iterate


from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '../adversarial-attacks-pytorch/')
sys.path.append('..')
import torchattacks


config = {
	'dataset':'svhn',
	'training_step':'c_step',
	# 'checkpoint':'checkpoints/Sep04_18-28-00_Theseus_svhnfull_FasterRCNN_ordinary_step_007.pt',
	# 'initialization':'xavier_init',
	'batch_size':32,
	'optimizer':'Adadelta',
	'optimizer_config':{
		# 'lr':0.005,
		# 'momentum':0.9,
		# 'weight_decay':0.0005,
	},
	'scheduler':'StepLR',
	'scheduler_config':{
		'step_size':100,
		'gamma':0.1
	},
	'attack':'Square_',
	'attack_config':{
		'eps':8/255,
		'loss':'ce',
		'n_queries':10
		# 'alpha':0.2,
		# 'steps':40,
		# 'random_start':True,
	},
	'device':'cuda' if torch.cuda.is_available() else 'cpu',
	'validation_step':'c_step',
	'attacked_step':'attacked_c_step'
}

m = torch.hub.load('cat-claws/nn', 'resnet_cifar', pretrained= False, num_classes=10, blocks=14, bottleneck=False).to(config['device'])

if 'checkpoint' in config:
	m.load_state_dict({k:v for k,v in torch.load(config['checkpoint']).items() if k in m.state_dict()})
# if 'initialization' in config:
# 	m.apply(vars(misc)[config['initialization']])

writer = SummaryWriter(comment = f"_{config['dataset']}_{m._get_name()}_{config['training_step']}", flush_secs=10)

import json
with open("checkpoints/configs.json", 'a') as f:
	f.write(json.dumps({**{'run':writer.log_dir.split('/')[-1]}, **config}) + '\n')
	print(json.dumps(config, indent=4))

for k, v in config.items():
	if k.endswith('_step'):
		config[k] = vars(steps)[v]
	elif k == 'optimizer':
		config[k] = vars(torch.optim)[v]([p for p in m.parameters() if p.requires_grad], **config[k+'_config'])
		config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['scheduler_config'])
	elif k == 'adversarial' or k == 'attack':
		config[k] = vars(torchattacks)[v](m, **config[k+'_config'])
		
train_loader = c_train_loader(config['batch_size'])
test_loader = c_test_loader(config['batch_size'])

import os
for epoch, ckpt in enumerate(sorted(os.listdir('checkpoints_c'))):
# for epoch in range(100):
	if epoch < 0:
		iterate.train(m,
			train_loader = train_loader,
			epoch = epoch,
			writer = writer,
			**config
		)

	# m.load_state_dict({k:v for k,v in torch.load(f'checkpoints_/Sep05_23-18-24_Theseus_svhnfull_FasterRCNN_ordinary_step_293.pt').items() if k in m.state_dict()})
	m.load_state_dict({k:v for k,v in torch.load('checkpoints_c/' + ckpt).items() if k in m.state_dict()})
	print(epoch, ckpt)

	if epoch < 0:
		iterate.validate(m,
			val_loader = test_loader,
			epoch = epoch,
			writer = writer,
			**config
		)

	else:
		iterate.attack(m,
			val_loader = test_loader,
			epoch = epoch,
			writer = writer,
			atk = config['attack'],
			**config
		)

	torch.save(m.state_dict(), "checkpoints/" + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(m)

# outputs = iterate.predict(m,
# 	steps.predict_step,
# 	val_loader = val_loader,
# 	**config
# )

# print(outputs.keys(), outputs['predictions'])
writer.flush()
writer.close()