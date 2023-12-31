import torch

import steps
from load_svhn import d_test_loader
from utils import iterate


from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '../adversarial-attacks-pytorch/')
sys.path.append('..')
import torchattacks


config = {
	'dataset':'svhnfull',
	'training_step':'transcribe_eval_step',
	# 'checkpoint':'checkpoints/Sep04_18-28-00_Theseus_svhnfull_FasterRCNN_ordinary_step_007.pt',
	# 'initialization':'xavier_init',
	'batch_size':256,
	# 'optimizer':'SGD',
	# 'optimizer_config':{
	# 	'lr':0.005,
	# 	'momentum':0.9,
	# 	'weight_decay':0.0005,
	# },
	# 'scheduler':'StepLR',
	# 'scheduler_config':{
	# 	'step_size':3,
	# 	'gamma':0.1
	# },
	'attack':'Square_',
	'attack_config':{
		'eps':8/255,
		'loss':'sim',
		'n_queries':20
		# 'alpha':0.2,
		# 'steps':40,
		# 'random_start':True,
	},
	'device':'cuda' if torch.cuda.is_available() else 'cpu',
	'validation_step':'transcribe_eval_step',
	'attacked_step':'attacked_transcribe_step'
}

from transcriber import Transcriber
m  = Transcriber().to(config['device'])

if 'checkpoint' in config:
	m.load_state_dict({k:v for k,v in torch.load(config['checkpoint']).items() if k in m.state_dict()})


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
		
test_loader = d_test_loader(config['batch_size'])

# for epoch in range(10):
import os
epoch = 0
for ckpt_d in sorted(os.listdir('checkpoints_d'))[::8]:

	m.detector.load_state_dict({k:v for k,v in torch.load(f'checkpoints_d/' + ckpt_d).items() if k in m.detector.state_dict()})
	
	for ckpt_c in sorted(os.listdir('checkpoints_c'))[::16]:
		m.classifier.load_state_dict({k:v for k,v in torch.load('checkpoints_c/' + ckpt_c).items() if k in m.classifier.state_dict()})
		print(epoch, ckpt_d, ckpt_c, '\n', sep = '\n')

		if epoch > 5000:
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

		epoch += 1

    # torch.save(m.state_dict(), "checkpoints/" + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(m)

# outputs = iterate.predict(m,
# 	steps.predict_step,
# 	val_loader = val_loader,
# 	**config
# )

# print(outputs.keys(), outputs['predictions'])
writer.flush()
writer.close()