import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from model_d import detector
import steps
from load_svhn import c_test_loader, T_1
from torchvision import datasets

c_train_loader = lambda x: torch.utils.data.DataLoader(
    torch.utils.data.random_split(datasets.SVHN('SVHN', download=True, split = 'test', transform=T_1), [1, 10])[0],
    batch_size=x,
    # collate_fn = collate,
    num_workers = 4
)

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
	'batch_size':128,
	'optimizer':'SGD',
	'optimizer_config':{
		'lr':0.0005,
		'momentum':0.9,
		'weight_decay':0.0005,
	},
	'scheduler':'ConstantLR',
	'scheduler_config':{
		# 'step_size':300,
		# 'gamma':0.1
	},
	'attack':'Square_',
	'attack_config':{
		'eps':8/255,
		'loss':'iou_',
		'n_queries':5000
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
		
train_loader = d_train_loader(config['batch_size'])
test_loader = d_test_loader(config['batch_size'])

import os
checkpoints = ['checkpoints2/' + x for x in os.listdir('checkpoints2') if x.startswith('S')]

import random

for epoch, c in enumerate(checkpoints):
    m.load_state_dict(torch.load(c))

    for j in range(200):
        x = random.randint(1, 99)
        train_loader = torch.utils.data.DataLoader(svhn_full['test'].with_transform(transforms).shard(100, x), batch_size=config['batch_size'], collate_fn = collate, num_workers = 4)

        iterate.train(m,
            train_loader = train_loader,
            epoch = epoch,
            writer = writer,
            **config
        )

        # m.load_state_dict({k:v for k,v in torch.load(f'checkpoints/Sep01_17-16-53_Theseus_svhnfull_FasterRCNN_ordinary_step_{epoch:03}.pt').items() if k in m.state_dict()})

        m.eval()
        outputs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                output = config['validation_step'](m, batch, batch_idx, **config)
                outputs.append({k:v.detach().cpu() for k, v in output.items()})

        outputs = {k: sum([dic[k] for dic in outputs]) / len(test_loader.dataset) for k in outputs[0]}
        print(c, outputs['correct'].item())
        if outputs['correct'] > 0.948:
            if outputs['correct'] < 0.949:
                torch.save(m.state_dict(), "checkpoints2_/" + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

            break

        if outputs['correct'] < 0.8:
            break

print(m)


writer.flush()
writer.close()