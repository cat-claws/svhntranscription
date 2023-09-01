import torch

# object detection

from datasets import load_dataset
svhn_full = load_dataset('svhn', 'full_numbers')

import numpy as np
import torchvision.transforms.v2 as T
from torchvision.ops import box_convert


def transforms(e):
    e['image'] = T.ToTensor()(e['image'])
    for d in e['digits']:
        d['bbox'] = box_convert(torch.tensor(d['bbox']), 'xywh', 'xyxy')
    return e
    # box_convert(torch.from_numpy(digits['bbox'] * np.tile(np.array((112, 112)) / image.size, 2)), 'xywh', 'xyxy')



def collate(e):
    # {k: [d[k] for d in e] for k in e[0]}
    images = [d.pop('image') for d in e]
    targets = [
        {'boxes':d['digits']['bbox'].float(), 'labels':torch.tensor(d['digits']['label']).fill_(1).long()} for d in e
    ]
    return images, targets

d_train_loader = lambda x: torch.utils.data.DataLoader(svhn_full['train'].with_transform(transforms), batch_size=x, collate_fn = collate, num_workers = 4, shuffle = True)
d_test_loader = lambda x: torch.utils.data.DataLoader(svhn_full['test'].with_transform(transforms), batch_size=x, collate_fn = collate, num_workers = 4)
