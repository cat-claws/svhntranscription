import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# object detection

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



def collate(e):
    # {k: [d[k] for d in e] for k in e[0]}
    images = [d.pop('image') for d in e]
    targets = [
        {
            'boxes':d['digits']['bbox'].float(),
            'labels':torch.tensor(d['digits']['label']).fill_(1).long(), 
            'string':torch.tensor(d['digits']['label']).long()
        } for d in e
    ]
    return images, targets

d_train_loader = lambda x: torch.utils.data.DataLoader(
    svhn_full['train'].with_transform(transforms),
    batch_size = x,
    collate_fn = collate,
    num_workers = 4,
    shuffle = True
)

d_test_loader = lambda x: torch.utils.data.DataLoader(
    svhn_full['test'].with_transform(transforms),
    batch_size = x,
    collate_fn = collate,
    num_workers = 4
)


from torchvision import datasets

T_1 = T.Compose([
    T.ToImageTensor(),
    T.ConvertImageDtype(),
    ])

T_2 = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T_1
    ])

c_train_loader = lambda x: torch.utils.data.DataLoader(
    datasets.SVHN('SVHN', download=True, split = 'train', transform=T_2),
    batch_size=x,
    num_workers = 4
)

c_test_loader = lambda x: torch.utils.data.DataLoader(
    datasets.SVHN('SVHN', download=True, split = 'test', transform=T_1),
    batch_size=x,
    num_workers = 4
)