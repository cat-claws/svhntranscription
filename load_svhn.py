import torch

# object detection

from datasets import load_dataset
svhn_full = load_dataset('svhn', 'full_numbers')

import numpy as np
import torchvision.transforms.v2 as T
from torchvision.ops import box_convert

def transforms(e):
    boxes, labels = [], []
    for image, digits in zip(e['image'], e['digits']):
        boxes.append(
            box_convert(torch.from_numpy(digits['bbox'] * np.tile(np.array((112, 112)) / image.size, 2)), 'xywh', 'xyxy')
        )
        labels.append(torch.tensor(digits['label']).fill_(1).long())

    images = T.ToTensor()(T.Resize((112, 112))(e['image']))

    return {'image': images, 'boxes': boxes, 'labels': labels}

def collate(e):
    # e = {k: [d[k] for d in e] for k in e[0]}
    images = torch.stack([d.pop('image') for d in e])
    return images, e

d_train_loader = lambda x: torch.utils.data.DataLoader(svhn_full['train'].with_transform(transforms), batch_size=x, collate_fn = collate, num_workers = 2, shuffle = True)
d_test_loader = lambda x: torch.utils.data.DataLoader(svhn_full['test'].with_transform(transforms), batch_size=x, collate_fn = collate, num_workers = 2)
