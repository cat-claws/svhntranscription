import torch

# object detection

from datasets import load_dataset
svhn_full = load_dataset('svhn', 'full_numbers')

import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose(
    [A.Resize(height=112, width=112, always_apply=True), ToTensorV2()],
    bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))


from torchvision.ops import box_convert
import numpy as np

def transforms(e):
    images, bboxes, labels = [], [], []
    for image, digits in zip(e['image'], e['digits']):      
        out = transform(
            image=np.array(image),
            bboxes=digits['bbox'],
            class_labels=digits['label']
        )
        images.append(out['image'])
        bboxes.append(box_convert(torch.tensor(out['bboxes']), 'xywh', 'xyxy'))
        labels.append(out['class_labels'])

    return {'image': images, 'bbox': bboxes, 'label': labels}

def collate(e):
    e = {k: [d[k] for d in e] for k in e[0]}
    e['image'] = torch.stack(e['image'])
    return e


d_train_loader = torch.utils.data.DataLoader(svhn_full['train'].with_transform(transforms), batch_size=32, collate_fn = collate)
d_test_loader = torch.utils.data.DataLoader(svhn_full['test'].with_transform(transforms), batch_size=32, collate_fn = collate)
