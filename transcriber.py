import torch
import torch.nn as nn
import torchvision

from transformers.utils import ModelOutput


class Transcriber(nn.Module):
    def __init__(self, detector_weight = None, classifier_weight = None):
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(num_classes = 2)
        self.classifier = torch.hub.load('cat-claws/nn', 'resnet_cifar', pretrained= False, num_classes=10, blocks=14, bottleneck=False)
        if detector_weight:
            self.detector.load_state_dict(detector_weight)
        if classifier_weight:
            self.classifier.load_state_dict(classifier_weight)
    
    def forward(self, x):
        with torch.no_grad():
            outputs = [{k: v.detach().to('cpu') for k, v in t.items()} for t in self.detector(x)]

            length, cropped = [], []
            for image, p in zip(x, outputs):
                ind = p['boxes'][:, 0].argsort()[p['scores'] > .5]
                p['boxes'] = p['boxes'][ind]
                p['scores'] = p['scores'][ind]
                p['labels'] = p['labels'][ind]

                cropped_ = []
                for box in p['boxes']:
                    x1, y1, x2, y2 = map(int, box)
                    cropped_.append(torchvision.transforms.Resize((32, 32), antialias=True)(image[:, y1:y2, x1:x2]))
                cropped.extend(cropped_)
                length.append(len(p['boxes']))

            logits = self.classifier(torch.stack(cropped)).detach()
            predictions = logits.argmax(-1)

            return ModelOutput(
                logits = torch.split(logits, length, dim = 0),
                predictions = torch.split(predictions, length, dim = 0),
                **{k: [d[k] for d in outputs] for k in outputs[0]},
            )