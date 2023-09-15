import torch

def compute_single_class_mAP_from_ious(ious, pred_scores, iou_threshold=0.5):
    # Sort the IoU matrix based on prediction scores
    sorted_indices = torch.argsort(pred_scores, descending=True)
    ious = ious[sorted_indices]

    # Identify the best ground truth match for each prediction
    max_ious, best_gt_indices = ious.max(dim=1)

    # Identify True Positives and False Positives
    tp_flags = max_ious > iou_threshold
    used = torch.zeros(ious.shape[1], dtype=torch.bool) # ious.shape[1] gives number of ground truths
    
    for i in range(tp_flags.shape[0]):
        if tp_flags[i] and not used[best_gt_indices[i]]:
            used[best_gt_indices[i]] = True
        else:
            tp_flags[i] = False

    tp_cumsum = torch.cumsum(tp_flags.float(), dim=0)
    fp_cumsum = torch.cumsum(1 - tp_flags.float(), dim=0)

    recalls = tp_cumsum / ious.shape[1]
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP
    recalls = torch.cat([torch.tensor([0.]), recalls, torch.tensor([1.])])
    precisions = torch.cat([torch.tensor([0.]), precisions, torch.tensor([0.])])

    # Correct precision for the recall value changes
    for i in range(precisions.size(0) - 1, 0, -1):
        precisions[i - 1] = torch.max(precisions[i - 1], precisions[i])

    # Compute the AP over recall range
    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    mAP = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]).item()

    return mAP