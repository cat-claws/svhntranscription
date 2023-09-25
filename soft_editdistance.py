import editdistance

def similarity_from_edit_distance(s1, s2):
    edit_dist = editdistance.eval(s1, s2)
    max_dist = max(len(s1), len(s2))
    return [edit_dist, 1 - edit_dist / max_dist]

import torch
import torch.nn.functional as F

def soft_similarity_from_edit_distance(s1_probs, s2):
    device = s1_probs.device
    m, n = s1_probs.shape[0], len(s2)
    
    # Convert s2 to one-hot representation using PyTorch
    s2_indices = torch.tensor([int(char) for char in s2], dtype=torch.int64).to(device)
    s2_onehot = F.one_hot(s2_indices, num_classes=10).float()

    # Compute the soft replace costs
    replace_cost = 1.0 - torch.mm(s1_probs, s2_onehot.t())

    # Initialize the DP table
    dp = torch.zeros(m+1, n+1).to(device)
    dp[:, 0] = torch.arange(m+1)
    dp[0, :] = torch.arange(n+1)

    # Compute the soft edit distance
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i, j] = torch.min(
                torch.tensor([dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + replace_cost[i-1, j-1]])
            )
    
    return 1 - (dp[m, n] / len(s2))

# Example
"""
s1_probs_tensor = torch.tensor([
    [0.1, 0.1, 0.1, 0.6, 0.1, 0, 0, 0, 0, 0],
    [0.1, 0.1, 0.6, 0.1, 0.1, 0, 0, 0, 0, 0],
]).cuda()  # Assuming you have a CUDA-enabled GPU

s2 = "34"

similarity_score = soft_similarity_from_edit_distance(s1_probs_tensor, s2)
print("Similarity score:", similarity_score.item())

# Convert to back to distance
soft_distance = (1 - similarity_score.item() )* len(s2)
print("Soft edit distance:", soft_distance)
"""