# loss_functions.py

import torch
import torch.nn.functional as F

def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss = torch.mean(
        label * torch.pow(euclidean_distance, 2) +
        (1 - label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    )
    return loss

def triplet_loss(anchor, positive, negative, margin=1.0):
    positive_distance = F.pairwise_distance(anchor, positive)
    negative_distance = F.pairwise_distance(anchor, negative)
    loss = torch.mean(F.relu(positive_distance - negative_distance + margin))
    return loss
