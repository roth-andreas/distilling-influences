import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.utils import softmax, to_dense_adj, subgraph, negative_sampling, add_self_loops


def kd_criterion(logits, labels, teacher_logits, criterion, alpha=0.5, T=1):
    """Logit-based KD, [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)
    """
    loss_cls = criterion(logits, labels)

    loss_kd = criterion(logits / T, teacher_logits / T)
    
    loss = loss_kd* (alpha* T* T) + loss_cls* (1-alpha)

    return loss, loss_cls, loss_kd


def nce_criterion(logits, labels, feat, teacher_feat, criterion, beta=0.5, nce_T=0.075, max_samples=8192):
    """Graph Contrastive Representation Distillation, [Joshi et al., TNNLS 2022](https://arxiv.org/abs/2111.04964)
    """
    loss_cls = criterion(logits, labels)

    if max_samples < feat.shape[0]:
        sampled_inds = np.random.choice(feat.shape[0], max_samples, replace=False)
        feat = feat[sampled_inds]
        teacher_feat = teacher_feat[sampled_inds]
    
    feat = F.normalize(feat, p=2, dim=-1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)

    nce_logits = torch.mm(feat, teacher_feat.transpose(0, 1))
    nce_labels = torch.arange(feat.shape[0]).to(feat.device)

    loss_nce = F.cross_entropy(nce_logits/ nce_T, nce_labels)
    
    loss = loss_cls + beta* loss_nce

    return loss, loss_cls, loss_nce
