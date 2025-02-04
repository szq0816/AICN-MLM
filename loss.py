import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from torch.nn.functional import normalize
import math
from metric import  MMD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.autograd as autograd


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        # self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        # self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature_InfoNCE(self, h_i, h_j, batch_size=256):
        self.batch_size = batch_size

        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss




class Proto_Align_Loss(nn.Module):
    def __init__(self):
        super(Proto_Align_Loss, self).__init__()

    def forward(self, gt, P):
        mse = nn.MSELoss()
        Loss1 = mse(gt, P)

        return Loss1





class Proto_Align_Loss1(nn.Module):
    def __init__(self):
        super(Proto_Align_Loss1, self).__init__()
        self.eps = 1e-8

    def forward(self, gt, P):
        P = P.clamp(self.eps, 1.0 - self.eps)
        gt = gt.clamp(self.eps, 1.0 - self.eps)

        M = 0.5 * (gt + P)
        Loss1 = 0.5 * F.kl_div(gt.log(), M, reduction='batchmean') \
                + 0.5 * F.kl_div(P.log(), M, reduction='batchmean')

        return Loss1




def hard_sample_aware_infoNCE(S, M, pos_neg_weight, pos_weight, node_num):
    pos_neg = M * torch.exp(S * pos_neg_weight)
    pos = torch.cat([torch.diag(S, node_num), torch.diag(S, -node_num)], dim=0)
    pos = torch.exp(pos * pos_weight)
    neg = (torch.sum(pos_neg, dim=1) - pos)
    infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * node_num)
    return infoNEC







def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3,
                epsilon=1e-8):
    # """
    # Similarity Distribution Matching
    # """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:

        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask


    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    t2i_pred = F.softmax(text_proj_image, dim=1)

    i2t_MMD = MMD(labels_distribute, i2t_pred)
    t2i_MMD = MMD(labels_distribute, t2i_pred)

    loss = (i2t_MMD + t2i_MMD) / 2

    return loss





