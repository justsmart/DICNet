import torch
import torch.nn as nn
import torch.nn.functional as F

# This code is inspired by https://github.com/SubmissionsIn/MFLVC

class Loss(nn.Module):
    def __init__(self, t, device):
        super(Loss, self).__init__()
        self.t = t
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward_contrast(self, v1, v2, we1, we2):
        mask_miss_inst = we1.mul(we2).bool() # mask the unavailable instances

        v1 = v1[mask_miss_inst]
        v2 = v2[mask_miss_inst]
        n = v1.size(0)
        N = 2 * n
        if n == 0:
            return 0
        v1 = F.normalize(v1, p=2, dim=1) #normalize two vectors
        v2 = F.normalize(v2, p=2, dim=1)
        mask = torch.ones((N, N)) # get mask
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()
        h = torch.cat((v1, v2), dim=0)
        sim_mat = torch.matmul(h, h.T) / self.t
        positive_pairs = torch.cat((torch.diag(sim_mat, n), torch.diag(sim_mat, -n)), dim=0).reshape(N, 1)
        negative_pairs = sim_mat[mask].reshape(N, -1)
        targets = torch.zeros(N).to(positive_pairs.device).long()
        logits = torch.cat((positive_pairs, negative_pairs), dim=1)
        loss = self.criterion(logits, targets)
        return loss/N

