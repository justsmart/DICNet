import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, t, device):
        super(Loss, self).__init__()
        self.t = t
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def contrast_loss(self, v1, v2, we1, we2):
        mask_miss_inst = we1.mul(we2).bool() # mask the unavailable instances

        v1 = v1[mask_miss_inst]
        v2 = v2[mask_miss_inst]
        n = v1.size(0)
        N = 2 * n
        if n == 0:
            return 0
        v1 = F.normalize(v1, p=2, dim=1) #normalize two vectors
        v2 = F.normalize(v2, p=2, dim=1)
        z = torch.cat((v1, v2), dim=0)
        similarity_mat = torch.matmul(z, z.T) / self.t
        similarity_mat = similarity_mat.fill_diagonal_(0)
        label = torch.cat((torch.tensor(range(n,N)),torch.tensor(range(0,n)))).to(self.device)
        # print(label)
        loss = self.criterion(similarity_mat, label)
        return loss/N
    def wmse_loss(self, input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        ret = torch.mean(ret)
        return ret
    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))).mul(inc_L_ind))
        
        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res