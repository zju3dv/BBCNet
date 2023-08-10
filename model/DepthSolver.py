import torch
import torch.nn as nn

# TODO:
# 1. add adaptor
class DepthSolver(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prior = 1e-5

    # @torch.autocast(dtype=torch.float32,device_type='cuda')
    def forward(self,sparse_depth,x):

        B,C,H,W = x.shape
        diag = self.prior * torch.eye(C,C,device = x.device) # for numberical stablity
        fb = self.prior * torch.ones(C,1,device = x.device) # for numberical stablity

        sd_flatten = sparse_depth.view(B,H*W,1) # B X HW X 1
        x_flatten = x.view(B,C,H*W).permute(0,2,1).contiguous() # B X HW X C

        # N = 200 if self.training else 125
        N = (sd_flatten > 0).sum(1).max().item()

        # sorted_sd,sorted_indices = torch.sort(sd_flatten,dim=1,descending=True)
        sorted_sd,sorted_indices = torch.topk(sd_flatten,dim=1,sorted=False,k=N)
        sorted_indices = sorted_indices.detach()
        sd = sorted_sd.contiguous().detach() # B X N X 1
        f_sparse_sorted = torch.gather(x_flatten,dim=1,index=sorted_indices.repeat(1,1,C))
        f_sparse = f_sparse_sorted * (sd > 0).expand_as(f_sparse_sorted).detach()

        FTF = torch.bmm(f_sparse.mT, f_sparse) # B X C X C
        FTF = FTF + diag
        eigval = torch.linalg.eigvalsh(FTF)
        condition_number = eigval[:,-1] / eigval[:,0]
        condition_number = torch.log(condition_number + 1) #* 16 / x.shape[1] #* x.shape[1] / 16
        # print(FT.shape,sd.shape)
        FTD = torch.bmm(f_sparse.mT,sd)
        FTD = FTD + fb
        # weight = torch.linalg.solve(FTF,FTD)
        weight = torch.linalg.solve(FTF,FTD)
        # print(FTF.shape,weight.shape,eigval.shape,FTD.shape,f_sparse.shape)
        final_depth = torch.bmm(weight.view(B,1,weight.shape[1]), x_flatten.mT)
        # print(final_depth.shape)
        final_depth = final_depth.view(B,1,H,W)
        cond_sum = condition_number.mean() + (weight - 1).abs().mean()
        # print(f_sparse.mean(),x_flatten.mean(),(sd > 0).sum(dim=(-1,-2)).min().item())
        return final_depth,cond_sum


