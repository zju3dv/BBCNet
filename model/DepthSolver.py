import torch

def solve(sparse_depth,x):
    cond_sum = torch.zeros_like(x[0,0,0,0])
    out = []
    B,C,H,W = x.shape
    min_solvable_number = 20
    for i in range(sparse_depth.shape[0]):
        mask = sparse_depth[i] > 0 # H,W
        mask = mask.detach()
        if(mask.sum() < min_solvable_number):
            out.append(x[i].sum(dim=0).view(1,1,H,W))
            continue
        sd = sparse_depth[i][mask] # HW
            
        mask = mask[0]
        f_sparse_orig = x[i].permute(1,2,0)
        f_sparse = f_sparse_orig[mask]
        # f_one = torch.ones_like(f_sparse[:,0:1],device=f_sparse.device)
        # f_sparse = torch.cat([f_sparse,f_one],dim=1)
        prior = 1e-5
        FTF = f_sparse.T @ f_sparse

        FTF = FTF + torch.eye(FTF.shape[0],device=FTF.device) * prior
        eigval = torch.linalg.eigvals(FTF).real.abs()
        condition_number = eigval.max() / eigval.min()
        condition_number = torch.log(condition_number + 1) #* 16 / x.shape[1] #* x.shape[1] / 16
        # if not self.training:
        #     if(condition_number >= 130):#50 for 4 bases
        #         prior =10 
        #         FTF = f_sparse.T @ f_sparse + torch.eye(FTF.shape[0],device=FTF.device) * prior

        FTD = f_sparse.T @ sd 
        FTD = FTD.view(FTD.shape[0],1)
        FTD = FTD + torch.ones_like(FTD) * prior
        weight = torch.linalg.solve(FTF,FTD)

        predict = x[i].view(x.shape[1],x.shape[2] * x.shape[3])

        final_depth = weight.view(1,weight.shape[0]) @ predict   

        final_depth = final_depth.view(1,1,x.shape[2],x.shape[3]) 
        out.append(final_depth)
        cond_sum = cond_sum + condition_number / B 

    x = torch.cat(out,dim=0)      
    return x,cond_sum

