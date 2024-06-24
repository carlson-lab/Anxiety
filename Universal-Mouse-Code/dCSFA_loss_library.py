import torch

class itakuraSaitoLoss(torch.nn.modules.loss._Loss):
    def __init__(self,reduction='mean'):
        super(itakuraSaitoLoss,self).__init__()
        self.reduction = reduction
        
    def forward(self,X_hat,X):
        EPS = 1e-9
        n_samp = X.shape[0]
        n_feat = X.shape[1]
        X_flattened = X.view(-1,1)
        X_hat_flattened = X_hat.view(-1,1)
        
        eps_idx = X_flattened > EPS
        X_hat_flattened = X_hat_flattened[eps_idx].view(-1,1)
        X_flattened = X_flattened[eps_idx].view(-1,1)
        
        X_hat_flattened[X_hat_flattened==0] = EPS
        div = X_flattened / X_hat_flattened
        res = torch.sum(div) - torch.sum(torch.log(div)) - n_samp*n_feat

        if self.reduction == 'mean':
            res /= (n_samp*n_feat)
        elif self.reduction == 'batchmean':
            res /= n_samp
        #print(res)
        return res