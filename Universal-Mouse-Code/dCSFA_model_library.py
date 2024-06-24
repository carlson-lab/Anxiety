import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC
from tqdm import tqdm


class dCSFA_RNN(nn.Module):
    def __init__(self,n_freqs,n_components,n_total_features):

        super(dCSFA_RNN,self).__init__()
        self.n_freqs = n_freqs
        self.n_components = n_components
        self.n_total_features = n_total_features

        self.rnn = nn.RNN(self.n_freqs,self.n_components*2)
        self.fc_1 = nn.Linear(self.n_components*2,self.n_components*2)
        self.fc_to_pos = nn.Sequential(nn.Linear(n_components,n_components),nn.Softplus())
        self.lr = nn.Sequential(nn.Linear(1,1),nn.Sigmoid())

        self.W_nmf = nn.Parameter(torch.randn(n_components,n_total_features))

    def forward(self,X):
        output, h_m = self.rnn(X)
        h = nn.Tanh()(self.fc_1(h_m)).squeeze()
        mean, log_var = torch.tensor_split(h,2,dim=1)
        latent_dist = Normal(mean,torch.exp(log_var))
        z = latent_dist.rsample()
        z_pos = self.fc_to_pos(z)
        y_pred = self.lr(z_pos[:,0].view(-1,1))
        X_recon = z_pos @ nn.ReLU()(self.W_nmf)
        X_recon_z1 = z_pos[:,0].view(-1,1) @ nn.ReLU()(self.W_nmf[0,:].view(1,-1))
        return X_recon, X_recon_z1, y_pred, z_pos

class dCSFA_NMF_VAE(nn.Module):
    def __init__(self,n_components,dim_in,h=100):
        super(dCSFA_NMF_VAE,self).__init__()
        self.n_components = n_components
        self.dim_in = dim_in

        self.Encoder = nn.Sequential(nn.Linear(dim_in,h),
                                     nn.ELU(),
                                     nn.Linear(h,h),
                                     nn.ELU(),
                                     nn.Linear(h,n_components*2))

        self.fc_to_pos = nn.Sequential(nn.Linear(n_components,n_components),nn.Softplus())
        self.lr = nn.Sequential(nn.Linear(1,1),nn.Sigmoid())
        self.W_nmf = nn.Parameter(torch.randn(n_components,dim_in))

    def forward(self,X):
        h = self.Encoder(X)
        mean, log_var = torch.tensor_split(h,2,dim=1)
        latent_dist = Normal(mean,torch.exp(log_var))
        z = latent_dist.sample()
        z_pos = self.fc_to_pos(z)
        y_pred = self.lr(z_pos[:,0].view(-1,1))
        X_recon = z_pos @ nn.ReLU()(self.W_nmf)
        X_recon_z1 = z_pos[:,0].view(-1,1) @ nn.ReLU()(self.W_nmf[0,:].view(1,-1))
        return X_recon, X_recon_z1, y_pred, z_pos

class dCSFA_NMF(nn.Module):
    def __init__(self,n_components,dim_in,h=100):
        super(dCSFA_NMF,self).__init__()
        self.n_components = n_components
        self.dim_in = dim_in

        self.Encoder = nn.Sequential(nn.Linear(dim_in,n_components),nn.ReLU())
        self.fc_to_pos = nn.Sequential(nn.Linear(n_components,n_components),nn.Softplus())
        self.lr = nn.Sequential(nn.Linear(1,1),nn.Sigmoid())
        self.W_nmf = nn.Parameter(torch.randn(n_components,dim_in))

    def forward(self,X):
        z_pos = self.Encoder(X)
        y_pred = self.lr(z_pos[:,0].view(-1,1))
        X_recon = z_pos @ nn.Softplus()(self.W_nmf)
        X_recon_z1 = z_pos[:,0].view(-1,1) @ nn.Softplus()(self.W_nmf[0,:].view(1,-1))
        return X_recon, X_recon_z1, y_pred, z_pos

class dCSFA_NMF_RI(nn.Module):
    def __init__(self,n_components,dim_in,n_intercepts):
        super(dCSFA_NMF_RI,self).__init__()
        self.n_components = n_components
        self.dim_in = dim_in
        self.n_intercepts = n_intercepts

        self.Encoder = nn.Sequential(nn.Linear(dim_in,n_components),nn.ReLU())
        self.fc_to_pos = nn.Sequential(nn.Linear(n_components,n_components),nn.Softplus())

        self.Phi = nn.Parameter(torch.randn(1))
        self.B_list = nn.Parameter(torch.randn(self.n_intercepts,1))
        self.W_nmf = nn.Parameter(torch.randn(n_components,dim_in))

    def forward(self,X,intercept_mask=None):
        z_pos = self.Encoder(X)
        if intercept_mask:
            y_pred = nn.Sigmoid()(z_pos[:,0]*self.Phi + intercept_mask@self.B_list)
        else:
            y_pred = nn.Sigmoid()(z_pos[:,0]*self.Phi + torch.mean(self.B_list))
        X_recon = z_pos@nn.ReLU()(self.W_nmf)
        X_recon_z1 = z_pos[:,0].view(-1,1) @ nn.ReLU()(self.W_nmf[0,:].view(1,-1))

        return X_recon, X_recon_z1, y_pred, z_pos

class dCSFA_UMC(nn.Module):
    def __init__(self,n_freqs,n_components,n_total_features,n_networks=1):
        super(dCSFA_UMC,self).__init__()
        self.n_freqs = n_freqs
        self.n_components = n_components
        self.n_total_features = n_total_features
        assert n_networks <= n_components
        self.n_networks = n_networks
        
        self.rnn = nn.RNN(self.n_freqs,self.n_components*2)
        self.fc_1 = nn.Linear(self.n_components*2,self.n_components*2)
        self.fc_to_pos = nn.Sequential(nn.Linear(n_components,n_components),nn.Softplus())
        self.lr = nn.Sequential(nn.Linear(1,1),nn.Sigmoid())
        self.lr2 = nn.Sequential(nn.Linear(1,1),nn.Sigmoid())

        self.lr_list = nn.ModuleList([nn.Sequential(nn.Linear(1,1),nn.Sigmoid())
                                         for network in range(n_networks)])


        self.W_nmf = nn.Parameter(torch.randn(n_components,n_total_features))

    def forward(self,X):
        output, h_m = self.rnn(X)
        h = nn.Tanh()(self.fc_1(h_m)).squeeze()
        mean, log_var = torch.tensor_split(h,2,dim=1)
        latent_dist = Normal(mean,torch.exp(log_var))
        z = latent_dist.rsample()
        z_pos = self.fc_to_pos(z)
        #y_pred = self.lr(z_pos[:,0].view(-1,1))
        #X_recon_z1 = z_pos[:,0].view(-1,1) @ nn.ReLU()(self.W_nmf[0,:].view(1,-1))
        X_recon = z_pos @ nn.ReLU()(self.W_nmf)
        y_pred_list = [self.lr_list[network](z_pos[network,0].view(-1,1)) 
                                for network in range(self.n_networks)]
        X_recon_z1_list = [z_pos[network,:].view(-1,1)@nn.ReLU()(self.W_nmf[network,:].view(1,-1)) 
                                for network in range(self.n_networks)]
        
        
        return X_recon, X_recon_z1_list, y_pred_list, z_pos 