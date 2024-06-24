"""
dCSFA using a recurrent neural network encoder
for learning of multiple predictive networks
"""

__date__ = "December 2021"

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC
from tqdm import tqdm
import os
from sklearn.base import BaseEstimator
import numpy as np
#from dCSFA_loss_library import itakuraSaitoLoss

if float(torch.__version__[:3]) >= 1.9:
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

FLOAT = torch.float32
INT = torch.int64
MAX_LABEL = 1000
EPSILON = 1e-6
FIT_ATTRIBUTES = ['classes_']

class dCSFA_UMC(nn.Module,BaseEstimator):

    def __init__(self,n_networks=1,sup_strength=1.0,z_dim=32,weight_reg=0.0,
                z1_recon_strength=.01,optimizer='AdamW',recon_l='IS',
                nonnegative=True,variational=True,kl_factor=1.0,
                n_iter=50000,lr=1e-3,batch_size=356,beta=0.5,device='auto'):
        super(dCSFA_UMC,self).__init__()
        assert isinstance(n_networks,(int))
        assert n_networks <= z_dim
        self.n_networks = n_networks
        assert kl_factor >= 0.0, f"{kl_factor} < 0"
        # Set parameters.
        assert isinstance(sup_strength, (int, float))
        assert sup_strength >= 0.0
        self.sup_strength = float(sup_strength)
        assert isinstance(z_dim, int)
        assert z_dim >= 1
        self.z_dim = z_dim
        assert isinstance(weight_reg, (int, float))
        assert weight_reg >= 0.0
        self.weight_reg = float(weight_reg)
        assert isinstance(nonnegative, bool)
        self.nonnegative = nonnegative
        assert isinstance(variational, bool)
        self.variational = variational
        assert isinstance(kl_factor, (int, float))
        assert kl_factor >= 0.0
        self.kl_factor = float(kl_factor)
        assert isinstance(n_iter, int)
        assert n_iter > 0
        self.n_iter = n_iter
        assert isinstance(lr, (int, float))
        assert lr > 0.0
        self.lr = float(lr)
        assert isinstance(batch_size, int)
        assert batch_size > 0
        self.batch_size = batch_size
        assert isinstance(beta, (int, float))
        assert beta >= 0.0 and beta <= 1.0
        self.beta = float(beta)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.recon_l = recon_l
        self.device = device
        self.classes_ = None       
        self.Recon_Loss = self._get_recon_loss()
        self.Prediction_Loss = nn.BCELoss()
        self.z1_recon_strength = z1_recon_strength

    def _initialize(self,n_freqs,n_features):
        self.n_freqs = n_freqs
        self.n_features = n_features
        self.rnn = nn.RNN(n_freqs,self.z_dim*2)
        self.fc_1 = nn.Linear(self.z_dim*2,self.z_dim*2)
        self.fc_to_pos = nn.Sequential(nn.Linear(self.z_dim,self.z_dim),nn.Softplus())
        self.lr_list = nn.ModuleList([nn.Sequential(nn.Linear(1,1),nn.Sigmoid())
                                         for network in range(self.n_networks)])
        self.W_nmf = nn.Parameter(torch.randn(self.z_dim,n_features*n_freqs))
        self.to(self.device)
    def _get_recon_loss(self):
        '''
        Returns the reconstruction loss corresponding to the recon_l string. Defaults to MSE if an incorrect
        string is sent.
        
        Note: If you are using directed-spectrum features, you should use IS loss.
        '''
        if self.recon_l == 'MSE':
            return nn.MSELoss()
        elif self.recon_l =='IS':
            return itakuraSaitoLoss()
        else:
            print("No loss or unsupported loss specified - using MSE")
            return nn.MSELoss()
        
    def forward(self,X):
        _,h_m = self.rnn(X)
        h = nn.Tanh()(self.fc_1(h_m)).squeeze()
        mean, log_var = torch.tensor_split(h,2,dim=1)
        latent_dist = Normal(mean,torch.exp(log_var))
        z = latent_dist.rsample()
        z_pos = self.fc_to_pos(z)
        X_recon = z_pos @ nn.ReLU()(self.W_nmf)
        y_pred_list = [self.lr_list[network](z_pos[:,network].view(-1,1))
                                for network in range(self.n_networks)]
                                
        X_recon_z1_list = [z_pos[:,network].view(-1,1)@nn.ReLU()(self.W_nmf[network,:].view(1,-1)) 
                                for network in range(self.n_networks)]
        
        return X_recon, X_recon_z1_list, y_pred_list, z_pos

    def fit(self,X,y,y_network=None,M=None,n_epochs=100,verbose=False):
        #Zero out the training loss histories
        self.training_hist = []
        self.recon_hist = []
        self.recon_z1_hist = []
        self.pred_hist = []

        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)  

        if y_network is None:
            print("Learning a single network")
            y_network = torch.zeros_like(y).to(self.device)
        else:
            assert len(np.unique(y_network)) == self.n_networks
            print(f"Learning {len(np.unique(y_network))} networks")
            y_network = torch.Tensor(y_network).to(self.device)

        if M is None:
            M = torch.ones_like(X).to(self.device)
        else:
            assert M.shape == X.shape
            M = torch.Tensor(M).to(self.device)

        dset = TensorDataset(X,M,y,y_network)
        loader = DataLoader(dset,batch_size=self.batch_size,shuffle=True)
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)

        if verbose: print("Training the Model")

        for epoch in range(n_epochs):
            #Initialize running epoch losses at each epoch
            epoch_loss = 0.0
            recon_e_loss = 0.0
            recon_z1_loss = 0.0
            pred_e_loss = 0.0

            for X_batch, M_batch, y_batch,y_network_batch in loader:
                X_batch_3d = X_batch.view(-1,self.n_features,self.n_freqs)
                X_batch_3d = torch.swapaxes(X_batch_3d,0,1)
                optimizer.zero_grad()
                X_recon, X_recon_z1_list, y_pred_list, _ = self.forward(X_batch_3d)

                #print([X_recon_z1.shape for X_recon_z1 in X_recon_z1_list])
                #print(X_batch.shape)
                #print(X_recon.shape)
                l_recon = self.Recon_Loss(X_recon*M_batch,X_batch)

                l_recon_z1 = 0
                for X_recon_z1 in X_recon_z1_list:
                    l_recon_z1 += self.Recon_Loss(X_recon_z1*M_batch,X_batch)
                    
                l_pred = 0
                for network, y_pred in enumerate(y_pred_list):
                    #print(y_pred_list)
                    #print(y_pred.shape,y_batch.shape)
                    l_pred += self.Prediction_Loss(y_pred[y_network_batch==network],y_batch[y_network_batch==network].view(-1,1))

                #l_recon_z1 = self.z1_recon_strength*torch.sum([self.Recon_Loss(X_recon_z1*M_batch,X_batch) 
                #                                                for X_recon_z1 in X_recon_z1_list])
                #l_pred = self.sup_strength*torch.sum([self.Prediction_Loss(y_pred[y_network_batch==network],
                #                                                            y_batch[y_network_batch==network])
                #                                        for network, y_pred in enumerate(y_pred_list)])
                
                loss = l_recon + l_recon_z1 + l_pred
                loss.backward()
                optimizer.step()

                #Add to the epoch losses
                epoch_loss += loss.item()
                recon_e_loss += l_recon.item()
                recon_z1_loss += l_recon_z1.item()
                pred_e_loss += l_pred.item()
            #Append the training histories
            self.training_hist.append(epoch_loss)
            self.recon_hist.append(recon_e_loss)
            self.recon_z1_hist.append(recon_z1_loss)
            self.pred_hist.append(pred_e_loss)
            if verbose:
                print(f'Epoch: {epoch}, loss: {epoch_loss},recon: {recon_e_loss} \n recon_z1: {recon_z1_loss} prediction: {pred_e_loss}')
            
            self.components = nn.ReLU()(self.W_nmf)
                        







