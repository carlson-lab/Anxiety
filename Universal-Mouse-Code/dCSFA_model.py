import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC
from tqdm import tqdm,trange
from dCSFA_model_library import *
from dCSFA_loss_library import itakuraSaitoLoss
import numpy as np
from torch.utils.data import WeightedRandomSampler

EPSILON=1e-8
class dCSFA_model(ABC):
    '''
    CLASS DESCRIPTION
    ----------------------------
    dCSFA_model is an abstraction class that can handle the multiple types of 
    factor models that we use in the collective for psychiatric neuroengineering.
    This class acts as a wrapper around one of the supported types of dCSFA and
    self contains training loops, transformation, and training loss storage. 

    The Objective for dCSFA_model is to allow for simple running of .fit()
    and .transform() methods that we frequently take advantage of in sklearn.

    This class is heavily dependent on the dCSFA_model_library which contains
    the pytorch modules for all of the dCSFA abstractions that work simply with
    this code set. A full list of which dCSFA model types are supported is listed
    in the doc-string for the init function of this class.
    '''
    def __init__(self,n_freqs,n_features,n_components,model_type='dCSFA_RNN',
                    optim_name='AdamW',batch_size=12,lr=1e-3,h=100,device='cpu',
                    recon_l='MSE',recon_z1_weight=1.0,sup_weight=1.0,
                    recon_weight=1.0,n_intercepts=None):
        '''
        Inputs
        -------------------------
        n_freqs - int - The number of frequencies per feature. In CPNE this value
        is often 56 due to strong power noise at 60Hz

        n_features - int - The cumulative sum of the number of unique features where
        CPNE typically uses power spectral density, coherence, granger causality,
        or linear directed spectrum features. Each feature should have a measurement
        at every frequency. For example, for the PSD-Hippocampus feature, we expect
        the power to be evaluated at frequencies 1-56 inclusively. If your data is
        shape N x M, where N is the number of samples, M = n_freqs*n_features

        n_components - int - The number of brain networks we wish to model. This is
        also the latent dimensionality we project into with the dCSFA encoder. The
        first component is what is used for supervision. Refining how we pick the
        number of components is a future area of research, however, for current
        practicality, we typically choose between 10-20.

        model_type - str - must be in the following set: {dCSFA_RNN,dCSFA_NMF_VAE,dCSFA_NMF}.
        dCSFA_RNN is used for missing data. dCSFA_NMF_VAE utilizes a variational autoencoder
        and a neural network encoder. dCSFA_NMF is the simplest using a linear encoder.

        optim_name - str - must be in the following set: {AdamW,Adam,SGD}. These outline
        different optimizers that can be used for training. Documentation can be found on
        the pytorch website. It should be noted that AdamW uses an L2 penalty on the model
        parameters by default, however, you will need to edit the code to add an L2 loss for
        the other optimizers.

        batch_size - int - The size of minibatches used for stochastic gradient descent based
        training. 

        lr - float - Step size for the optimizers

        h - int - size of the hidden layers for the neural network encoder for dCSFA_NMF_VAE

        device - str - Must be in the set {cpu,cuda:X}, where X is the gpu number if you are
        using multiple GPUs. Typically this will be cuda:0 for most use-cases of this model.

        recon_z1_weight - float - weight parameter for training the first component reconstruction
        term in training

        sup_weight - float - weight parameter for the supervision components of the training loss

        '''

        self.n_freqs = n_freqs
        self.n_features = n_features
        self.n_components = n_components
        self.model_type = model_type
        self.optim_name = optim_name
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.h = h
        self.w_r_z1 = recon_z1_weight
        self.w_sup = sup_weight
        self.recon_l = recon_l
        self.n_intercepts = n_intercepts
        self.recon_weight = recon_weight

        #Get the model and optimizer functions
        self.model = self._get_model()
        self.optimizer = self._get_optim()

        #Define the losses - note this will be abstracted in 
        #the future to allow for more flexibility in loss
        #choice
        self.Recon_Loss = self._get_recon_loss()
        self.Prediction_Loss = nn.BCELoss()

        #Training histories
        self.training_hist = []
        self.recon_hist = []
        self.recon_z1_hist = []
        self.pred_hist = []

    def _get_model(self):
        '''
        Returns the model corresponding to the model_type string. Raises an error if an incorrect string
        is sent
        '''
        if self.model_type == 'dCSFA_RNN':
            return dCSFA_RNN(self.n_freqs,self.n_components,self.n_freqs*self.n_features).to(self.device)
        elif self.model_type =='dCSFA_NMF_VAE':
            return dCSFA_NMF_VAE(self.n_components,self.n_freqs*self.n_features,self.h).to(self.device)
        elif self.model_type =='dCSFA_NMF':
            return dCSFA_NMF(self.n_components,self.n_freqs*self.n_features).to(self.device)
        elif self.model_type =='dCSFA_NMF_RI':
            return dCSFA_NMF_RI(self.n_components,self.n_freqs*self.n_features,self.n_intercepts).to(self.device)
        else:
            raise ValueError("Model type: ",self.model_type," not supported.")

    def _get_optim(self):
        '''
        Returns the optimizer corresponding to the optim_name string. Raises an error if an incorrect
        string is sent
        '''
        if self.optim_name == 'AdamW':
            return torch.optim.AdamW(self.model.parameters(),lr=self.lr)
        elif self.optim_name == 'Adam':
            return torch.optim.Adam(self.model.parameters(),lr=self.lr)
        elif self.optim_name == 'SGD':
            return torch.optim.SGD(self.model.parameters(),lr=self.lr)
        else:
            raise ValueError("Optimizer name: ",self.optim_name, " is not supported.")
            
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
        
    
    def fit(self,X,y,M=None, n_epochs=25,batch_size=12,verbose=False,print_rate=5,
            n_pretrain_epochs=0):
        '''
        FIT
        ----------------------------------------
        Trains the model

        Inputs
        ----------------------------------------
        X - np.array - N x M - Input data of shape N x M where N is the number of samples
        and M is n_freqs*n_features.

        y - np.array - N x 1 - Binary task labels.

        M - np.array - N x M - Binary observation mask corresponding to X. Each element is
        a 1 if it was observed, and a 0 if it was not. This input is required if using dCSFA-
        RNN

        intercept_mask - NxQ - Number of samples by number of mice binary mask to be used with
        the random intercept models.

        n_epochs - int - Number of training epochs where an epoch consists of the number of mini-
        batch training steps to approximately cover all of the training data once.

        batch_size - int - The number of samples used for each minibatch

        verbose - bool - indicates whether or not you want mid-training printouts. TQDM will always
        indicate progress bars however.
        '''

        #Zero out the training loss histories
        self.training_hist = []
        self.recon_hist = []
        self.recon_z1_hist = []
        self.pred_hist = []

        #Prepare sampler information
        class_sample_counts = np.array([np.sum(np.logical_not(y)),np.sum(y)])
        weight = 1. / class_sample_counts
        samples_weights = np.array([weight[t] for t in y.astype(int)])
        samples_weights = torch.from_numpy(samples_weights)

        #If using dCSFA_RNN check that M is the right
        #shape. Otherwise, generate an M matrix of all
        #ones.
        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)

        if self.model_type=='dCSFA_RNN':
            assert M.shape == X.shape
            M = torch.Tensor(M).to(self.device)

        else:
            M = torch.ones_like(X).to(self.device)

        #Create a balanced sampler
        
        #Create a dataloader for easy batching
        dset = TensorDataset(X,M,y)
        sampler = WeightedRandomSampler(samples_weights.type("torch.DoubleTensor"),len(samples_weights))
        loader = DataLoader(dset,batch_size=batch_size,sampler=sampler)
        #loader = DataLoader(dset,batch_size=batch_size,shuffle=True)        
        if verbose:
            print("Training the Model")
            
        for epoch in trange(n_epochs+n_pretrain_epochs, disable= not verbose, leave=False):
            if epoch < n_pretrain_epochs:
                pretrain_weight = 0
            else:
                pretrain_weight = 1
            #Initialize running epoch losses at each epoch
            epoch_loss = 0.0
            recon_e_loss = 0.0
            recon_z1_loss = 0.0
            pred_e_loss = 0.0

            for X_batch, M_batch, y_batch in loader:
                #If using dCSFA_RNN, the model will expect
                #a different shaped input that is
                #(LxNxH) where L is the number of features
                #N is the number of samples, and H is the
                #number of frequencies
                if self.model_type=='dCSFA_RNN':
                    X_batch_3d = X_batch.view(-1,self.n_features,self.n_freqs)
                    X_batch_3d = torch.swapaxes(X_batch_3d,0,1)
                    #Zero gradients and do the forward pass
                    self.optimizer.zero_grad()
                    X_recon, X_recon_z1, y_pred, _ = self.model(X_batch_3d)
                else:
                    #Zero gradients and do the forward pass
                    self.optimizer.zero_grad()
                    X_recon, X_recon_z1, y_pred, _ = self.model(X_batch)

                #Collect the reconstruction, supervised component reconstruction
                #and the prediction losses
                l_recon = self.recon_weight * self.Recon_Loss(X_recon*M_batch,X_batch*M_batch)
                l_recon_z1 = pretrain_weight * self.w_r_z1*self.Recon_Loss(X_recon_z1*M_batch,X_batch*M_batch)
                l_pred = pretrain_weight * self.w_sup*self.Prediction_Loss(y_pred.view(-1,1),y_batch.view(-1,1))

                #Sum the weighted losses and backproject and step
                loss = l_recon + l_recon_z1 + l_pred
                loss.backward()
                self.optimizer.step()

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
            if verbose and epoch%print_rate==0:
                tqdm.write(f'Epoch: {epoch}, loss: {epoch_loss},recon: {recon_e_loss} \n recon_z1: {recon_z1_loss} prediction: {pred_e_loss}')
        self.components = nn.ReLU()(self.model.W_nmf)

    def transform(self,X):
        '''
        transform
        ------------------
        returns a forward pass of the model

        Inputs
        -----------------
        X - np.array - N x M - Input data of shape N x M where N is the number of samples
        and M is n_freqs*n_features.

        Returns
        ----------------
        X_recon - np.array - N x M - Reconstruction of X
        X_recon_z1 - np.array - N x M - Reconstruction of X using only z1
        y_pred - np.array - N x 1 - prediction probabilities for y_pred
        z_pos - np.array - N x K - K is the number of components, this provides
        the raw scores for the networks
        '''
        X = torch.Tensor(X).to(self.device)

        if self.model_type=='dCSFA_RNN':
            X_3d = X.view(-1,self.n_features,self.n_freqs)
            X_3d = torch.swapaxes(X_3d,0,1)
            X_recon, X_recon_z1,y_pred,z_pos = self.model(X_3d)
        else:
            X_recon, X_recon_z1,y_pred,z_pos = self.model(X)
            
        X_recon = X_recon.cpu().detach().numpy()
        X_recon_z1 = X_recon_z1.cpu().detach().numpy()
        y_pred = y_pred.view(-1,1).cpu().detach().numpy() 
        z_pos = z_pos.cpu().detach().numpy()

        return X_recon, X_recon_z1, y_pred, z_pos