import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc

class Encoders(nn.Module):
    """
    Encoders                                                        --- ZAC 03/11/2021: it would be helpful to explain that these are parts of VAEs somewhere around here
    -------------------------
    Encoders creates either a list of encoders that map to latent   --- ZAC 03/11/2021: 'encoder' refers the first half of the VAEs being trained?
    space z of size n_components. The number of encoders are        --- ZAC 03/11/2021: what does 'either' refer to? - the first sentence cuts off early.
    defined by the length of md_dim_list. 

    members                                                         --- ZAC 03/11/2021: It would be good to split 'members' into 'hyper params' and 'members', or something, because mean_layer and logvar_layer can't be adjusted via input, unlike the other members listed here. It is also unclear as to why W_pos and b_pos are not described.
    -------------------------
    n_components - int type - indicates the dimensionality of 
    latent dimension z

    md_dim_list - list type int - lists the different patterns      --- ZAC 03/11/2021: what does 'pattern' refer to?
    of missingness. Note that if the number of missing features     --- ZAC 03/11/2021: are the int values the 'number of missing features'? - if so, why is there a list?
    between two sets are identical, but the identity of missing     --- ZAC 03/11/2021: are the int values the 'identity of missing features'? - if so, what determines the id?
    features is different both numbers should be listed.
    For example [605,605].                                          --- ZAC 03/11/2021: I like having an example - it might be good to expand it and/or include more. Note that this example contradicts the one given for NMF_Missing_Data class

    n_hidden - int type - indicates the number of hidden nodes      --- ZAC 03/11/2021: I would say 'encoder hidden nodes', refering to the NN architecture, to clarify these are not nodes in a brain-model

    encoder_list - nn.ModuleList() - list of encoders for different
    patterns of missingness.                                        --- ZAC 03/11/2021: 'pattern of missingness' should be changed to 'input data dimensionalities' - see line 56 comment

    mean_layer - nn.Linear - function encoding from z to mean 
    logvar_layer - nn.Linear - function encoding from z to logvar


    """
    def __init__(self,n_components,md_dim_list,n_hidden=100,
                 device='cpu'):
        super(Encoders,self).__init__()

        #Define Constants and initialize module list
        self.n_components = n_components
        self.md_dim_list = md_dim_list
        self.encoder_list = nn.ModuleList()
        self.n_hidden=100
        self.device = device

        #Create encoders for each pattern of missingness
        for pm_shape in md_dim_list:                              # --- ZAC 03/11/2021: Oh! 'pattern of missingness' is 'input dimensionality' - this should be clarified above. Edit: I do not know what this is supposed to mean still, given lines 222-225.
            self.encoder_list.append(
                nn.Sequential(
                    nn.Linear(pm_shape,n_hidden),
                    nn.ELU(),
                    nn.Linear(n_hidden,n_components),
                    nn.ELU()
                )
            )
        
        #Create Layers to extract latent distribution info
        self.mean_layer = nn.Linear(self.n_components,
                                    self.n_components)
        self.logvar_layer = nn.Linear(self.n_components, 
                                self.n_components)
        
        #Create constants for positive transformation
        self.W_pos = nn.Parameter(                                # --- ZAC 03/11/2021: need a description for W_pos
                        torch.randn((n_components,n_components)),
                        requires_grad=True)
        self.b_pos = nn.Parameter(                                # --- ZAC 03/11/2021: need a description for b_pos
                            torch.randn(n_components),
                            requires_grad=True)

    def Sampling(self,mean,logvar):
        '''
        Reparamaterization trick function                           --- ZAC 03/11/2021: needs to reference VAE paper, otherwise it is unclear why this function is needed
        '''
        eps = torch.randn(logvar.shape).to(self.device)
        sample = mean + torch.exp(logvar/2)*eps

        return sample

    def forward(self,X,idx):
        '''
        X - matrix type torch.float32 - Input data
        idx - missing data encoder index. Corresponds to md_dim_list. - ZAC 03/11/2021: rather than 'missing data encoder', maybe just 'encoder', unless I'm misunderstanding 'pattern of missingness' still
        '''
        #select which encoder will be used
        model_index = idx
        x = self.encoder_list[model_index](X)
        #evaluate mean and logvar values
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        #sample (reparameterization trick)
        z = self.Sampling(mean,logvar)
        #Shift scores to be positive per NMF constraints
        z_pos = F.softplus(z@self.W_pos + self.b_pos)

        return (z,z_pos,mean,logvar)


class Decoder(nn.Module):
    '''
    Decoder                                                        --- ZAC 03/11/2021: This set of docs is much more clear than the docs for Encoder class, nice!
    ----------------------
    NMF style decoder where z @ components_ = X. In the case of
    missing data, the reconstruction should include all observed
    features.

    Member variables                                               --- ZAC 03/11/2021: I still think it would be good to split 'Member variables' into 'hyper params' and 'members', or something, because some members can't be adjusted via constructor
    ---------------------
    n_components - type int - size of latent dimensionality        --- ZAC 03/11/2021: I like that this naming convention holds with Encoder class convention

    out_dim - type int - number of unique featuers and the 
            dimensionality of the output space

    components_ - matrix type torch.float32 - decoding positive
                definite matrix, also called networks for lpne     --- ZAC 03/11/2021: good detail

    '''
    def __init__(self,n_components,out_dim,device='cpu'):
        super(Decoder,self).__init__()
        self.n_components = n_components
        self.out_dim = out_dim
        self.components_ = nn.Parameter(torch.randn(
                                n_components,out_dim
                                ),
                                requires_grad=True
                            )
        
    def forward(self,z):                                         # --- ZAC 03/11/2021: would be good to explain what the two different reconstructions are for
        #Get complete reconstruction
        X_recon = torch.matmul(z,F.relu(self.components_))
        #Get reconstruction for z1
        X_recon_from_z1 = torch.matmul(z[:,0].reshape(-1,1),
                                    F.relu(self.components_[0,:]).reshape(1,-1))
        return X_recon,X_recon_from_z1    

class LR_Classifier(nn.Module):
    '''
    LR_Classifier                                                  --- ZAC 03/11/2021: also very clear docs
    ------------------
    Logistic Regression classifier for the supervised components
    of the model. Our models only supervise the first component
    so our logistic regression is only of dim=1

    Member variables
    ------------------
    Phi_ - type int torch parameter - Coefficient
    B_ - type int torch parameter - Intercept term
    '''
    def __init__(self):
        super(LR_Classifier,self).__init__()
        self.Phi_ = nn.Parameter(torch.randn(1),
                        requires_grad=True)
        self.B_ = nn.Parameter(torch.randn(1),
                        requires_grad=True)

    def forward(self,x):
        y_pred = F.sigmoid(x*self.Phi_ + self.B_)
        return y_pred

class NMF_Missing_Data(nn.Module):
    '''
    NMF_Missing_Data                                               --- ZAC 03/11/2021: needs summary of what class is to be used for
    ----------------------------

    Member Variables
    ----------------------------
    n_components - int - dimensionality of latent dimension z

    md_dim_list - list type int - each value in this list 
        corresponds to a different pattern of missingness. This is --- ZAC 03/11/2021: 'pattern of missingness' should be changed to 'input data dimensionalities' - see line 56 comment
        with respect to unique variables, so if two patterns of 
        missingness have the same number of features, both will be
        notated ([605,605,800])                                    --- ZAC 03/11/2021: I like having an example - it might be a good idea to expand it and/or include more. Note that this example contradicts the one given for Encoder class

    out_dim - int - number of unique features and size of the 
        decoded data

    n_hidden - int - number of nodes in the hidden layers of the 
        encoders

    task_list - list type string - list of each classification task -- ZAC 03/11/2021: what is this for? I see it appear on lines 216 and 217, but not sure what's happening.
        used ['FLX','OFT','EPM]

    device - string - {'cpu','cuda:0','cuda:1'...
        'cuda:I_wish_I_had_more_GPUs'}                             --- ZAC 03/11/2021: bahaha
    '''
    def __init__(self,n_components,md_dim_list,out_dim,
                task_list,n_hidden=100,device='cpu'):
        super(NMF_Missing_Data,self).__init__()

        #Store constants
        self.n_components = n_components
        self.md_dim_list = md_dim_list
        self.out_dim = out_dim
        self.n_hidden=n_hidden
        self.task_list = task_list
        self.device=device

        #Initialize Encoder and Decoder
        self.md_encoder = Encoders(n_components,md_dim_list,
                                n_hidden,device)
        self.md_decoder = Decoder(n_components,out_dim,device)

        #Initialize Classifiers
        self.shared_lr = LR_Classifier()
        self.multi_task_dict = nn.ModuleDict()
        for task in task_list:
            self.multi_task_dict[task] = LR_Classifier()         # --- ZAC 03/11/2021: why train multiple LR models?

    def forward(self,X,idx,task,md=False):
        '''
        X - matrix type torch.float32 - Input Data
        idx - int - index in md_dim_list for the pattern of missingness. --- ZAC 03/11/2021: 'pattern of missingness' is not 'input dimensionality'? - this should be clarified above, particularly given that it appears throughout the doc.
            Needs to be an index and not a key as it is possible to have
            2 patterns of missingness with the same number of unique
            features that are not the same features.
        task - string - name of prediction task used in task_list
        md - bool - whether or not missing data is used.
        '''
        #encode data
        if md:
            z_raw,z,z_mean,logvar = self.md_encoder(X,idx)
        else:
            z_raw,z,z_mean,logvar = self.md_encoder(X,0)

        encoder_tuple = z_raw,z,z_mean,logvar
        #Reconstruct
        X_recon, X_recon_z1 = self.md_decoder(z)
        #Get shared and task specific prediction
        y_pred_shared = self.shared_lr(z[:,0])
        y_pred_task = self.multi_task_dict[task](z[:,0])

        return encoder_tuple,X_recon,X_recon_z1,y_pred_shared, y_pred_task
        
