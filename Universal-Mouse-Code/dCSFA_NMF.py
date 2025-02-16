from numpy import require
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC
from tqdm import tqdm, trange
import numpy as np
from torch.utils.data import WeightedRandomSampler
from sklearn.decomposition import NMF
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torchbd.loss import BetaDivLoss
from scipy.stats import mannwhitneyu
import warnings

class dCSFA_NMF(nn.Module):
    def __init__(self,n_components,dim_in,device='auto',n_intercepts=1,n_sup_networks=1,
                optim_name='AdamW',recon_loss='IS',sup_recon_weight=1.0,sup_weight=1.0,
                useDeepEnc=True,h=256,sup_recon_type="Residual",feature_groups=None,group_weights=None,
                fixed_corr=None):
        """
        Inputs
        ----------------------------
        n_components - int - The number of latent factors to learn. This is also often referred to as the number of networks
        dim_in - int - The number of features of the input. This is often the second dimension of the feature matrix. This can
                        be expressed as the number of frequencies multiplied by the number of features F
        device - {'auto','cpu','cuda'} - The device that the model will be ran on. "auto" will check for a gpu and then select it if
                                        available. Otherwise it picks cpu.
        n_intercepts - int - The number of unique intercepts to use. This allows for a different intercept for different tasks.
        n_sup_networks - int - The number of supervised networks you would like to learn. Must be less than the number of components
        optim_name - {'AdamW','Adam','SGD'} - The optimizer being used.
        recon_loss - {'IS','MSE'} - Reconstruction loss. MSE corresponds to Mean Squared Error. IS corresponds to itakura-saito.
                                    For IS loss be sure to run `pip install beta-divergence-metrics`
        sup_recon_weight - float - weight coefficient for the first component reconstruction loss.
        sup_weight - float - supervision importance weight coefficient
        useDeepEnc - bool - indicates if a neural network encoder should be used
        sup_recon_type - {"Residual","All"} - controls whether you would like to encourage the first component to reconstruct the residual of the
                                            other component reconstructions ("Residual"), or if you would like to encourage reconstruction of the
                                            whole dataset using MSE ("All")
        variational - bool - indicates if a variational autoencoder will be used.
        prior_mean - int - only used if variational. Zero is the default.
        prior_var - int - only used if variational. One is the default.
        fixed_corr - {None, "positive","negative"}

        Other Member Variables
        ---------------------------
        optim_f - torch.optim.<function> - optimizer function to be used for training
        recon_loss_f - torch.optim.<function> - reconstruction loss function to be used for training
        Encoder - nn.Sequential - Single hidden layer nn.Sequential encoder that makes use of batch norm and LeakyReLU
        Encoder_A - nn.Parameter - Torch tensor for linear transformation for the linear encoder
        Encoder_b - nn.Parameter - Torch tensor for the bias in the linear encoder
        W_nmf - nn.Parameter - Input for the W_nmf decoder function that returns nn.Softplus(W_nmf)
        phi_ - nn.Parameter - Coefficient for the logistic regression classifier
        beta_ nn.Parameter - Bias vector for the logistic regression classifier

        Example Code
        ---------------------------
        #Load the data
        with open(TRAIN_PATH,"rb") as f:
            train_dict = pickle.load(f)
        with open(VAL_PATH,"rb") as f:
            validation_dict = pickle.load(f)

        #Create Scaling Vectors
        NUM_FREQS = 56
        num_features = (train_dict['X_psd'].shape[1] + train_dict['X_ds'].shape[1]) // NUM_FREQS
        scale_vector = np.array([np.arange(1,57) for feature in range(num_features)]).flatten()

        #Format the data
        X_train = np.hstack([train_dict["X_psd"],train_dict["X_ds"]])*scale_vector
        #X_train = X_train / np.mean(X_train,axis=0)
        y_task_train = train_dict['y_flx'].astype(bool)
        y_mouse_train = train_dict['y_mouse']

        X_val = np.hstack([validation_dict["X_psd"],validation_dict["X_ds"]])*scale_vector
        #X_val = X_val / np.mean(X_train,axis=0)
        y_task_val = validation_dict['y_flx'].astype(bool)
        y_mouse_val = validation_dict['y_mouse']

        model = dCSFA_NMF(n_components=20,dim_in = X_train.shape[1],device='auto',n_intercepts=1,
                optim_name='Adam',recon_loss='IS',sup_recon_weight=.001,sup_weight=1.0,
                useDeepEnc="True",h=256)
        model.fit(X_train,y_task_train,intercept_mask=None,n_epochs=100,n_pre_epochs=25,batch_size=128,lr=1e-3, 
                pretrain=True,verbose=True,print_rate=20)

        skl_mse, dcsfa_mse = model.get_skl_nn_mse(X_train)
        X_recon,X_recon_z1,y_pred,scores = model.transform(X_train)
        y_pred = model.predict(X_train)
        """
        super(dCSFA_NMF,self).__init__()
        self.n_components = n_components
        self.dim_in = dim_in
        self.n_sup_networks = n_sup_networks
        self.optim_name = optim_name
        self.optim_f = self.get_optim(optim_name)
        self.recon_loss = recon_loss
        self.recon_loss_f = self.get_recon_loss(recon_loss)
        self.sup_recon_weight = sup_recon_weight
        self.sup_weight = sup_weight
        self.n_intercepts = n_intercepts
        self.useDeepEnc = useDeepEnc
        self.sup_recon_type = sup_recon_type
        self.h = h
        self.fixed_corr = fixed_corr
        self.skl_pretrain_model = None
        self.feature_groups = feature_groups
        if feature_groups is not None and group_weights is None:
            group_weights = []
            for idx,(lb,ub) in enumerate(feature_groups):
                group_weights.append((feature_groups[-1][-1] - feature_groups[0][0])/(ub - lb))
            self.group_weights = group_weights
        else:
            self.group_weights = group_weights
            
            
        #Use deep encoder or linear
        if useDeepEnc:
            self.Encoder = nn.Sequential(nn.Linear(dim_in,self.h),
                            nn.BatchNorm1d(self.h),
                            nn.LeakyReLU(),
                            nn.Linear(self.h,n_components),
                            nn.Softplus(),
                            )
        else:
            self.Encoder_A = nn.Parameter(torch.randn(dim_in,n_components))
            self.Encoder_b = nn.Parameter(torch.randn(n_components))

        #Define nmf decoder parameter
        self.W_nmf = nn.Parameter(torch.rand(n_components,dim_in))

        #Logistic Regression Parameters
        self.phi_ = nn.Parameter(torch.randn(self.n_sup_networks))
        self.beta_ = nn.Parameter(torch.randn(n_intercepts,self.n_sup_networks))

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.to(self.device)
    
    @torch.no_grad()
    def get_optim(self,optim_name):
        '''
        returns a torch optimizer based on text input from the user
        '''
        if optim_name == "AdamW":
            return torch.optim.AdamW
        elif optim_name == "Adam":
            return torch.optim.Adam
        elif optim_name == "SGD":
            return torch.optim.SGD
        else:
            raise ValueError(f"{optim_name} is not supported")
    
    @torch.no_grad()
    def get_recon_loss(self,recon_loss):
        '''
        get the reconstruction loss
        '''
        if recon_loss == "MSE":
            return nn.MSELoss()
        elif recon_loss == "IS":
            return BetaDivLoss(beta=0,reduction="mean")
        else:
            raise ValueError(f"{recon_loss} is not supported")

    @staticmethod
    def inverse_softplus(x, eps=1e-5):
        '''
        Gets the inverse softplus for sklearn model pretraining
        '''
        # Calculate inverse softplus
        x_inv_softplus = np.log(np.exp(x) - (1.0 - eps))
        
        # Return inverse softplus
        return x_inv_softplus

    def get_W_nmf(self):
        W = nn.Softplus()(self.W_nmf)
        return W

    def get_phi(self):
        if self.fixed_corr == None:
            return self.phi_
        elif self.fixed_corr.lower() == "positive":
            return nn.Softplus()(self.phi_)
        elif self.fixed_corr.lower() == "negative":
            return -1*nn.Softplus()(self.phi_)
        else:
            raise ValueError("Unsupported fixed_corr value")

    def get_weighted_recon_loss_f(self,X_pred,X_true):
        recon_loss = 0.0
        for weight,(lb,ub) in zip(self.group_weights,self.feature_groups):
            recon_loss += weight * self.recon_loss_f(X_pred[:,lb:ub],X_true[:,lb:ub])

        return recon_loss

    @torch.no_grad()
    def skl_pretrain(self,X,y):
        '''
        Description
        ----------------------
        This method trains an sklearn NMF model to initialize W_nmf. First W_nmf is trained (using IS or MSE loss respectively) and we get the scores.
        W_nmf is then sorted according to the predictive ability of each of the components as gotten by the sklearn Logistic Regression. The sorted components
        are then saved.
        '''
        print("Pretraining NMF...")
        if self.recon_loss == "IS":
            skl_NMF = NMF(n_components=self.n_components,solver="mu",beta_loss="itakura-saito",init='nndsvda',max_iter=15000)
        else:
            skl_NMF = NMF(n_components=self.n_components,max_iter=15000)
        s_NMF = skl_NMF.fit_transform(X)
    

        class_coef_list = []
        class_bias_list = []
        class_auc_list = []
        pMask = y==1
        nMask = ~pMask
        print("Identifying predictive components...")
        for component in tqdm(range(self.n_components)):

            s_pos = s_NMF[pMask==1,component].reshape(-1,1)
            s_neg = s_NMF[nMask==1,component].reshape(-1,1)
            U,pval = mannwhitneyu(s_pos,s_neg)
            U = U.squeeze()
            auc = U/(len(s_pos)*len(s_neg))
            
            skl_classifier = LogisticRegression()
            skl_classifier.fit(s_NMF[:,component].reshape(-1,1),y)

            class_coef_list.append(skl_classifier.coef_)
            class_bias_list.append(skl_classifier.intercept_)
            class_auc_list.append(auc)

        class_auc_list = np.array(class_auc_list) 

        if self.fixed_corr == None:
            #find most seperable auc
            predictive_order = np.argsort(np.abs(class_auc_list - 0.5))[::-1]
            print("Selecting most separable correlated NMF factor for init. Network {}, AUC: {}".format(predictive_order[0],class_auc_list[predictive_order[0].astype(int)]))
            
        elif self.fixed_corr.lower()=="positive":
            #find most positive auc
            predictive_order = np.argsort(class_auc_list)[::-1]
            print("Selecting most positively correlated NMF factor for init. Network {}, AUC: {}".format(predictive_order[0],class_auc_list[predictive_order[0].astype(int)]))
        elif self.fixed_corr.lower() == "negative":
            #find most negative auc
            predictive_order = np.argsort(1-class_auc_list)[::-1]
            print("Selecting most negatively correlated NMF factor for init. Network {}, AUC: {}".format(predictive_order[0],class_auc_list[predictive_order[0].astype(int)]))


        sorted_NMF = skl_NMF.components_[predictive_order]
        self.W_nmf.data = torch.from_numpy(self.inverse_softplus(sorted_NMF).astype(np.float32)).to(self.device)
        self.skl_pretrain_model = skl_NMF

    def encoder_pretrain(self,X,n_pre_epochs=25,batch_size=128,verbose=False,print_rate=5):
        '''
        Description
        ------------------------
        This method freezes the W_nmf parameter and trains the Encoder using only the full reconstruction loss.
        '''
        self.W_nmf.requires_grad = False
        self.pretrain_hist = []
        X = torch.Tensor(X).to(self.device)
        dset = TensorDataset(X)
        loader = DataLoader(dset,batch_size=batch_size,shuffle=True)
        pre_optimizer = self.optim_f(self.parameters(),lr=1e-3)
        epoch_iter = tqdm(range(n_pre_epochs))
        for epoch in epoch_iter:
            r_loss = 0.0
            for X_batch, in loader:
                pre_optimizer.zero_grad()
                X_recon = self.forward(X_batch,avgIntercept=True)[0]
                if self.feature_groups is not None:
                    loss_recon = self.get_weighted_recon_loss_f(X_recon,X_batch)
                else:
                    loss_recon = self.recon_loss_f(X_recon,X_batch)
                loss = loss_recon
                loss.backward()
                pre_optimizer.step()
                r_loss += loss_recon.item()
            self.pretrain_hist.append(r_loss)
            if verbose:
                epoch_iter.set_description("Pretrain Epoch: %d, Recon Loss: %0.2f"%(epoch,r_loss))
        self.W_nmf.requires_grad = True

    @torch.no_grad()
    def transform(self,X,intercept_mask=None):
        '''
        This method returns a forward pass without tracking gradients. Use this to get model reconstructions
        '''
        if intercept_mask is not None:
            assert intercept_mask.shape == (X.shape[0],self.n_intercepts)
            intercept_mask = torch.Tensor(intercept_mask).to(self.device)
        else:
            avgIntercept=True
        #Move to device
        X = torch.Tensor(X).to(self.device)
        X_recon,sup_recon_loss,y_pred_proba,s = self.forward(X,intercept_mask,avgIntercept=avgIntercept)

        return X_recon.cpu().detach().numpy(), sup_recon_loss.cpu().detach().numpy(), y_pred_proba.cpu().detach().numpy(), s.cpu().detach().numpy()

    @torch.no_grad()
    def project(self,X):
        s = self.transform(X)[3]
        return s
        
    @torch.no_grad()
    def predict(self,X,intercept_mask=None,include_scores=False):
        '''
        Returns a boolean array of predicted labels. Use include_scores=True to get
        the original scores
        '''
        y_pred_proba,s = self.transform(X,intercept_mask)[2:]
        
        y_class = y_pred_proba > 0.5
        if include_scores:
            return y_class, s
        else:
            return y_class

    def get_sup_recon(self,s):
        return s[:,self.n_sup_networks].view(-1,self.n_sup_networks) @ self.get_W_nmf()[:self.n_sup_networks,:].view(self.n_sup_networks,-1)

    def get_residual_scores(self,X,s):
        resid = (X-s[:,self.n_sup_networks:] @ self.get_W_nmf()[self.n_sup_networks:,:])
        w_sup = self.get_W_nmf()[:self.n_sup_networks,:].view(self.n_sup_networks,-1)
        s_h = resid @ w_sup.T @ torch.inverse(w_sup@w_sup.T)
        return s_h

    def residual_loss_f(self,s,s_h):
        res_loss = torch.norm(s[:,:self.n_sup_networks].view(-1,self.n_sup_networks)-s_h) / (1 - torch.exp(-3*torch.norm(s_h)))
        return res_loss

    def forward(self,X,intercept_mask=None,avgIntercept=False):
        #Encode X using the deep or linear encoder
        if self.useDeepEnc:
            s = self.Encoder(X)
        else:
            s = nn.Softplus()(X @ self.Encoder_A + self.Encoder_b)

        if self.n_intercepts == 1:
            y_pred_proba = nn.Sigmoid()(s[:,:self.n_sup_networks].view(-1,self.n_sup_networks) * self.get_phi() + self.beta_).squeeze()
        elif self.n_intercepts > 1 and not avgIntercept:
            y_pred_proba = nn.Sigmoid()(s[:,:self.n_sup_networks].view(-1,self.n_sup_networks) * self.get_phi() + intercept_mask @ self.beta_).squeeze()
        else:
            intercept_mask = torch.ones(X.shape[0],self.n_intercepts).to(self.device) / self.n_intercepts
            y_pred_proba = nn.Sigmoid()(s[:,:self.n_sup_networks].view(-1,self.n_sup_networks) * self.get_phi() + intercept_mask @ self.beta_).squeeze()

        X_recon = s @ self.get_W_nmf()

        if self.sup_recon_type == "Residual":
            s_h = self.get_residual_scores(X,s)
            sup_recon_loss = self.residual_loss_f(s,s_h)
        elif self.sup_recon_type == "All":
            X_recon = self.get_sup_recon(s)
            sup_recon_loss = self.recon_loss_f(X_recon,X)
        else:
            raise ValueError("self.sup_recon_type must be one of the following: {'Residual','All'}")
        
        return X_recon,sup_recon_loss, y_pred_proba, s

    def fit(self,X,y,y_sample_groups=None,intercept_mask=None,task_mask=None,n_epochs=25,n_pre_epochs=25,batch_size=128,lr=1e-3, 
            pretrain=True,verbose=False,print_rate=5,X_val=None,y_val=None,task_mask_val=None,best_model_name="dCSFA-NMF-best-model.pt",momentum=0):

        if intercept_mask is not None:
            assert intercept_mask.shape == (X.shape[0],self.n_intercepts)
            intercept_mask = torch.Tensor(intercept_mask).to(self.device)
        elif intercept_mask is None and self.n_intercepts==1:
            intercept_mask = torch.ones(X.shape[0],1).to(self.device)
        else:
            raise ValueError("intercept mask cannot be type None and n_intercepts greater than 1")


        #Zero out the training loss histories
        self.training_hist = []
        self.recon_hist = []
        self.recon_z1_hist = []
        self.score_reg_hist = []
        self.pred_hist = []
        if verbose: print("Pretraining....")
        if pretrain:
            self.skl_pretrain(X,y)
            self.encoder_pretrain(X,n_pre_epochs=n_pre_epochs,verbose=verbose,
                                    print_rate=print_rate,batch_size=batch_size)
        if verbose: print("Pretraining Complete")

        #Prepare sampler information
        if y_sample_groups is None:
            y_sample_groups = y
        
        class_sample_counts = np.array([np.sum(y_sample_groups==group) for group in np.unique(y_sample_groups)])
        weight = 1. / class_sample_counts
        samples_weights = np.array([weight[t] for t in y_sample_groups.astype(int)]).squeeze()
        samples_weights = torch.Tensor(samples_weights)
        #Send information to Tensors
        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)

        if task_mask is None:
            task_mask = torch.ones_like(y).to(self.device)
        else:
            task_mask = torch.Tensor(task_mask).to(self.device)

        if X_val is not None and y_val is not None:
            self.best_model_name = best_model_name
            self.best_val_loss = 1e8
            self.val_loss_hist = []
            self.val_recon_loss_hist = []
            self.val_sup_recon_loss_hist = []
            self.val_pred_loss_hist = []
            X_val = torch.Tensor(X_val).to(self.device)
            y_val = torch.Tensor(y_val).to(self.device)
            if task_mask_val is None:
                task_mask_val = torch.ones_like(y_val).to(self.device)
            else:
                task_mask_val = torch.Tensor(task_mask_val).to(self.device)

        dset = TensorDataset(X,y,intercept_mask,task_mask)
        sampler = WeightedRandomSampler(samples_weights,len(samples_weights))
        loader = DataLoader(dset,batch_size=batch_size,sampler=sampler)
        if self.optim_name=="SGD":
            optimizer = self.optim_f(self.parameters(),lr=lr,momentum=momentum)
        else:
            optimizer = self.optim_f(self.parameters(),lr=lr)
        if verbose: print("Beginning Training")
        epoch_iter = tqdm(range(n_epochs))
        for epoch in epoch_iter:
        #for epoch in trange(n_epochs,disable = not verbose, leave=False):

            #Initialize running epoch losses at each epoch
            epoch_loss = 0.0
            recon_e_loss = 0.0
            sup_recon_e_loss = 0.0
            pred_e_loss = 0.0

            for X_batch, y_batch, b_mask_batch,task_mask_batch in loader:
                optimizer.zero_grad()
                X_recon, sup_recon_loss, y_pred_proba, s = self.forward(X_batch,b_mask_batch,avgIntercept=False)
                if self.feature_groups is not None:
                    recon_loss = self.get_weighted_recon_loss_f(X_recon,X_batch)
                else:
                    recon_loss = self.recon_loss_f(X_recon,X_batch)
                sup_recon_loss = self.sup_recon_weight * sup_recon_loss
                pred_loss = self.sup_weight * nn.BCELoss()(y_pred_proba*task_mask_batch,y_batch)
                loss = recon_loss + pred_loss + sup_recon_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                recon_e_loss += recon_loss.item()
                sup_recon_e_loss += sup_recon_loss.item()
                pred_e_loss += pred_loss.item()
            
            self.training_hist.append(epoch_loss)
            self.recon_hist.append(recon_e_loss)
            self.recon_z1_hist.append(sup_recon_e_loss)
            self.pred_hist.append(pred_e_loss)

            if X_val is not None and y_val is not None:
                with torch.no_grad():
                    X_recon_val,sup_recon_loss_val,y_pred_proba_val,_ = self.forward(X_val,avgIntercept=True)
                    if self.feature_groups is not None:
                        val_recon_loss = self.get_weighted_recon_loss_f(X_recon_val,X_val)
                    else:
                        val_recon_loss = self.recon_loss_f(X_recon_val,X_val)
                    val_sup_recon_loss = self.sup_recon_weight * sup_recon_loss_val
                    val_pred_loss = self.sup_weight * nn.BCELoss()(y_pred_proba_val*task_mask_val,y_val)
                    val_loss = val_recon_loss+val_sup_recon_loss+val_pred_loss
                    self.val_loss_hist.append(val_loss.item())
                    self.val_recon_loss_hist.append(val_recon_loss.item())
                    self.val_sup_recon_loss_hist.append(val_sup_recon_loss.item())
                    self.val_pred_loss_hist.append(val_pred_loss.item())
                
                if val_loss.item() < self.best_val_loss:
                    self.best_epoch = epoch
                    self.best_val_loss = val_loss.item()
                    torch.save(self.state_dict(),self.best_model_name)

            if verbose and (X_val is not None and y_val is not None):
                epoch_iter.set_description(f"Epoch: {epoch}, Best Val (Epoch/Loss): {self.best_epoch,self.best_val_loss}, loss: {epoch_loss}, recon: {recon_e_loss}, pred: {pred_e_loss}")
            elif verbose:
                epoch_iter.set_description(f"Epoch: {epoch}, loss: {epoch_loss}, recon: {recon_e_loss}, pred: {pred_e_loss}")
        
        if X_val is not None and y_val is not None:
            print("Loading best model...")
            self.load_state_dict(torch.load(self.best_model_name))
            print("Done!")

    def _component_recon(self,h,component):
        W = self.get_W_nmf()
        X_recon = h[:,component].view(-1,1) @ W[component,:].view(1,-1)
        return X_recon
    
    @torch.no_grad()
    def get_comp_recon(self,h,component):
        h = torch.Tensor(h).to(self.device)
        X_recon = self._component_recon(h,component)
        return X_recon.cpu().detach().numpy()

    @torch.no_grad()
    def get_skl_mse_score(self,X):
        if self.skl_pretrain_model is not None:
            s_skl = self.skl_pretrain_model.transform(X)
        else:
            warnings.warn("No Pretraining NMF model present - Training new one from scratch")
            if self.recon_loss == "IS":
                skl_NMF = NMF(n_components=self.n_components,solver="mu",beta_loss="itakura-saito",init='nndsvda',max_iter=500)
            else:
                skl_NMF = NMF(n_components=self.n_components,max_iter=500)
            s_skl = skl_NMF.fit_transform(X)
            self.skl_pretrain_model = skl_NMF
        X_recon_skl = s_skl @ self.skl_pretrain_model.components_
        skl_mse = np.mean((X_recon_skl-X)**2)
        return skl_mse

    @torch.no_grad()
    def get_mse_score(self,X):
        X_recon_nn = self.transform(X)[0]
        nn_mse = np.mean((X_recon_nn-X)**2)

        return nn_mse

    @torch.no_grad()
    def get_skl_nn_mse(self,X):
        "Returns the sklearn mse and dCSFA mse"
        skl_mse = self.get_skl_mse_score(X)
        nn_mse = self.get_mse_score(X)
        return skl_mse, nn_mse
