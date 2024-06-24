from lpne.models import DcsfaNmf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os
from sklearn.preprocessing import OneHotEncoder
import torch
DATA_PATH = "/work/mk423/Anxiety/Anxiety_Network_Generation_Data.pkl"

with open(DATA_PATH,"rb") as f:
    dataDict = pickle.load(f)
    
    
for n_components in range(1,40):
    model = DcsfaNmf(
        n_components=n_components,
        n_intercepts=np.unique(dataDict["y_mouse_train"]).shape[0],
        optim_name="SGD",
        sup_recon_weight=0.1,
        sup_recon_type="All",
        save_folder="/hpc/home/mk423/Anxiety/MultiTaskWork/Models/lpneNumNetVal/",
        fixed_corr=["positive"],
        feature_groups=dataDict["feature_groups"],
        )
    model.fit(dataDict["X_train"],
              dataDict["y_train"].reshape(-1,1),
              intercept_mask=OneHotEncoder().fit_transform(dataDict["y_mouse_train"].reshape(-1,1)).todense(),
              n_epochs=3000,
              n_pre_epochs=500,
              nmf_max_iter=15000,
              verbose=True,
              X_val=dataDict["X_val"],
              y_val=dataDict["y_val"].reshape(-1,1),
              best_model_name="3000_epoch_{}_network_lpne_dcsfa_model.pt".format(n_components))
    
    torch.save(model,"/hpc/home/mk423/Anxiety/MultiTaskWork/Models/lpneNumNetVal/3000_epoch_{}_network_lpne_dcsfa_full_model.pt")