import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import pandas as pd

DATA_PATH = "/work/mk423/Anxiety/"
sys.path.append(DATA_PATH)
FLX_TRAIN_FILE = DATA_PATH + "FLX_train_dict_old_features.pkl"
FLX_VAL_FILE = DATA_PATH + "FLX_validation_dict_old_features.pkl"

EPM_TRAIN_FILE = DATA_PATH + "EPM_train_dict_May_17.pkl"
EPM_VAL_FILE = DATA_PATH + "EPM_val_dict_May_17.pkl"

OFT_TRAIN_FILE = DATA_PATH + "OFT_train_dict_old_features_hand_picked.pkl"
OFT_VAL_FILE = DATA_PATH + "OFT_validation_dict_old_features_hand_picked.pkl"

FEATURE_LIST = ['X_power_1_2','X_coh_1_2','X_gc_1_2']
#FEATURE_LIST = ['X_psd','X_ds']
FEATURE_VECTOR = FEATURE_LIST
FEATURE_WEIGHT = [10,1,1]

UMC_PATH = "/hpc/home/mk423/Anxiety/Universal-Mouse-Code"

sys.path.append(UMC_PATH)
#from dCSFA_model import dCSFA_model
import umc_data_tools as umc_dt
from dCSFA_NMF import dCSFA_NMF

if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"


print("Using device: %s"%(device))

#For Consistency
RANDOM_STATE=42

with open(OFT_TRAIN_FILE,'rb') as f:
    train_dict = pickle.load(f)

with open(OFT_VAL_FILE,'rb') as f:
    val_dict = pickle.load(f)

running_idx = 0
feature_groups = []
for idx,feature in enumerate(FEATURE_LIST):
    f_begin = running_idx
    f_end = f_begin + train_dict[feature].shape[1] 
    if idx == 0:
        f_end = f_end -1
    feature_groups.append((f_begin,f_end))

    running_idx = f_end

NUM_FREQS = 56
NUM_FEATURES = np.hstack([train_dict[feature] for feature in FEATURE_LIST]).shape[1] // NUM_FREQS
scale_vector = np.array([np.arange(1,NUM_FREQS+1) for feature in range(NUM_FEATURES)]).flatten()

#Train Arrays
oft_X_train = np.hstack([train_dict[feature]*weight for feature,weight in zip(FEATURE_LIST,FEATURE_WEIGHT)])#*scale_vector
#oft_X_train[np.isnan(oft_X_train)] = 0
#oft_X_train[oft_X_train<0] = 0
oft_y_hc_train = train_dict['y_Homecage'].astype(bool)
oft_y_task_train = ~oft_y_hc_train
oft_y_ROI_train = train_dict['y_ROI']
oft_y_vel_train = train_dict['y_vel']
oft_y_mouse_train = train_dict['y_mouse']
oft_y_time_train = train_dict['y_time']

#Validation Arrays
oft_X_val = np.hstack([val_dict[feature]*weight for feature,weight in zip(FEATURE_LIST,FEATURE_WEIGHT)])#*scale_vector
oft_y_hc_val = val_dict['y_Homecage'].astype(bool)
oft_y_task_val = ~oft_y_hc_val
oft_y_ROI_val = val_dict['y_ROI']
oft_y_vel_val = val_dict['y_vel']
oft_y_mouse_val = val_dict['y_mouse']
oft_y_time_val = val_dict['y_time']

oft_X = np.vstack([oft_X_train,oft_X_val])
oft_y_task = np.hstack([oft_y_task_train,oft_y_task_val])
oft_y_mouse = np.hstack([oft_y_mouse_train,oft_y_mouse_val])

with open(FLX_TRAIN_FILE,"rb") as f:
    flx_train_dict = pickle.load(f)

with open(FLX_VAL_FILE,"rb") as f:
    flx_validation_dict = pickle.load(f)

flx_X_train = np.hstack([flx_train_dict[feature]*weight for feature,weight in zip(FEATURE_LIST,FEATURE_WEIGHT)])
flx_y_train = flx_train_dict['y_flx']
flx_y_mouse_train = flx_train_dict['y_mouse']
flx_y_expDate_train = flx_train_dict['y_expDate']
flx_y_time_train = flx_train_dict['y_time']

flx_X_validation = np.hstack([flx_validation_dict[feature]*weight for feature,weight in zip(FEATURE_LIST,FEATURE_WEIGHT)])
flx_y_validation = flx_validation_dict['y_flx']
flx_y_mouse_validation = flx_validation_dict['y_mouse']
flx_y_expDate_validation = flx_validation_dict['y_expDate']
flx_y_time_validation = flx_validation_dict['y_time']

flx_X = np.vstack([flx_X_train,flx_X_validation])
flx_y_task = np.hstack([flx_y_train,flx_y_validation])
flx_y_mouse = np.hstack([flx_y_mouse_train,flx_y_mouse_validation])
flx_y_expDate = np.hstack([flx_y_expDate_train,flx_y_expDate_validation])
flx_y_time = np.hstack([flx_y_time_train,flx_y_time_validation])

with open(EPM_TRAIN_FILE,"rb") as f:
    epm_train_dict = pickle.load(f)

with open(EPM_VAL_FILE,"rb") as f:
    epm_validation_dict = pickle.load(f)
#Load the data
NUM_FREQS = 56
NUM_FEATURES = (epm_train_dict["X_power_1_2"].shape[1] + \
                epm_train_dict["X_coh_1_2"].shape[1] + \
                epm_train_dict["X_gc_1_2"].shape[1]) // NUM_FREQS
SCALE_VECTOR = np.array([np.arange(1,57) for feature in range(NUM_FEATURES)]).flatten()

X_train = np.hstack([epm_train_dict[feature]*weight for feature,weight in zip(FEATURE_LIST,FEATURE_WEIGHT)])
X_train[X_train<0] = 0
y_train = (epm_train_dict['y_ROI']%2).astype(bool)
y_in_task_mask_train = ~epm_train_dict['y_Homecage'].astype(bool)
y_mouse_train = epm_train_dict['y_mouse']
y_time_train = epm_train_dict['y_time']
train_nan_mask = (epm_train_dict['y_ROI'] > 0)


X_train_task = X_train[np.logical_and(y_in_task_mask_train==1,train_nan_mask)==1]
y_train_task = y_train[np.logical_and(y_in_task_mask_train==1,train_nan_mask)==1]
y_mouse_train_task = y_mouse_train[np.logical_and(y_in_task_mask_train==1,train_nan_mask)==1]
y_time_train_task = y_time_train[np.logical_and(y_in_task_mask_train==1,train_nan_mask)==1]
X_val = np.hstack([epm_validation_dict[feature]*weight for feature,weight in zip(FEATURE_LIST,FEATURE_WEIGHT)])


y_val = (epm_validation_dict['y_ROI']%2).astype(bool)
y_in_task_mask_val= ~epm_validation_dict['y_Homecage'].astype(bool)
y_mouse_val = epm_validation_dict['y_mouse']
y_time_val = epm_validation_dict['y_time']
val_nan_mask = (epm_validation_dict['y_ROI'] > 0)

X_val_task = X_val[np.logical_and(y_in_task_mask_val==1,val_nan_mask)==1]
y_val_task = y_val[np.logical_and(y_in_task_mask_val==1,val_nan_mask)==1]
y_mouse_val_task = y_mouse_val[np.logical_and(y_in_task_mask_val==1,val_nan_mask)==1]
y_time_val_task = y_time_val[np.logical_and(y_in_task_mask_val==1,val_nan_mask)==1]

epm_X = np.vstack([X_train_task,X_val_task])
epm_y_task = np.hstack([y_train_task,y_val_task])
epm_y_mouse = np.hstack([y_mouse_train_task,y_mouse_val_task])
epm_y_time = np.hstack([y_time_train_task,y_time_val_task])


mt_X_train = np.vstack([flx_X_train,oft_X_train,X_train])
mt_y_train = np.hstack([flx_y_train,oft_y_task_train,y_in_task_mask_train])
mt_y_mouse_train = np.hstack([flx_y_mouse_train,oft_y_mouse_train,y_mouse_train])

mt_X_val = np.vstack([flx_X_validation,oft_X_val,X_val])
mt_y_val = np.hstack([flx_y_validation,oft_y_task_val,y_in_task_mask_val])
mt_y_mouse_val = np.hstack([flx_y_mouse_validation,oft_y_mouse_val,y_mouse_val])

intercept_mask = OneHotEncoder().fit_transform(mt_y_mouse_train.reshape(-1,1)).todense()

TRAIN = True
DIM_IN = flx_X_train.shape[1]
NETWORK_CONSTRAINT = "Positive"
SAVE_FOLDER = "../Models/"
#Jan_24 Changed to sup_recon all, sup recon weight to .01, nmf_max_iter to 1500 from 500, added validation early stopping
SAVE_FILE = "{}_comp_x_val_Jan_24.pt"
skl_mse_list = []
dcsfa_mse_list = []
dcsfa_w_list = []
if TRAIN:
    for components in range(2,30):

        model = dCSFA_NMF(n_components=components,dim_in=DIM_IN,device='auto',n_intercepts=intercept_mask.shape[1],
                        n_sup_networks=1,optim_name='SGD',recon_loss='MSE',sup_recon_weight=.01,sup_weight=1,
                        useDeepEnc=True,h=256,sup_recon_type="All",feature_groups=feature_groups,fixed_corr=NETWORK_CONSTRAINT)
        model.fit(mt_X_train,mt_y_train,intercept_mask=intercept_mask,batch_size=128,lr=1e-3,
                n_pre_epochs=100,n_epochs=1500,verbose=True,pretrain=True,best_model_name=SAVE_FOLDER + SAVE_FILE.format(components),
                momentum=0.9,X_val=mt_X_val,y_val=mt_y_val)
        
        torch.save(model,SAVE_FOLDER + SAVE_FILE.format(components))
        model.Encoder.eval()
        skl_mse, dcsfa_mse = model.get_skl_nn_mse(mt_X_val)
        skl_mse_list.append(skl_mse)
        dcsfa_mse_list.append(dcsfa_mse)
        print("n_components: {}, val MSE: {:.4}, val SKL MSE: {:.4}".format(components,dcsfa_mse,skl_mse))
else:
    for components in range(2,24):
        model = torch.load(SAVE_FOLDER + SAVE_FILE.format(components),map_location="cpu")
        model.device="cpu"
        model.Encoder.eval()
        skl_mse, dcsfa_mse = model.get_skl_nn_mse(mt_X_val)
        skl_mse_list.append(skl_mse)
        dcsfa_mse_list.append(dcsfa_mse)
        dcsfa_w_list.append(model.get_W_nmf().cpu().detach().numpy()[0,:].reshape(1,-1))
        print("n_components: {}, val MSE: {:.4}, val SKL MSE: {:.4}".format(components,dcsfa_mse,skl_mse))
        
        