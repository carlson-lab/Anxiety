import torch
import numpy as np
import pickle
from lpne.models import DcsfaNmf
from lpne.plotting import circle_plot
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image
import os, sys
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score

umc_data_tools_path = "/hpc/home/mk423/Anxiety/Universal-Mouse-Code/"
sys.path.append(umc_data_tools_path)
import umc_data_tools as umc_dt


def score_lr(y_true,y_pred,y_group):
    auc_list = []
    
    for group in np.unique(y_group):
        group_mask = y_group==group
        
        if not (len(np.unique(y_true[group_mask==1])) < 2):
            group_auc = roc_auc_score(y_true[group_mask==1],y_pred[group_mask==1])
            auc_list.append(group_auc)
        
    mean = np.mean(auc_list)
    sterr = np.std(auc_list) / np.sqrt(len(auc_list))
    
    return mean, sterr
        
featureIdx = int(os.environ['SLURM_ARRAY_TASK_ID']) #0-91

flx_data_path = "/work/mk423/Anxiety/final_FLX_{}.pkl"
epm_data_path = "/work/mk423/Anxiety/EPM_{}_dict_May_17.pkl"
oft_data_path = "/work/mk423/Anxiety/OFT_{}_dict_old_features_hand_picked.pkl"

anx_info_dict = "/work/mk423/Anxiety/Anx_Info_Dict.pkl"

feature_list = ["X_psd","X_coh","X_gc"]
old_feature_list = ["X_power_1_2","X_coh_1_2","X_gc_1_2"]
feature_weights = [10,1,1]
RANDOM_STATE = 42
EPSILON = 1e-6

anxInfoDict = pickle.load(open(anx_info_dict,"rb"))
allFeatures = np.hstack([anxInfoDict["powerFeatures"],anxInfoDict["cohFeatures"],anxInfoDict["gcFeatures"]])
currentFeature = np.unique([feature.split(" ")[0] for feature in allFeatures[56*featureIdx:56*(featureIdx+1)]])[0]
print("Running ",currentFeature)

saved_model_path = "/work/mk423/Anxiety/MultiTask/singleFeatureModels/"
saved_model_name = "{}_singleFeature_model_{}_components.pt"

results_path = "/hpc/home/mk423/Anxiety/FullDataWork/Validations/LR_SingleFeature/"
results_file = results_path + "{}_singleFeature_model_LR_results.pkl"


with open(flx_data_path.format("train"),"rb") as f:
    train_dict = pickle.load(f)
    
with open(flx_data_path.format("val"),"rb") as f:
    val_dict = pickle.load(f)
    
with open(flx_data_path.format("test"),"rb") as f:
    test_dict = pickle.load(f)
    
flx_X_train = np.hstack([train_dict[feature]*weight for feature,weight in zip(feature_list,feature_weights)])
flx_y_train = train_dict['y_flx']
flx_y_mouse_train = train_dict['y_mouse']
flx_y_hab_train = train_dict['y_hab']
flx_y_time_train = train_dict['y_time']

flx_X_val = np.hstack([val_dict[feature]*weight for feature,weight in zip(feature_list,feature_weights)])
flx_y_val = val_dict['y_flx']
flx_y_mouse_val = val_dict['y_mouse']
flx_y_hab_val = val_dict['y_hab']
flx_y_time_val = val_dict['y_time']

flx_X_test = np.hstack([test_dict[feature]*weight for feature,weight in zip(feature_list,feature_weights)])
flx_y_test = test_dict['y_flx']
flx_y_mouse_test = test_dict['y_mouse']
flx_y_hab_test = test_dict['y_hab']
flx_y_time_test = test_dict['y_time']
flx_y_expDate_test = test_dict['y_expDate']

#Stack Validation and Training Data from timepoints after 30 minutes
flx_X = np.vstack([flx_X_train[flx_y_hab_train==1],flx_X_val[flx_y_hab_val==1]])
flx_y = np.hstack([flx_y_train[flx_y_hab_train==1],flx_y_val[flx_y_hab_val==1]])
flx_y_mouse = np.hstack([flx_y_mouse_train[flx_y_hab_train==1], flx_y_mouse_val[flx_y_hab_val==1]])

with open(oft_data_path.format("train"),'rb') as f:
    train_dict = pickle.load(f)

with open(oft_data_path.format("validation"),'rb') as f:
    val_dict = pickle.load(f)
    
with open(oft_data_path.format("test"),'rb') as f:
    test_dict = pickle.load(f)
    

running_idx = 0
feature_groups = []
for idx,feature in enumerate(old_feature_list):
    f_begin = running_idx
    f_end = f_begin + train_dict[feature].shape[1] 
    if idx == 0:
        f_end = f_end -1
    feature_groups.append((f_begin,f_end))

    running_idx = f_end

NUM_FREQS = 56
NUM_FEATURES = np.hstack([train_dict[feature] for feature in old_feature_list]).shape[1] // NUM_FREQS
scale_vector = np.array([np.arange(1,NUM_FREQS+1) for feature in range(NUM_FEATURES)]).flatten()

#Train Arrays
oft_X_train = np.hstack([train_dict[feature]*weight for feature,weight in zip(old_feature_list,feature_weights)])
oft_y_hc_train = train_dict['y_Homecage'].astype(bool)
oft_y_task_train = ~oft_y_hc_train
oft_y_ROI_train = train_dict['y_ROI']
oft_y_vel_train = train_dict['y_vel']
oft_y_mouse_train = train_dict['y_mouse']
oft_y_time_train = train_dict['y_time']

#Validation Arrays
oft_X_val = np.hstack([val_dict[feature]*weight for feature,weight in zip(old_feature_list,feature_weights)])
oft_y_hc_val = val_dict['y_Homecage'].astype(bool)
oft_y_task_val = ~oft_y_hc_val
oft_y_ROI_val = val_dict['y_ROI']
oft_y_vel_val = val_dict['y_vel']
oft_y_mouse_val = val_dict['y_mouse']
oft_y_time_val = val_dict['y_time']

#Test Arrays
oft_X_test = np.hstack([test_dict[feature]*weight for feature,weight in zip(old_feature_list,feature_weights)])
oft_y_hc_test = test_dict['y_Homecage'].astype(bool)
oft_y_task_test = ~oft_y_hc_test
oft_y_ROI_test = test_dict['y_ROI']
oft_y_vel_test = test_dict['y_vel']
oft_y_mouse_test = test_dict['y_mouse']
oft_y_time_test = test_dict['y_time']
oft_y_expDate_test = test_dict['y_expDate']

oft_X = np.vstack([oft_X_train,oft_X_val])
oft_y_task = np.hstack([oft_y_task_train,oft_y_task_val])
oft_y_mouse = np.hstack([oft_y_mouse_train,oft_y_mouse_val])

with open(epm_data_path.format("train"),"rb") as f:
    epm_train_dict = pickle.load(f)

with open(epm_data_path.format("val"),"rb") as f:
    epm_validation_dict = pickle.load(f)
    
with open(epm_data_path.format("test"),"rb") as f:
    epm_test_dict = pickle.load(f)

#Load the data
NUM_FREQS = 56
NUM_FEATURES = (epm_train_dict["X_power_1_2"].shape[1] + \
                epm_train_dict["X_coh_1_2"].shape[1] + \
                epm_train_dict["X_gc_1_2"].shape[1]) // NUM_FREQS
SCALE_VECTOR = np.array([np.arange(1,57) for feature in range(NUM_FEATURES)]).flatten()

X_train = np.hstack([epm_train_dict[feature]*weight for feature,weight in zip(old_feature_list,feature_weights)])
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
X_val = np.hstack([epm_validation_dict[feature]*weight for feature,weight in zip(old_feature_list,feature_weights)])


y_val = (epm_validation_dict['y_ROI']%2).astype(bool)
y_in_task_mask_val= ~epm_validation_dict['y_Homecage'].astype(bool)
y_mouse_val = epm_validation_dict['y_mouse']
y_time_val = epm_validation_dict['y_time']
val_nan_mask = (epm_validation_dict['y_ROI'] > 0)

X_val_task = X_val[np.logical_and(y_in_task_mask_val==1,val_nan_mask)==1]
y_val_task = y_val[np.logical_and(y_in_task_mask_val==1,val_nan_mask)==1]
y_mouse_val_task = y_mouse_val[np.logical_and(y_in_task_mask_val==1,val_nan_mask)==1]
y_time_val_task = y_time_val[np.logical_and(y_in_task_mask_val==1,val_nan_mask)==1]

X_test = np.hstack([epm_test_dict[feature]*weight for feature,weight in zip(old_feature_list,feature_weights)])
y_test = (epm_test_dict['y_ROI']%2).astype(bool)
y_in_task_mask_test= ~epm_test_dict['y_Homecage'].astype(bool)
y_mouse_test = epm_test_dict['y_mouse']
y_time_test = epm_test_dict['y_time']
test_nan_mask = (epm_test_dict['y_ROI'] > 0)
epm_y_expDate_test = epm_test_dict['y_expDate']

X_test_task = X_test[np.logical_and(y_in_task_mask_test==1,test_nan_mask)==1]
y_test_task = y_test[np.logical_and(y_in_task_mask_test==1,test_nan_mask)==1]
y_mouse_test_task = y_mouse_test[np.logical_and(y_in_task_mask_test==1,test_nan_mask)==1]
y_time_test_task = y_time_test[np.logical_and(y_in_task_mask_test==1,test_nan_mask)==1]

epm_X = np.vstack([X_train,X_val])
epm_y_task = np.hstack([y_in_task_mask_train,y_in_task_mask_val])
epm_y_mouse = np.hstack([y_mouse_train,y_mouse_val])
epm_y_time = np.hstack([y_time_train,y_time_val])

mt_X_train = np.vstack([flx_X,oft_X,epm_X])[:,56*featureIdx:56*(featureIdx+1)]
mt_y_train = np.hstack([flx_y,oft_y_task,epm_y_task]).reshape(-1,1)
mt_y_mouse_train = np.hstack([flx_y_mouse,oft_y_mouse,epm_y_mouse])

mt_X_test = np.vstack([flx_X_test[flx_y_hab_test==1],oft_X_test,X_test])[:,56*featureIdx:56*(featureIdx+1)]
mt_y_test = np.hstack([flx_y_test[flx_y_hab_test==1],oft_y_task_test,y_in_task_mask_test]).reshape(-1,1)
mt_y_mouse_test = np.hstack([flx_y_mouse_test[flx_y_hab_test==1],oft_y_mouse_test,y_mouse_test])

intercept_mask = OneHotEncoder().fit_transform(mt_y_mouse_train.reshape(-1,1)).todense()
sample_groups = OrdinalEncoder().fit_transform(mt_y_mouse_train.reshape(-1,1))
nmf_groups = np.hstack(
    [
        np.ones(flx_X.shape[0])*0,
        np.ones(oft_X.shape[0]),
        np.ones(epm_X.shape[0])*2
    ]
)

TRAIN = True

model = LogisticRegression(max_iter=4000)
model.fit(mt_X_train,mt_y_train)
    
flx_test_pred = model.predict_proba(flx_X_test[flx_y_hab_test==1,56*featureIdx:56*(featureIdx+1)])
flx_mean, flx_sterr = score_lr(flx_y_test[flx_y_hab_test==1],
                               flx_test_pred[:,1],
                               flx_y_mouse_test[flx_y_hab_test==1])

epm_test_pred = model.predict_proba(X_test[:,56*featureIdx:56*(featureIdx+1)])
epm_mean, epm_sterr = score_lr(y_in_task_mask_test,
                               epm_test_pred[:,1],
                               y_mouse_test)

oft_test_pred = model.predict_proba(oft_X_test[:,56*featureIdx:56*(featureIdx+1)])
oft_mean, oft_sterr = score_lr(oft_y_task_test,
                               oft_test_pred[:,1],
                               oft_y_mouse_test)


results_dict = {
    "flx_mean_test_auc":flx_mean,
    "flx_stderr_test_auc":flx_sterr,

    "epm_mean_test_auc":epm_mean,
    "epm_stderr_test_auc":epm_sterr,

    "oft_mean_test_auc":oft_mean,
    "oft_stderr_test_auc":oft_sterr,
    
    "feature":currentFeature
}

with open(results_file.format(currentFeature),"wb") as f:
    pickle.dump(results_dict,f)