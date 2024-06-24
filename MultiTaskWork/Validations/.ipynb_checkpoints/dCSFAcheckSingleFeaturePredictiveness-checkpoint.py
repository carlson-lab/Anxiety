N_EPOCHS = 1500
N_PRE_EPOCHS = 500
VERBOSE = False
PRETRAIN = False

DATA_LOCATION = "/work/mk423/Anxiety/"
UMC_PATH = "/hpc/home/mk423/Anxiety/Universal-Mouse-Code/"
INFO_DICT = DATA_LOCATION + "Anx_Info_Dict.pkl"

FLX_TRAIN_FILE = DATA_LOCATION + "FLX_train_dict_old_features.pkl"
FLX_VAL_FILE = DATA_LOCATION + "FLX_validation_dict_old_features.pkl"
FLX_TEST_FILE = DATA_LOCATION + "FLX_test_dict_old_features.pkl"

EPM_TRAIN_FILE = DATA_LOCATION + "EPM_train_dict_May_17.pkl"
EPM_VAL_FILE = DATA_LOCATION + "EPM_val_dict_May_17.pkl"
EPM_TEST_FILE = DATA_LOCATION + "EPM_test_dict_May_17.pkl"

OFT_TRAIN_FILE = DATA_LOCATION + "OFT_train_dict_old_features_hand_picked.pkl"
OFT_VAL_FILE = DATA_LOCATION + "OFT_validation_dict_old_features_hand_picked.pkl"
OFT_TEST_FILE = DATA_LOCATION + "OFT_test_dict_old_features_hand_picked.pkl"

FEATURE_LIST = ['X_power_1_2','X_coh_1_2','X_gc_1_2']
FEATURE_VECTOR = FEATURE_LIST
FEATURE_WEIGHT = [10,1,1]

import sys, os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from lpne.models import DcsfaNmf
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.append(UMC_PATH)
sys.path.append(DATA_LOCATION)

with open(OFT_TRAIN_FILE,'rb') as f:
    train_dict = pickle.load(f)

with open(OFT_VAL_FILE,'rb') as f:
    val_dict = pickle.load(f)

with open(OFT_TEST_FILE,'rb') as f:
    test_dict = pickle.load(f)
    
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

oft_X_test = np.hstack([test_dict[feature]*weight for feature,weight in zip(FEATURE_LIST,FEATURE_WEIGHT)])
oft_y_hc_test = test_dict['y_Homecage'].astype(bool)
oft_y_task_test = ~oft_y_hc_test
oft_y_ROI_test = test_dict['y_ROI']
oft_y_vel_test = test_dict['y_vel']
oft_y_mouse_test = test_dict['y_mouse']
oft_y_time_test = test_dict['y_time']

oft_X = np.vstack([oft_X_train,oft_X_val])
oft_y_task = np.hstack([oft_y_task_train,oft_y_task_val])
oft_y_mouse = np.hstack([oft_y_mouse_train,oft_y_mouse_val])

with open(FLX_TRAIN_FILE,"rb") as f:
    flx_train_dict = pickle.load(f)

with open(FLX_VAL_FILE,"rb") as f:
    flx_validation_dict = pickle.load(f)

with open(FLX_TEST_FILE,"rb") as f:
    flx_test_dict = pickle.load(f)

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

flx_X_test = np.hstack([flx_test_dict[feature]*weight for feature,weight in zip(FEATURE_LIST,FEATURE_WEIGHT)])
flx_y_test = flx_test_dict['y_flx']
flx_y_mouse_test = flx_test_dict['y_mouse']
flx_y_expDate_test = flx_test_dict['y_expDate']
flx_y_time_test = flx_test_dict['y_time']

flx_X = np.vstack([flx_X_train,flx_X_validation])
flx_y_task = np.hstack([flx_y_train,flx_y_validation])
flx_y_mouse = np.hstack([flx_y_mouse_train,flx_y_mouse_validation])
flx_y_expDate = np.hstack([flx_y_expDate_train,flx_y_expDate_validation])
flx_y_time = np.hstack([flx_y_time_train,flx_y_time_validation])

with open(EPM_TRAIN_FILE,"rb") as f:
    epm_train_dict = pickle.load(f)

with open(EPM_VAL_FILE,"rb") as f:
    epm_validation_dict = pickle.load(f)

with open(EPM_TEST_FILE,"rb") as f:
    epm_test_dict = pickle.load(f)

#Load the data
NUM_FREQS = 56
NUM_FEATURES = (epm_train_dict["X_power_1_2"].shape[1] + \
                epm_train_dict["X_coh_1_2"].shape[1] + \
                epm_train_dict["X_gc_1_2"].shape[1]) // NUM_FREQS
SCALE_VECTOR = np.array([np.arange(1,57) for feature in range(NUM_FEATURES)]).flatten()

X_train = np.hstack([epm_train_dict[feature]*weight for feature,weight in zip(FEATURE_LIST,FEATURE_WEIGHT)])
#X_train[X_train<0] = 0
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

X_test = np.hstack([epm_test_dict[feature]*weight for feature,weight in zip(FEATURE_LIST,FEATURE_WEIGHT)])
y_test = (epm_test_dict['y_ROI']%2).astype(bool)
y_in_task_mask_test= ~epm_test_dict['y_Homecage'].astype(bool)
y_mouse_test = epm_test_dict['y_mouse']
y_time_test = epm_test_dict['y_time']
test_nan_mask = (epm_test_dict['y_ROI'] > 0)

X_test_task = X_test[np.logical_and(y_in_task_mask_test==1,test_nan_mask)==1]
y_test_task = y_test[np.logical_and(y_in_task_mask_test==1,test_nan_mask)==1]
y_mouse_test_task = y_mouse_test[np.logical_and(y_in_task_mask_test==1,test_nan_mask)==1]
y_time_test_task = y_time_test[np.logical_and(y_in_task_mask_test==1,test_nan_mask)==1]

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

mt_X_test = np.vstack([flx_X_test,oft_X_test,X_test])
mt_y_test = np.hstack([flx_y_test,oft_y_task_test,y_in_task_mask_test])
mt_y_mouse_test = np.hstack([flx_y_mouse_test,oft_y_mouse_test,y_mouse_test])

with open(INFO_DICT,"rb") as f:
    anx_info_dict = pickle.load(f)
    
anx_info_dict.keys()

powerFeatures = anx_info_dict['powerFeatures']


flx_val_mean_list = []
flx_val_std_err_list = []
epm_val_mean_list = []
epm_val_std_err_list = []
oft_val_mean_list = []
oft_val_std_err_list = []

val_data_tuples = [("flx",np.vstack((flx_X_validation,flx_X_test)),np.hstack((flx_y_validation,flx_y_test)),np.hstack((flx_y_mouse_validation,flx_y_mouse_test))),
                   ("epm",np.vstack((X_val,X_test)),np.hstack((y_in_task_mask_val,y_in_task_mask_test)),np.hstack((y_mouse_val,y_mouse_test))),
                   ("oft",np.vstack((oft_X_val,oft_X_test)),np.hstack((oft_y_task_val,oft_y_task_test)),np.hstack((oft_y_mouse_val,oft_y_mouse_test)))]


results_auc_dict = {}
results_auc_dict["feature"] = []
results_auc_dict["auc"] = []
results_auc_dict["n_networks"] = []

for area in anx_info_dict['area']:
    val_auc_list = []
    
    freq_check = [area + ' ' + str(freq) for freq in range(56)]
    feature_mask = np.array([powerFeature in freq_check for powerFeature in powerFeatures])
    x_power = mt_X_train[:,:len(powerFeatures)][:,feature_mask==1]
    x_power_val = mt_X_val[:,:len(powerFeatures)][:,feature_mask==1]
    
    for n_networks in range(2,30):
        
        model = DcsfaNmf(
            n_components=n_networks,
            n_intercepts = np.unique(mt_y_mouse_train).shape[0],
            optim_name="SGD",
            sup_recon_weight=0.01,
            sup_recon_type="All",
            save_folder = "/work/mk423/Anxiety/singleFeaturedCSFAModels/",
            fixed_corr=["positive"],
        )
        
        model.fit(x_power,
                  mt_y_train.reshape(-1,1),
                  intercept_mask=OneHotEncoder().fit_transform(mt_y_mouse_train.reshape(-1,1)).todense(),
                  n_epochs=N_EPOCHS,
                  n_pre_epochs=N_PRE_EPOCHS,
                  verbose=VERBOSE,
                  pretrain=PRETRAIN,
                  X_val=x_power_val,
                  y_val = mt_y_val.reshape(-1,1),
                  best_model_name="{}_{}_networks_dcsfa.pt".format(area,n_networks))
        
        torch.save(model,"/work/mk423/Anxiety/singleFeaturedCSFAModels/" + "{}_{}_networks_dcsfa.pt".format(area,n_networks))
        
        model._initialize(x_power.shape[1])
        model.encoder.eval()
        
        validation_auc = model.score(x_power_val,mt_y_val.reshape(-1,1))
        validation_recon = model.reconstruct(x_power_val)
        validation_mse = np.mean((validation_recon - x_power_val)**2)
        
        val_auc_list.append(validation_auc)
    
    results_auc_dict["feature"].append(area) 
    results_auc_dict["auc"].append(np.max(val_auc_list))
    results_auc_dict["n_networks"].append(range(2,30)[np.argmax(val_auc_list)])
    
    print("{} , auc {}, n_net {}".format(area,results_auc_dict["auc"][-1],results_auc_dict["n_networks"][-1]))
    
    
coh_features = ["Amy-PrL_Cx","Nac-VTA","Cg_Cx-PrL_Cx","Cg_Cx-IL_Cx","IL_Cx-PrL_Cx"]
all_features = np.concatenate([anx_info_dict["powerFeatures"],anx_info_dict["cohFeatures"],anx_info_dict["gcFeatures"]])

for area in coh_features:
    val_auc_list = []
    freq_check = [area + ' ' + str(freq) for freq in range(56)]
    feature_mask = np.array([feature in freq_check for feature in all_features])
    
    x_coh = mt_X_train[:,feature_mask==1]
    x_coh_val = mt_X_val[:,feature_mask==1]
    
    for n_networks in range(2,30):
    
        model = DcsfaNmf(
            n_components=n_networks,
            n_intercepts = np.unique(mt_y_mouse_train).shape[0],
            optim_name="SGD",
            sup_recon_weight=0.01,
            sup_recon_type="All",
            save_folder = "/work/mk423/Anxiety/singleFeaturedCSFAModels/",
            fixed_corr=["positive"],
        )

        model.fit(x_power,
                  mt_y_train.reshape(-1,1),
                  intercept_mask=OneHotEncoder().fit_transform(mt_y_mouse_train.reshape(-1,1)).todense(),
                  n_epochs=N_EPOCHS,
                  n_pre_epochs=N_PRE_EPOCHS,
                  verbose=VERBOSE,
                  pretrain=PRETRAIN,
                  X_val=x_power_val,
                  y_val = mt_y_val.reshape(-1,1),
                  best_model_name="{}_{}_networks_dcsfa.pt".format(area,n_networks))

        torch.save(model,"/work/mk423/Anxiety/singleFeaturedCSFAModels/" + "{}_{}_networks_dcsfa.pt".format(area,n_networks))

        model._initialize(x_power.shape[1])
        model.encoder.eval()

        validation_auc = model.score(x_coh_val,mt_y_val.reshape(-1,1))
        validation_recon = model.reconstruct(x_coh_val)
        validation_mse = np.mean((validation_recon - x_coh_val)**2)

        val_auc_list.append(validation_auc)

    results_auc_dict["feature"].append(area) 
    results_auc_dict["auc"].append(np.max(val_auc_list))
    results_auc_dict["n_networks"].append(range(2,30)[np.argmax(val_auc_list)])
    
    print("{} , auc {}, n_net {}".format(area,results_auc_dict["auc"][-1],results_auc_dict["n_networks"][-1]))
    
df = pd.DataFrame.from_dict(results_auc_dict)
df.to_csv("/hpc/home/mk423/Anxiety/MultiTaskWork/Validations/SingleFeatureAUCs_demo.csv")