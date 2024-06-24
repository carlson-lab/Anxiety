import torch
import numpy as np
import pickle
from lpne.models import DcsfaNmf
from lpne.plotting import circle_plot
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from PIL import Image
import matplotlib.pyplot as plt
import os, sys

umc_data_tools_path = "/hpc/home/mk423/Anxiety/Universal-Mouse-Code/"
sys.path.append(umc_data_tools_path)
import umc_data_tools as umc_dt

N_COMPONENTS=30

fold = int(os.environ['SLURM_ARRAY_TASK_ID'])

flx_data_path = "/work/mk423/Anxiety/flx_kf_dict_fold_{}.pkl".format(fold)
epm_data_path = "/work/mk423/Anxiety/epm_kf_dict_fold_{}.pkl".format(fold)
oft_data_path = "/work/mk423/Anxiety/oft_kf_dict_fold_{}.pkl".format(fold)
anx_info_dict = "/work/mk423/Anxiety/Anx_Info_Dict.pkl"

saved_model_path = "/work/mk423/Anxiety/kfold_models/"
saved_model_name = "Adam_2_sup_net_30_net_kf_fold_{}_model.pt".format(fold)

results_path = "/hpc/home/mk423/Anxiety/FullDataWork/Validations/"
results_file = results_path + "Adam_2_sup_net_30_net_kf_fold_{}_results.pkl".format(fold)

projection_save_path = "/hpc/home/mk423/Anxiety/FullDataWork/Projections/"
plots_path = "/hpc/home/mk423/Anxiety/FullDataWork/Figures/"


def reshapeData(X_psd,X_coh,n_rois,n_freqs,pow_features,coh_features,areas):
    X_3d = np.zeros((n_rois,n_rois,n_freqs))
    
    for i in range(n_rois):
        X_3d[i,i,:] = X_psd[i*n_freqs:(i+1)*n_freqs]
        
    
    split_coh_features = np.array([feature.split(' ')[0] for feature in coh_features])
    #print(split_coh_features)
    unique_coh_features = np.unique(split_coh_features)
    for i in range(n_rois):
        for j in range(n_rois):
            if i != j:
                area_1 = areas[i]
                area_2 = areas[j]
                temp_feature = area_1 + "-" + area_2
                temp_feature_2 = area_2 + "-" + area_1
                if temp_feature in unique_coh_features:
                    feature_mask = np.where(split_coh_features==temp_feature,True,False)
                    X_3d[i,j,:] = X_coh[feature_mask==1]
                    X_3d[j,i,:] = X_coh[feature_mask==1]

                elif temp_feature_2 in unique_coh_features:
                    feature_mask = np.where(split_coh_features==temp_feature_2,1,0)
                    X_3d[i,j,:] = X_coh[feature_mask==1]
                    X_3d[j,i,:] = X_coh[feature_mask==1]

                else:
                    print("temp_feature: {} not found".format(temp_feature))

    return X_3d

with open(flx_data_path,"rb") as f:
    flx_dict = pickle.load(f)
    
with open(epm_data_path,"rb") as f:
    epm_dict = pickle.load(f)
    
with open(oft_data_path,"rb") as f:
    oft_dict = pickle.load(f)
    
with open(anx_info_dict,"rb") as f:
    anxInfo = pickle.load(f)

info_dict = anxInfo
feature_groups = [(0,len(info_dict["powerFeatures"])),
                   (len(info_dict["powerFeatures"]),len(info_dict["powerFeatures"])+len(info_dict["cohFeatures"])),
                   (len(info_dict["powerFeatures"])+len(info_dict["cohFeatures"]),
                    len(info_dict["powerFeatures"])+len(info_dict["cohFeatures"])+len(info_dict["gcFeatures"]))]
                   
mt_X_train = np.vstack([flx_dict["X_train"],epm_dict["X_train"],oft_dict["X_train"]])
mt_y_train = np.hstack([flx_dict["y_train"],epm_dict["y_train"],oft_dict["y_train"]]).reshape(-1,1)
mt_y_train_2_net = np.hstack([mt_y_train,mt_y_train])
mt_y_mouse_train = np.hstack([flx_dict["y_mouse_train"],epm_dict["y_mouse_train"],oft_dict["y_mouse_train"]])
mt_y_exp_train = np.hstack([np.ones(flx_dict["X_train"].shape[0])*0,
                           np.ones(epm_dict["X_train"].shape[0]),
                           np.ones(oft_dict["X_train"].shape[0])*2])
intercept_mask = OneHotEncoder().fit_transform(mt_y_mouse_train.reshape(-1,1)).todense()
sample_groups = OrdinalEncoder().fit_transform(mt_y_mouse_train.reshape(-1,1))

train_idxs = np.random.binomial(1,.7,size=mt_X_train.shape[0])
val_idxs = 1 - train_idxs
mt_X_val = np.vstack([flx_dict["X_val"],epm_dict["X_val"],oft_dict["X_val"]])
mt_y_val = np.hstack([flx_dict["y_val"],epm_dict["y_val"],oft_dict["y_val"]]).reshape(-1,1)
mt_y_val_2_net = np.hstack([mt_y_val,mt_y_val])
mt_y_mouse_val = np.hstack([flx_dict["y_mouse_val"],epm_dict["y_mouse_val"],oft_dict["y_mouse_val"]])

model = DcsfaNmf(
    n_components=N_COMPONENTS,
    n_intercepts=intercept_mask.shape[1],
    n_sup_networks=2,
    optim_name="Adam",
    recon_loss="MSE",
    sup_recon_type="Residual",
    sup_recon_weight=1,
    sup_weight=3,
    feature_groups=feature_groups,
    fixed_corr=["positive","positive"],
    save_folder=saved_model_path,
)

model.fit(mt_X_train,
          mt_y_train_2_net,
          intercept_mask=intercept_mask,
          y_sample_groups=mt_y_exp_train,
          batch_size=128,
          lr=1e-3,
          n_pre_epochs=500,
          n_epochs=1000,
          nmf_max_iter=2000,
          pretrain=True,
          verbose=True,
          #X_val=mt_X_val,
          #y_val=mt_y_val_2_net,
          best_model_name=saved_model_name)

torch.save(model,saved_model_path + saved_model_name)

#Multitask Performance
mt_train_auc = model.score(mt_X_train,mt_y_train_2_net)
mt_val_auc = model.score(mt_X_val,mt_y_val_2_net)

#FLX Performance

flx_y_train = np.hstack([flx_dict["y_train"].reshape(-1,1),flx_dict["y_train"].reshape(-1,1)])
flx_y_val = np.hstack([flx_dict["y_val"].reshape(-1,1),flx_dict["y_val"].reshape(-1,1)])

flx_train_auc = model.score(flx_dict["X_train"],flx_y_train,
                           flx_dict['y_mouse_train'],return_dict=True)
flx_val_auc = model.score(flx_dict["X_val"],flx_y_val,
                          flx_dict["y_mouse_val"],return_dict=True)

#EPM Performance
epm_y_train = np.hstack([epm_dict["y_train"].reshape(-1,1),epm_dict["y_train"].reshape(-1,1)])
epm_y_val = np.hstack([epm_dict["y_val"].reshape(-1,1),epm_dict["y_val"].reshape(-1,1)])
epm_train_auc = model.score(epm_dict["X_train"],epm_y_train,
                           epm_dict["y_mouse_train"],return_dict=True)
epm_val_auc = model.score(epm_dict["X_val"],epm_y_val,
                          epm_dict["y_mouse_val"],return_dict=True)

#OFT Performance
oft_y_train = np.hstack([oft_dict["y_train"].reshape(-1,1),oft_dict["y_train"].reshape(-1,1)])
oft_y_val = np.hstack([oft_dict["y_val"].reshape(-1,1),oft_dict["y_val"].reshape(-1,1)])

oft_train_auc = model.score(oft_dict["X_train"],oft_y_train,
                            oft_dict['y_mouse_train'],return_dict=True)
oft_val_auc = model.score(oft_dict["X_val"],oft_y_val,
                          oft_dict['y_mouse_val'],return_dict=True)

print("\nflx train",flx_train_auc)
print("\nflx val",flx_val_auc)
print("\nepm train",epm_train_auc)
print("\nepm val",epm_val_auc)
print("\noft train",oft_train_auc)
print("\noft val",oft_val_auc)

s = model.project(mt_X_val)
X_sup_recon = model.get_comp_recon(torch.Tensor(s).to("cuda"),0)
X_sup_recon_2 = model.get_comp_recon(torch.Tensor(s).to("cuda"),1)
X_recon = model.reconstruct(mt_X_val)

net_1_recon_contribution = np.mean(X_sup_recon/X_recon,axis=0)
net_2_recon_contribution = np.mean(X_sup_recon_2/X_recon,axis=0)

rec_psd = net_1_recon_contribution[:len(anxInfo["powerFeatures"])]
rec_coh = net_1_recon_contribution[len(anxInfo["powerFeatures"]):(len(anxInfo["powerFeatures"]) + len(anxInfo["cohFeatures"]))]
rec_3d = reshapeData(rec_psd,rec_coh,8,56,anxInfo["powerFeatures"],anxInfo["cohFeatures"],anxInfo["area"])

rec_psd_2 = net_2_recon_contribution[:len(anxInfo["powerFeatures"])]
rec_coh_2 = net_2_recon_contribution[len(anxInfo["powerFeatures"]):(len(anxInfo["powerFeatures"]) + len(anxInfo["cohFeatures"]))]
rec_3d_2 = reshapeData(rec_psd_2,rec_coh_2,8,56,anxInfo["powerFeatures"],anxInfo["cohFeatures"],anxInfo["area"])

circle_plot(rec_3d,anxInfo["area"],freqs=np.arange(56),freq_ticks=np.arange(0,56,5),
            min_max_quantiles=(0.85,0.9999),fn=plots_path + "Adam_net_1_30_component_kf_fold_{}_electome.png".format(fold))

circle_plot(rec_3d_2,anxInfo["area"],freqs=np.arange(56),freq_ticks=np.arange(0,56,5),
            min_max_quantiles=(0.85,0.9999),fn=plots_path + "Adam_net_2_30_component_kf_fold_{}_electome.png".format(fold))

umc_dt.makeUpperTriangularPlot_pow_coh_gc(net_1_recon_contribution.reshape(1,-1),anxInfo["area"],anxInfo["powerFeatures"],
                                          anxInfo["cohFeatures"],anxInfo["gcFeatures"],
                                          saveFile=plots_path + "Adam_net_1_30_component_kf_fold_{}_up-tri.png".format(fold))
umc_dt.makeUpperTriangularPlot_pow_coh_gc(net_2_recon_contribution.reshape(1,-1),anxInfo["area"],anxInfo["powerFeatures"],
                                          anxInfo["cohFeatures"],anxInfo["gcFeatures"],
                                          saveFile=plots_path + "Adam_net_2_30_component_kf_fold_{}_up-tri.png".format(fold))

results_dict = {
    "flx_train_auc":flx_train_auc,
    "flx_val_auc":flx_val_auc,
    "epm_train_auc":epm_train_auc,
    "epm_val_auc":epm_val_auc,
    "oft_train_auc":oft_train_auc,
    "oft_val_auc":oft_val_auc,
    "recon_cont_net_1":net_1_recon_contribution,
    "recon_cont_net_2":net_2_recon_contribution,
}

with open(results_file,"wb") as f:
    pickle.dump(results_dict,f)