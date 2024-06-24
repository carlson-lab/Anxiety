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

TRAIN=False
N_COMPONENTS=30
EPSILON = 1e-6
fold = int(os.environ['SLURM_ARRAY_TASK_ID'])
sup_components_list = [1,2,3,4,5,10,20]

flx_data_path = "/work/mk423/Anxiety/fixed_flx_kf_dict_fold_{}.pkl".format(fold)
epm_data_path = "/work/mk423/Anxiety/fixed_epm_kf_dict_fold_{}.pkl".format(fold)
oft_data_path = "/work/mk423/Anxiety/fixed_oft_kf_dict_fold_{}.pkl".format(fold)
anx_info_dict = "/work/mk423/Anxiety/Anx_Info_Dict.pkl"

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
mt_y_train_3_net = np.hstack([mt_y_train,mt_y_train,mt_y_train])
mt_y_mouse_train = np.hstack([flx_dict["y_mouse_train"],epm_dict["y_mouse_train"],oft_dict["y_mouse_train"]])

mt_y_exp_train = np.hstack([np.ones(flx_dict["X_train"].shape[0])*0,
                           np.ones(epm_dict["X_train"].shape[0]),
                           np.ones(oft_dict["X_train"].shape[0])*2])
intercept_mask = OneHotEncoder().fit_transform(mt_y_mouse_train.reshape(-1,1)).todense()
sample_groups = OrdinalEncoder().fit_transform(mt_y_mouse_train.reshape(-1,1))

mt_X_val = np.vstack([flx_dict["X_val"],epm_dict["X_val"],oft_dict["X_val"]])
mt_y_val = np.hstack([flx_dict["y_val"],epm_dict["y_val"],oft_dict["y_val"]]).reshape(-1,1)
mt_y_val_3_net = np.hstack([mt_y_val,mt_y_val,mt_y_val])
mt_y_mouse_val = np.hstack([flx_dict["y_mouse_val"],epm_dict["y_mouse_val"],oft_dict["y_mouse_val"]])

for n_sup_networks in sup_components_list:
    
    saved_model_path = "/work/mk423/Anxiety/kfold_models/MultiTask/"
    saved_model_name = "{}_sup_net_fold_{}_model.pt".format(n_sup_networks,fold)

    results_path = "/hpc/home/mk423/Anxiety/FullDataWork/Validations/MultiTask/"
    results_file = results_path + "{}_sup_net_fold_{}_results_w_scores.pkl".format(n_sup_networks,fold)

    projection_save_path = "/hpc/home/mk423/Anxiety/FullDataWork/Projections/"
    plots_path = "/hpc/home/mk423/Anxiety/FullDataWork/Figures/MultiTask/"
    plot_name = plots_path + "sup_net_{}_of_{}_fold_{}_{}_w_scores.png"
    
    if TRAIN:
        model = DcsfaNmf(
            n_components=30,
            n_sup_networks=n_sup_networks,
            sup_weight=3,
            sup_type="sc",
            n_intercepts=intercept_mask.shape[1],
            optim_name="SGD",
            recon_loss="MSE",
            feature_groups=feature_groups,
            fixed_corr=["positive" for i in range(n_sup_networks)],
            save_folder=saved_model_path,
        )

        model.fit(mt_X_train,
                  mt_y_train,
                  intercept_mask=intercept_mask,
                  y_sample_groups=sample_groups,
                  batch_size=128,
                  lr=1e-3,
                  n_pre_epochs=1000,
                  n_epochs=1000,
                  nmf_max_iter=3000,
                  nmf_groups=mt_y_exp_train,
                  bootstrap=True,
                  pretrain=True,
                  verbose=True,
                  best_model_name=saved_model_name)

        torch.save(model,saved_model_path + saved_model_name)
    else:
        model = torch.load(saved_model_path + saved_model_name,map_location="cpu")
        model.device="cpu"

    #Multitask Performance
    mt_train_auc = model.score(mt_X_train,mt_y_train)
    mt_val_auc = model.score(mt_X_val,mt_y_val)

    #FLX Performance

    flx_y_train = np.hstack([flx_dict["y_train"].reshape(-1,1),flx_dict["y_train"].reshape(-1,1)])
    flx_y_val = flx_dict["y_val"].reshape(-1,1)

    flx_train_auc = model.score(flx_dict["X_train"],flx_y_train,
                               flx_dict['y_mouse_train'],return_dict=True)
    flx_val_auc = model.score(flx_dict["X_val"],flx_y_val,
                              flx_dict["y_mouse_val"],return_dict=True)

    #EPM Performance
    epm_y_train = np.hstack([epm_dict["y_train"].reshape(-1,1),epm_dict["y_train"].reshape(-1,1)])
    epm_y_val = epm_dict["y_val"].reshape(-1,1)

    epm_train_auc = model.score(epm_dict["X_train"],epm_y_train,
                               epm_dict["y_mouse_train"],return_dict=True)
    epm_val_auc = model.score(epm_dict["X_val"],epm_y_val,
                              epm_dict["y_mouse_val"],return_dict=True)

    #OFT Performance
    oft_y_train = np.hstack([oft_dict["y_train"].reshape(-1,1),oft_dict["y_train"].reshape(-1,1)])
    oft_y_val = oft_dict["y_val"].reshape(-1,1)

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
    X_recon = model.reconstruct(mt_X_val)+ EPSILON
    perc_recon_list = []
    for i in range(n_sup_networks):
        X_sup_recon = model.get_comp_recon(torch.Tensor(s).to(model.device),i) + EPSILON
        net_recon_contribution = np.mean(X_sup_recon/X_recon,axis=0) 
        perc_recon_list.append(net_recon_contribution)

        rec_psd = net_recon_contribution[:len(anxInfo["powerFeatures"])]
        rec_coh = net_recon_contribution[len(anxInfo["powerFeatures"]):(len(anxInfo["powerFeatures"]) + len(anxInfo["cohFeatures"]))]
        rec_3d = reshapeData(rec_psd,rec_coh,8,56,anxInfo["powerFeatures"],anxInfo["cohFeatures"],anxInfo["area"])


        circle_plot(rec_3d,anxInfo["area"],freqs=np.arange(56),freq_ticks=np.arange(0,56,5),
                    min_max_quantiles=(0.85,0.9999),fn=plot_name.format(i,n_sup_networks,fold,"electome"))

        umc_dt.makeUpperTriangularPlot_pow_coh_gc(net_recon_contribution.reshape(1,-1),anxInfo["area"],anxInfo["powerFeatures"],
                                                  anxInfo["cohFeatures"],anxInfo["gcFeatures"],
                                                  saveFile=plot_name.format(i,n_sup_networks,fold,"up-tri"))

    results_dict = {
        "flx_train_auc":flx_train_auc,
        "flx_val_auc":flx_val_auc,
        "epm_train_auc":epm_train_auc,
        "epm_val_auc":epm_val_auc,
        "oft_train_auc":oft_train_auc,
        "oft_val_auc":oft_val_auc,
        "net_recon_list":perc_recon_list,
        "coefficients":model.classifier[0].weight[0].cpu().detach().numpy(),
        "mean_scores":np.mean(s[:,:n_sup_networks],axis=0)
    }

    with open(results_file,"wb") as f:
        pickle.dump(results_dict,f)