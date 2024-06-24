# Universal-Mouse-Code
Private workspace for Universal Mouse Code development including work for missing data handling, domain adaptation, multitask learning

As an example of how to train the model

```
import sys
lpne_path = '{file path to lpne-data-analysis}'
data_path = '{path to data}'
sys.path.append(lpne_path)
sys.path.append(data_path)

from dCSFA_model import dCSFA_model
import umc_data_tools as umc_dt
import data_tools
import torch

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'

#time to train
N_EPOCHS=25
freqs = (1,56)
feature_list = ['power','directedSpectrum']
psd, ds, labels = data_tools.load_data(data_path,feature_list=feature_list,freqs=freqs)

X = np.hstack([psd,ds])
y = np.array(labels['windows']['task'])
y_mouse = np.array(labels['windows']['mouse'])

#80/20 train test split by mouse
X_train, X_test, y_train, y_test, mouse_train, mouse_test = umc_dt.lpne_train_test_split(X,y,y_mouse)

#Number of feature types
n_featuers = int(X.shape[1] / freqs[1])

model = dCSFA_model(n_freqs=freqs[1],n_features=n_features,n_components=20,
                    model_type='dCSFA_NMF',recon_l='IS',device=device)
                    
model.fit(X_train,y_train,batch_size=128,n_epochs=N_EPOCHS,verbose=False)

#save the model
model_save_name = "What you want to name your model"
torch.save(model,model_name)

#save the supervised network
network_save_name = "what you want to save the network as"
network = model.components.detach().numpy()
np.savetxt(network_save_name,network,delimiter=',')

#Get test AUC dictionary
y_pred_test,z = model.transform(X_test)[2:]
y_pred_test = y_pred_test > 0.5
auc_dict = umc_dt.lpne_auc(y_pred_test,y_test,mouse_test,z,mannWhitneyU=True)
mean_auc = np.mean([auc_dict[mouse][0] for mouse in np.unique(mouse_test)])
print(auc_dict)
print(mean_auc)

```
