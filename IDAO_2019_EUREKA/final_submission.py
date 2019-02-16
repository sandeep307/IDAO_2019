import pandas as pd
import numpy as np
import utils, scoring

import catboost as cb
import lightgbm as lgb
import glob, os
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from torch.nn.functional import relu
import torch
os.environ["CUDA_VISIBLE_DEVICES"]= "1" 

# loading data

test_df = pd.read_hdf('../data/test_private_v2_track_1.hdf')
test_hit = pd.read_csv('../data/private_test_closest_hits_features.csv')

test_df = pd.concat([test_df, test_hit], axis=1)
test_df.drop('Unnamed: 0', axis = 1, inplace=True)

test_df.set_index('id', inplace=True)

print ('Loading lgb model')
MODEL_NAME = "../models/track1_lgb_90perc_7886_n305.lgb"
model = lgb.Booster(model_file=MODEL_NAME)

test_pred_lgb = model.predict(test_df.values, num_iteration = model.best_iteration)
lgb_submit = pd.DataFrame(data={"prediction":  test_pred_lgb}, index=test_df.index)
# lgb_submit.to_csv( "lgb_final_test_private.csv", index_label=utils.ID_COLUMN)


## catboost model
print ('Loading catboost model')
cb_model = cb.CatBoostClassifier()
cb_model.load_model(fname = '../models/cbm_with_close_feats.cbm')

test_pred_cb = cb_model.predict_proba(test_df.values)[:,1]
cb_submit = pd.DataFrame(data={"prediction":  test_pred_cb}, index=test_df.index)
# cb_submit.to_csv( "cb_final_test_private.csv", index_label=utils.ID_COLUMN) 


## loading NN model
print ('Loading NN model')
def rank_gauss(x):
    from scipy.special import erfinv
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x


type_cols = [x for x in test_df.columns if 'TYPE' in x]    

test_df_NN = test_df.copy()

for i, col in enumerate(test_df_NN.drop(type_cols, axis = 1)):
    test_df_NN[col] = rank_gauss(test_df_NN[col].values)

    
class MyModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 4) 
        self.fc4 = torch.nn.Linear(4, 1) 
            
    def forward(self, x):
        fc1_op = relu(self.fc1(x))
        fc2_op = relu(self.fc2(fc1_op))
        fc3_op = relu(self.fc3(fc2_op))
        return self.fc4(fc3_op)

def predict(x, model):
    x = torch.from_numpy(x.values).float().cuda()
    return torch.sigmoid(model(x)).cpu().detach().numpy()[:, 0]

model_nn = MyModel(test_df_NN.shape[1]).cuda()
    
pretrained_model_file = '../models/NN_models_3_32init/epoch_51_ckpt.pth.tar'
checkpoint = torch.load(pretrained_model_file)
model_nn.load_state_dict(checkpoint['state_dict'])

use_gpu = True
if use_gpu:
    model_nn = model_nn.cuda()
    
model_nn.eval()
test_pred_NN = predict(test_df_NN, model_nn)  
nn_submit = pd.DataFrame(data={"prediction":  test_pred_NN}, index=test_df.index)
# nn_submit.to_csv( "nn_final_test_private.csv", index_label=utils.ID_COLUMN)


#ensembling
all_mean = pd.concat([lgb_submit, cb_submit, nn_submit], axis=1, ignore_index=True)
print (all_mean.shape)
print (all_mean.head() )
lgb50_cb_nn = (all_mean*[.5, .25, .25]).sum(axis=1)
print(lgb50_cb_nn.describe())

lgb50_cb_nn.index.name='id'
lgb50_cb_nn.to_frame('prediction').to_csv('final_test_lgb50_cb_nn_private.csv')
