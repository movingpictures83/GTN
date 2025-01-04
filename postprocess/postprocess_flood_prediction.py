#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory) 
sys.path.append(parent_directory)


# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"] = "4"


# In[3]:


from preprocess.BaselinePrerocess import baseline_process, gcn_process, baseline_process_for_gate_predictor
from preprocess.GraphTransformerPrerocess import graph_water_transformer_cov_process
from preprocess.GraphTransformerPrerocess import graph_global_transformer_local_process
from preprocess.GraphTransformerPrerocess import graph_water_transformer_cov_process_for_gate_predictor
from preprocess.graph import graph_topology, graph_topology_5
from tensorflow.keras.models import load_model
from postprocess.threshold import flood_threshold, drought_threshold, flood_threshold_t1, drought_threshold_t1
from postprocess.errors import estimate_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from math import sqrt
from spektral.layers import GCNConv
from baselines.tcn import TCN
from preprocess.helper import series_to_supervised
import time


# ### Data Preprocessing

# In[4]:


# ====== preprocessing parameters ======
n_hours = 72
K = 24 
masked_value = 1e-10
split_1 = 0.8
split_2 = 0.9
sigma2 = 0.1
epsilon = 0.5


# In[5]:


train_X_mask, val_X_mask, test_X_mask, train_ws_y, val_ws_y, test_ws_y, scaler, ws_scaler = baseline_process(n_hours, K, masked_value, split_1, split_2)

print(train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, train_ws_y.shape, val_ws_y.shape, test_ws_y.shape)


# ## MAE and RMSE

# In[7]:


#saved_models = ['WaLeF_mlp', 'WaLeF_rnn', 'WaLeF_cnn', 'WaLeF_rcnn', 'WaLeF_transformer']
saved_models = ['WaLeF_transformer']


for i in range(len(saved_models)):
    print("===================== {} =====================".format(saved_models[i]))
    
    if saved_models[i] == 'WaLeF_transformer':
        # load model and make prediction
        model = load_model('../saved_models/{}.h5'.format(saved_models[i]))
        start_time = time.perf_counter()
        yhat = model.predict(test_X_mask)
        end_time = time.perf_counter()
        used_time = end_time - start_time
        print(f"Usded time: {used_time} seconds")
    else:
        # load model and make prediction
        start_time = time.perf_counter()
        model = load_model('../saved_models/{}.h5'.format(saved_models[i]))
        yhat = model.predict(test_X_mask)
        end_time = time.perf_counter()
        used_time = end_time - start_time
        print(f"Usded time: {used_time} seconds")

    # inverse transformation
    inv_yhat = ws_scaler.inverse_transform(yhat)
    inv_yhat = inv_yhat[:, [0, 24, 48, 72]]
    inv_y = ws_scaler.inverse_transform(test_ws_y)
    inv_y = inv_y[:, [0, 24, 48, 72]]
    
    print(inv_y.shape)
    # compute MAE and RMSE
    print('MAE = {}'.format(float("{:.4f}".format(mae(inv_y, inv_yhat)))))
    print('RMSE = {}'.format(float("{:.4f}".format(sqrt(mse(inv_y, inv_yhat))))))
    
    errors = inv_yhat - inv_y
    print('Numbers of over/under estimate:', estimate_error(errors))
    


# ### TCN

# In[9]:


saved_models = ['WaLeF_tcn']

for i in range(len(saved_models)):
    print("===================== {} =====================".format(saved_models[i]))
    
    # load model and make prediction
    model = load_model('../saved_models/{}.h5'.format(saved_models[i]), custom_objects={'TCN': TCN})
    start_time = time.perf_counter()
    yhat = model.predict(test_X_mask)
    end_time = time.perf_counter()
    used_time = end_time - start_time
    print(f"Usded time: {used_time} seconds")
        
    # inverse transformation
    inv_yhat = ws_scaler.inverse_transform(yhat)
    inv_yhat = inv_yhat[:, [0, 24, 48, 72]]
    inv_y = ws_scaler.inverse_transform(test_ws_y)
    inv_y = inv_y[:, [0, 24, 48, 72]]

    # compute MAE and RMSE
    print('MAE = {}'.format(float("{:.4f}".format(mae(inv_y, inv_yhat)))))
    print('RMSE = {}'.format(float("{:.4f}".format(sqrt(mse(inv_y, inv_yhat))))))
    
    errors = inv_yhat - inv_y
    print('Numbers of over/under estimate:', estimate_error(errors))


# ### GCN

# In[10]:


train_X_mask_gcn, val_X_mask_gcn, test_X_mask_gcn, train_ws_y_gcn, val_ws_y_gcn, test_ws_y_gcn, scaler, ws_scaler = gcn_process(n_hours, K, masked_value, split_1, split_2)
print(train_X_mask_gcn.shape, val_X_mask_gcn.shape, test_X_mask_gcn.shape, train_ws_y_gcn.shape, val_ws_y_gcn.shape, test_ws_y_gcn.shape)


train_adj_mat, val_adj_mat, test_adj_mat = graph_topology(n_hours, K, sigma2, epsilon, len(train_ws_y), len(val_ws_y), len(test_ws_y))
print(train_adj_mat.shape, val_adj_mat.shape, test_adj_mat.shape)


# In[11]:


saved_models = ['WaLeF_gcn']

for i in range(len(saved_models)):
    print("===================== {} =====================".format(saved_models[i]))
    
    # load model and make prediction
    model = load_model('../saved_models/{}.h5'.format(saved_models[i]), custom_objects={'GCNConv': GCNConv})
    start_time = time.perf_counter()
    yhat = model.predict([test_X_mask_gcn, test_adj_mat])
    end_time = time.perf_counter()
    used_time = end_time - start_time
    print(f"Usded time: {used_time} seconds")

    # inverse transformation
    inv_yhat = ws_scaler.inverse_transform(yhat)
    inv_yhat = inv_yhat[:, [0, 24, 48, 72]]
    inv_y = ws_scaler.inverse_transform(test_ws_y)
    inv_y = inv_y[:, [0, 24, 48, 72]]

    # compute MAE and RMSE
    print('MAE = {}'.format(float("{:.4f}".format(mae(inv_y, inv_yhat)))))
    print('RMSE = {}'.format(float("{:.4f}".format(sqrt(mse(inv_y, inv_yhat))))))
    
    errors = inv_yhat - inv_y
    print('Numbers of over/under estimate:', estimate_error(errors))


# ### FloodGTN

# In[12]:


train_cov, val_cov, test_cov, train_tws_reshape, val_tws_reshape, test_tws_reshape, train_ws_y, val_ws_y, test_ws_y, scaler, ws_scaler = graph_water_transformer_cov_process(n_hours, K, masked_value, split_1, split_2)


# In[13]:


train_adj_mat, val_adj_mat, test_adj_mat = graph_topology_5(n_hours, K, sigma2, epsilon, len(train_ws_y), len(val_ws_y), len(test_ws_y))


# In[14]:


saved_models = ['WaLeF_gtn_p']

for i in range(len(saved_models)):
    print("===================== {} =====================".format(saved_models[i]))
    
    # load model and make prediction
    #model = load_model('../saved_models/{}.h5'.format(saved_models[i]), custom_objects={'GCNConv': GCNConv})
    model = load_model("../saved_models/"+saved_models[i]+".h5", custom_objects={'GCNConv': GCNConv})
    start_time = time.perf_counter()
    yhat = model.predict([test_cov, test_tws_reshape, test_adj_mat])
    end_time = time.perf_counter()
    used_time = end_time - start_time
    print(f"Usded time: {used_time} seconds")
    # inverse transformatio
    inv_yhat = ws_scaler.inverse_transform(yhat)
    inv_yhat = inv_yhat[:, [0, 24, 48, 72]]
    inv_y = ws_scaler.inverse_transform(test_ws_y)
    inv_y = inv_y[:, [0, 24, 48, 72]]
    # compute MAE and RMSE
    print('MAE = {}'.format(float("{:.4f}".format(mae(inv_y, inv_yhat)))))
    print('RMSE = {}'.format(float("{:.4f}".format(sqrt(mse(inv_y, inv_yhat))))))
    
    errors = inv_yhat - inv_y
    print('Numbers of over/under estimate:', estimate_error(errors))


# ## Flood time steps and areas

# ### Ground-truth water levels and flooding time steps and flooding areas
# 

# In[6]:


upper_threshold = 3.5
lower_threshold = 0
t1 = 1

inv_y = ws_scaler.inverse_transform(test_ws_y)
inv_y_reshape = inv_y.reshape((-1, 24, 4))

print("Over thresholds:", flood_threshold_t1(inv_y_reshape, t1, upper_threshold))
print("Under thresholds:", drought_threshold_t1(inv_y_reshape, t1, lower_threshold))


# In[7]:


train_X_mask, val_X_mask, test_X_mask, train_gate_pump_y, val_gate_pump_y, test_gate_pump_y, train_ws_y, val_ws_y, test_ws_y, scaler, gate_pump_scaler, ws_scaler = baseline_process_for_gate_predictor(n_hours, K, masked_value, split_1, split_2)


# In[8]:


#saved_models = ['WaLeF_mlp', 'WaLeF_rnn', 'WaLeF_cnn', 'WaLeF_rcnn', 'WaLeF_transformer']
saved_models = ['WaLeF_transformer']


for i in range(len(saved_models)):
    print("===================== {} =====================".format(saved_models[i]))
    
    if saved_models[i] == 'WaLeF_transformer':
        # load model and make prediction
        model = load_model('../saved_models/{}.h5'.format(saved_models[i]))
        yhat = model.predict(test_X_mask)
    else:
        # load model and make prediction
        model = load_model('../saved_models/{}.h5'.format(saved_models[i]))
        yhat = model(test_X_mask)

    # inverse transformation
    inv_yhat = ws_scaler.inverse_transform(yhat)
    inv_y = ws_scaler.inverse_transform(test_ws_y)
    
    # compute time steps and areas over and under thresholds
    inv_yhat_reshape = inv_yhat.reshape((-1, 24, 4))
    inv_y_reshape = inv_y.reshape((-1, 24, 4))
    
    print("Over thresholds:", flood_threshold_t1(inv_yhat_reshape, t1, upper_threshold))
    print("Under thresholds:", drought_threshold_t1(inv_yhat_reshape, t1, lower_threshold))


# In[9]:


saved_models = ['WaLeF_tcn']

for i in range(len(saved_models)):
    print("===================== {} =====================".format(saved_models[i]))
    
    # load model and make prediction
    model = load_model('../saved_models/{}.h5'.format(saved_models[i]), custom_objects={'TCN': TCN})
    yhat = model(test_X_mask)

    # inverse transformation
    inv_yhat = ws_scaler.inverse_transform(yhat)
    inv_y = ws_scaler.inverse_transform(test_ws_y)
    
    # compute time steps and areas over and under thresholds
    inv_yhat_reshape = inv_yhat.reshape((-1, 24, 4))
    inv_y_reshape = inv_y.reshape((-1, 24, 4))
    
    print("Over thresholds:", flood_threshold_t1(inv_yhat_reshape, t1, upper_threshold))
    print("Under thresholds:", drought_threshold_t1(inv_yhat_reshape, t1, lower_threshold))


# ### GCN

# In[10]:


train_X_mask_gcn, val_X_mask_gcn, test_X_mask_gcn, train_ws_y_gcn, val_ws_y_gcn, test_ws_y_gcn, scaler, ws_scaler = gcn_process(n_hours, K, masked_value, split_1, split_2)
print(train_X_mask_gcn.shape, val_X_mask_gcn.shape, test_X_mask_gcn.shape, train_ws_y_gcn.shape, val_ws_y_gcn.shape, test_ws_y_gcn.shape)


train_adj_mat, val_adj_mat, test_adj_mat = graph_topology(n_hours, K, sigma2, epsilon, len(train_ws_y), len(val_ws_y), len(test_ws_y))
print(train_adj_mat.shape, val_adj_mat.shape, test_adj_mat.shape)


# In[11]:


saved_models = ['WaLeF_gcn']

for i in range(len(saved_models)):
    print("===================== {} =====================".format(saved_models[i]))
    
    # load model and make prediction
    model = load_model('../saved_models/{}.h5'.format(saved_models[i]), custom_objects={'GCNConv': GCNConv})
    yhat = model([test_X_mask_gcn, test_adj_mat])

    # inverse transformation
    inv_yhat = ws_scaler.inverse_transform(yhat)
    inv_y = ws_scaler.inverse_transform(test_ws_y)

    # compute time steps and areas over and under thresholds
    inv_yhat_reshape = inv_yhat.reshape((-1, 24, 4))
    inv_y_reshape = inv_y.reshape((-1, 24, 4))
    
    print("Over thresholds:", flood_threshold_t1(inv_yhat_reshape, t1, upper_threshold))
    print("Under thresholds:", drought_threshold_t1(inv_yhat_reshape, t1, lower_threshold))


# ### FloodGTN-Parallel

# In[12]:


train_cov, val_cov, test_cov, train_tws_reshape, val_tws_reshape, test_tws_reshape, train_gate_pump_y, val_gate_pump_y, test_gate_pump_y, train_ws_y, val_ws_y, test_ws_y, scaler, ws_scaler, gate_scalar = graph_water_transformer_cov_process_for_gate_predictor(n_hours, K, masked_value, split_1, split_2)

train_adj_mat, val_adj_mat, test_adj_mat = graph_topology_5(n_hours, K, sigma2, epsilon, len(train_ws_y), len(val_ws_y), len(test_ws_y))


# In[13]:


saved_models = ['WaLeF_gtn_p']

for i in range(len(saved_models)):
    print("===================== {} =====================".format(saved_models[i]))
    
    # load model and make prediction
    model = load_model('../saved_models/{}.h5'.format(saved_models[i]), custom_objects={'GCNConv': GCNConv}
                      )
    yhat = model.predict([test_cov, test_tws_reshape, test_adj_mat])

    # inverse transformation
    inv_yhat = ws_scaler.inverse_transform(yhat)
    inv_y = ws_scaler.inverse_transform(test_ws_y)
    
    # compute time steps and areas over and under thresholds
    inv_yhat_reshape = inv_yhat.reshape((-1, 24, 4))
    inv_y_reshape = inv_y.reshape((-1, 24, 4))
    
    print("Over thresholds:", flood_threshold_t1(inv_yhat_reshape, t1, upper_threshold))
    print("Under thresholds:", drought_threshold_t1(inv_yhat_reshape, t1, lower_threshold))


# ### HEC-RAS

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from preprocess.helper import stage_series_to_supervised, series_to_supervised


# In[54]:


error_ras = pd.read_csv('../data/ras_1920.csv', index_col=0)
error_ras.reset_index(drop=True, inplace=True)
error_ras


# In[55]:


ras_concat = pd.concat([error_ras.loc[:, 'S1_RAS'], error_ras.loc[:, 'S25A_TW_RAS'], error_ras.loc[:, 'S25B_TW_OBS'], error_ras.loc[:, 'S26_TW_RAS']], axis=1)
ras_concat = pd.DataFrame(ras_concat)
ras_concat


# In[56]:


plt.rcParams["figure.figsize"] = (18, 6)
plt.plot(ras_concat.iloc[:, 0], label='S1')
plt.axhline(y = 3.5, color='r', linestyle='dashed', linewidth=2)
plt.axhline(y = 0, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Time', fontsize=18)
plt.ylabel('S1 Water Stage', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()


# In[50]:


#ras_concat['Timestamp'] = test_set_s1['Time']
#ras_concat


# In[51]:


ras_flood = ras_concat[ras_concat['S1_RAS'] > 3.5]
ras_flood


# In[38]:


plt.rcParams["figure.figsize"] = (18, 6)
plt.plot(ras_concat.iloc[5873:5913, 0], label='S1')
plt.axhline(y = 3.5, color='r', linestyle='dashed', linewidth=2)
plt.axhline(y = 0, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Time', fontsize=18)
plt.ylabel('S1 Water Stage', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()


# #### Test_set

# In[39]:


test_set = pd.read_csv('../data/test_data.csv')
test_set.columns


# In[40]:


# test_set_s1 = test_set.iloc[-17544:, 0:2]
test_set_s1 = test_set.iloc[:, 0:2]
test_set_s1.reset_index(drop=True, inplace=True)
test_set_s1


# In[41]:


filter_flood = test_set_s1[test_set_s1['WS_S1'] > 3.5]
filter_flood


# In[42]:


plt.rcParams["figure.figsize"] = (18, 6)
plt.plot(test_set_s1['WS_S1'], label='S1')
plt.axhline(y = 3.5, color='r', linestyle='dashed', linewidth=2)
plt.axhline(y = 0, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Time', fontsize=18)
plt.ylabel('S1 Water Stage', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()


# In[43]:


filter_flood.index


# In[44]:


plt.rcParams["figure.figsize"] = (18, 6)
plt.plot(test_set_s1.iloc[7640:7680, 1], label='S1')
plt.axhline(y = 3.5, color='r', linestyle='dashed', linewidth=2)
plt.axhline(y = 0, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Time', fontsize=18)
plt.ylabel('S1 Water Stage', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()


# In[45]:


test_set_s1.iloc[7640:7680, 0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[57]:


n_hours = 72
K = 24


# In[58]:


data_supervised = series_to_supervised(ras_concat, n_hours, K)
data_supervised


# In[59]:


col_names = ['S1_RAS', 'S25A_TW_RAS', 'S25B_TW_OBS', 'S26_TW_RAS'] * (n_hours+K)
    
data_supervised.reset_index(drop=True, inplace=True)
data_supervised.columns = [[i + '_' + j for i, j in zip(col_names, list(data_supervised.columns))]]
data_supervised


# In[60]:


data_supervised_array = data_supervised.to_numpy(dtype='float32')
data_supervised_array = data_supervised_array.reshape((-1, 96, 4))
data_supervised_array = data_supervised_array[:, -24:, :]
data_supervised_array.shape


# In[61]:


upper_threshold = 3.5

flood_threshold_t1(data_supervised_array, t1, upper_threshold)


# In[62]:


lower_threshold = 0

drought_threshold_t1(data_supervised_array, t1, lower_threshold)


# In[ ]:





# In[ ]:




