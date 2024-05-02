import torch
import numpy as np
import sklearn
from sklearn import datasets
import bbqdatasets

def mse_loss(y_pred,y_true,mask):
    loss = torch.pow(y_pred - y_true,2)
    loss = torch.mul(loss,mask)
    loss = torch.sum(loss)/torch.sum(mask)
    return loss

def load_data(data_name,tr_ratio,te_ratio,device,args):
    assert tr_ratio + te_ratio == 1.0

    if data_name in ['co2', 'airline', 'concrete']:
        if data_name == "co2":
            train_data, test_data = bbqdatasets.mauna_loa(rootDir=args.data_dir)
        elif data_name == "airline":
            train_data, test_data = bbqdatasets.airline_passengers(rootDir=args.data_dir)
        elif data_name == "concrete":
            train_data, test_data = bbqdatasets.concrete(rootDir=args.data_dir)

        tr_x = train_data[:, :-1]
        tr_y = train_data[:, [-1]]
        te_x = test_data[:, :-1]
        te_y = test_data[:, [-1]]

        data_x = np.concatenate((tr_x,te_x))
        data_y = np.concatenate((tr_y,te_y))

        data_x = torch.from_numpy(data_x).type(torch.FloatTensor).to(device)
        data_y = torch.from_numpy(data_y).type(torch.FloatTensor).to(device)

        # 90% / 10% split
        num_data, input_dim = data_x.shape
        num_data, output_dim = data_y.shape
        idx = np.arange(num_data)
        np.random.shuffle(idx)
        tr_idx = idx[:int(num_data * tr_ratio)]
        te_idx = idx[int(num_data * tr_ratio):]

    elif data_name == 'housing':
        data = sklearn.datasets.load_boston()
        data_x = data['data']
        data_y = data['target']

        max_x = 1; min_x = -1
        data_x_std = (data_x - data_x.min(axis=0)) / (data_x.max(axis=0) - data_x.min(axis=0))
        data_x = data_x_std * (max_x - min_x) + min_x
        max_y = 1; min_y = -1
        data_y_std = (data_y - data_y.min(axis=0)) / (data_y.max(axis=0) - data_y.min(axis=0))
        data_y = data_y_std * (max_y - min_y) + min_y

        data_x = torch.from_numpy(data_x).type(torch.FloatTensor).to(device)
        data_y = torch.from_numpy(data_y).type(torch.FloatTensor).to(device)
        data_y = torch.reshape(data_y,(-1,1))

        num_data, input_dim = data_x.shape
        num_data, output_dim = data_y.shape
        idx = np.arange(num_data)
        np.random.shuffle(idx)
        tr_idx = idx[:int(num_data * tr_ratio)]
        te_idx = idx[int(num_data * tr_ratio):]

    elif data_name == 'parkinsons':
        data_dir = 'data/ssdkldataset/' + data_name + '/'
        data_x = np.load(data_dir + 'X.npy')
        data_y = np.load(data_dir + 'y.npy')

        max_x = 1; min_x = -1
        data_x_std = (data_x - data_x.min(axis=0)) / (data_x.max(axis=0) - data_x.min(axis=0))
        data_x = data_x_std * (max_x - min_x) + min_x
        max_y = 1; min_y = -1
        data_y_std = (data_y - data_y.min(axis=0)) / (data_y.max(axis=0) - data_y.min(axis=0))
        data_y = data_y_std * (max_y - min_y) + min_y

        data_x = torch.from_numpy(data_x).type(torch.FloatTensor).to(device)
        data_y = torch.from_numpy(data_y).type(torch.FloatTensor).to(device)
        data_y = torch.reshape(data_y,(-1,1))
        num_data, input_dim = data_x.shape
        num_data, output_dim = data_y.shape
        idx = np.arange(num_data)
        np.random.shuffle(idx)
        tr_idx = idx[:int(num_data * tr_ratio)]
        te_idx = idx[int(num_data * tr_ratio):]

    tr_mask = np.zeros((num_data,1),dtype=np.float32)
    te_mask = np.zeros((num_data,1),dtype=np.float32)
    tr_mask[tr_idx] = 1
    te_mask[te_idx] = 1
    tr_mask = torch.from_numpy(tr_mask).to(device)
    te_mask = torch.from_numpy(te_mask).to(device)

    assert torch.sum(torch.mul(tr_mask,te_mask)) == 0

    tr_data = (data_x, data_y, tr_mask, tr_idx)
    te_data = (data_x, data_y, te_mask, te_idx)

    assert torch.sum(tr_mask) == len(tr_idx)
    assert torch.sum(te_mask) == len(te_idx)

    return tr_data,te_data,num_data,input_dim,output_dim