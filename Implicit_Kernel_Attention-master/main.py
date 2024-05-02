import torch
import numpy as np
import random
import argparse
import os
import logging
from baseTransformer import baseTransformer
from solver import train, evaluate
from utils import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=1000, help='seed')
parser.add_argument('--att_dropout', type=float, default=0.1, help="probability of an element to be zero")
parser.add_argument('--prior_var', type=float, default=0.1, help="prior for mikan")
parser.add_argument('--ENC_PF_DIM', type=int, default=64, help='The dimensionality of feed forward')
parser.add_argument('--HID_DIM', type=int, default=64, help='The dimensionality of feed forward')
parser.add_argument('--KEY_DIM', type=int, default=64, help='The dimensionality key query')
parser.add_argument('--ENC_HEADS', type=int, default=8, help='number of heads')
parser.add_argument('--ENC_LAYERS', type=int, default=1, help='number of layers')
parser.add_argument('--ENC_DROPOUT', type=float, default=0.0, help='dropout')
parser.add_argument('--LEARNING_RATE', type=float, default=0.01, help='lr')
parser.add_argument('--kl_lambda', type=float, default=0.1, help='kl loss lambda (weight)')
parser.add_argument('--copula_lambda', type=float, default=1.0, help='copula loss lambda')
parser.add_argument('--N_EPOCHS', type=int, default=1000, help='epoch')
parser.add_argument('--tr_ratio', type=float, default=0.9, help='train ratio')
parser.add_argument('--te_ratio', type=float, default=0.1, help='test ratio')
parser.add_argument('--log_dir', type=str, default='log/', help='logger directory')
parser.add_argument('--model', type=str, default='base')
parser.add_argument('--data', type=str,choices=['co2','airline','housing','concrete','parkinsons'], default='housing')
parser.add_argument('--data_dir', type=str, default='data/bbqdataset/', help='data directory')
parser.add_argument('--att_type', type=str, choices=['dot','ikandirect','mikan'], default='mikan')
parser.add_argument('--p_norm', type=float, default=2.0, help='L-p norm in magnitude term')
parser.add_argument('--M', type=int, default=64, help='num_samples')
args = parser.parse_args()

repeat = 10
''' dir make '''
model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.data,args.tr_ratio,args.att_type,args.M, args.p_norm)
dir_list = [args.log_dir]
for dir in dir_list:
    if not os.path.exists(dir):
        os.makedirs(dir)

''' random seed fix '''
np.random.seed(args.random_seed)
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

''' logger '''
PATH_LOG = args.log_dir + model_name + '.txt'
logger = logging.getLogger('Result_log')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(PATH_LOG)
logger.addHandler(file_handler)
logger.info("==" * 10)
for param in vars(args).keys():
    s = '--{0} : {1}'.format(param, vars(args)[param])
    logger.info(s)
logger.info("==" * 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_rmse_list = []
''' repeat 10 times '''
for random_seed in range(args.random_seed,args.random_seed+repeat):
    tr_data,te_data,num_data,input_dim,output_dim = \
        load_data(args.data,args.tr_ratio,args.te_ratio,device,args)

    model = baseTransformer(args,num_data,input_dim, output_dim, args.HID_DIM, args.ENC_LAYERS,
                        args.ENC_HEADS, args.ENC_PF_DIM, args.ENC_DROPOUT,device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.LEARNING_RATE)

    best_test_loss = float('inf')
    for epoch in range(args.N_EPOCHS):
        tr_pred, tr_mse, tr_rmse = train(model, tr_data, optimizer)
        te_pred, te_mse, te_rmse = evaluate(model, te_data)

        if te_rmse < best_test_loss:
            best_test_loss = te_rmse
        s = (f'Epoch: {epoch + 1:02} | '
             f'Train Loss: {tr_rmse:.4f} | Test Loss: {te_rmse:.4f}')
        logger.info(s)
    best_rmse_list.append(best_test_loss)

overall_mean = np.mean(best_rmse_list)
overall_std = np.std(best_rmse_list)
with open("overview.txt", "a") as f:
    f.write(args.data)
    f.write("|")
    f.write(args.att_type)
    f.write("|")
    f.write(str(overall_mean))
    f.write("|")
    f.write(str(overall_std))
    f.write("|")
    f.write(" ".join(str(item) for item in best_rmse_list))
    f.write("|")
    f.write(model_name)
    f.write("\n")