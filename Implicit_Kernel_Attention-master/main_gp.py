import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import os
import numpy as np
import bbqdatasets
import argparse
import sklearn
import random
import logging
from utils import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=1000, help='seed')
parser.add_argument('--N_EPOCHS', type=int, default=1000, help='epoch')
parser.add_argument('--num_mixture', type=int, default=1, help='num mixtures')
parser.add_argument('--tr_ratio', type=float, default=0.9, help='train ratio')
parser.add_argument('--te_ratio', type=float, default=0.1, help='test ratio')
parser.add_argument('--log_dir', type=str, default='log/', help='logger directory')
parser.add_argument('--data', type=str,choices=['co2','airline','housing','concrete','parkinsons'], default='co2')
parser.add_argument('--kernel_type', type=str, choices=['rbf','sm'], default='rbf')
parser.add_argument('--data_dir', type=str, default='data/bbqdataset/', help='data directory')
parser.add_argument('--model_save_dir', type=str, default='model_save/', help='model save')
args = parser.parse_args()

repeat = 10
''' dir make '''
model_name = 'GP_{0}_{1}_{2}'.format(args.data,args.kernel_type,args.num_mixture)
dir_list = [args.log_dir,args.model_save_dir]
for dir in dir_list:
    if not os.path.exists(dir):
        os.makedirs(dir)
args.model_save_dir = args.model_save_dir + model_name

''' random seed fix '''
np.random.seed(args.random_seed)
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args.te_ratio = 1 - args.tr_ratio

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
# device = 'cpu'

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_type):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel_type == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_type == 'sm':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.SpectralMixtureKernel(num_mixtures=args.num_mixture, ard_num_dims=n_dim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

best_rmse_list = []

for random_seed in range(args.random_seed,args.random_seed+repeat):
    tr_data,te_data,num_data,input_dim,output_dim = \
        load_data(args.data,args.tr_ratio,args.te_ratio,device,args)

    tr_x, tr_y, tr_mask, tr_idx = tr_data
    te_x, te_y, te_mask, te_idx = te_data

    tr_x = tr_x[tr_idx,:].to(device)
    tr_y = tr_y[tr_idx].squeeze(-1).to(device)
    te_x = te_x[te_idx,:].to(device)
    te_y = te_y[te_idx].squeeze(-1).to(device)
    n_dim = tr_x.size()[-1]
    del tr_mask, tr_idx, tr_data, te_mask, te_idx, te_data

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(tr_x, tr_y, likelihood, kernel_type=args.kernel_type).to(device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    best_rmse = float('inf')
    for epoch in range(args.N_EPOCHS):
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        train_output = model(tr_x)

        train_loss = -1*mll(train_output, tr_y)
        train_loss.backward()
        optimizer.step()

        model.eval()
        likelihood.eval()
        test_output = model(te_x)
        test_output = test_output.loc

        mse = torch.pow(test_output - te_y,2)
        mse = torch.mean(mse)
        mse = mse.item()
        rmse = np.sqrt(mse)

        if best_rmse > rmse:
            best_rmse = rmse

        s = (f'Epoch: {epoch + 1:02} | '
             f'Train Loss: {train_loss:.4f} | Test Loss: {rmse:.4f}')
        logger.info(s)

    best_rmse_list.append(best_rmse)

overall_mean = np.mean(best_rmse_list)
overall_std = np.std(best_rmse_list)
with open("overview.txt", "a") as f:
    f.write(args.data)
    f.write("|")
    f.write(args.kernel_type)
    f.write("|")
    f.write(str(overall_mean))
    f.write("|")
    f.write(str(overall_std))
    f.write("|")
    f.write(" ".join(str(item) for item in best_rmse_list))
    f.write("|")
    f.write(model_name)
    f.write("\n")