import torch
import numpy as np
from utils import mse_loss

def train(model, data, optimizer):
    model.train()

    x, y, mask, idx = data
    optimizer.zero_grad()

    output,KLD = model(x)
    output = output.squeeze(0)
    loss = mse_loss(output,y,mask)
    loss = loss + KLD/torch.sum(mask)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
    optimizer.step()
    mse = loss.item()
    rmse = np.sqrt(mse)
    return output, mse, rmse

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        x, y, mask, idx = data
        output,_ = model(x)
        output = output.squeeze(0)
        loss = mse_loss(output,y,mask)
        mse = loss.item()
        rmse = np.sqrt(mse)

    return output, mse, rmse