

import numpy as np
import torch
import torch.nn.functional as F

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    if type(y_pred) == torch.Tensor:
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            # mask = np.not_equal(y_true, null_val)
            mask = (y_true > null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    return np.sqrt(masked_mse_np(y_true=y_true,y_pred=y_pred, null_val=null_val))

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    if type(y_pred) == torch.Tensor:
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            # mask = np.not_equal(labels, null_val)
            mask = (y_true > null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(y_pred, y_true)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)

def masked_mae_np(y_true, y_pred, null_val=np.nan):
    if type(y_pred) == torch.Tensor:
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            # mask = np.not_equal(labels, null_val)
            mask = (y_true > null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(y_pred, y_true)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        # mask = (labels!=null_val)
        mask = (labels>null_val)
    
    mask = mask.float()
    mask /= torch.mean((mask))
    
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        # mask = (labels!=null_val)
        mask = (labels>null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        # mask = (labels!=null_val)
        mask = (labels>null_val)
    
    mask = mask.float()
    mask /=  torch.mean((mask))
    
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)*100


def All_Metrics(true,pred, mask1, mask2):
    #mask1 filter the very small value,
    # mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = masked_mae_np(true, pred, mask1)
        rmse = masked_rmse_np(true, pred, mask1)
        mape = masked_mape_np(true, pred, mask2)
    elif type(pred) == torch.Tensor:
        mae  = masked_mae(pred, true, mask1)
        rmse = masked_rmse(pred, true, mask1)
        mape = masked_mape(pred, true, mask2)
    else:
        raise TypeError
    return mae, rmse, mape


# init loss function
def masked_mse_loss(mask_value = 0.):
    def loss(preds, labels):
        mse = masked_mse(preds,labels,null_val=mask_value)
        return mse
    return loss

def masked_mae_loss(mask_value = 0.):
    def loss(preds, labels):
        mae = masked_mae(preds,labels,null_val=mask_value)
        return mae
    return loss

## loss_func = nn.MSELoss(reduction='mean')
def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss

if __name__ == '__main__':
    x = np.array([[1, 2, 3, 3],
                  [2, 0, 4, 4]])
    y = np.array([[0, 3, 4, 5],
                  [1, 2, 3, 4]])
    res = All_Metrics(y, x, 0, 1)
    print(res)