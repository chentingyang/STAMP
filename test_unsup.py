
import  os
import argparse

import numpy as np

from lib.evaluate import *

parser = argparse.ArgumentParser(description='PyTorch Prediction Model on Time-series Dataset')
parser.add_argument('--data', type=str, default='SWaT',
                    help='type of the dataset (SWaT, WADI, ...)')
parser.add_argument('--filename', type=str, default='SWaT_Dataset_Normal_v1.csv',
                    help='filename of the dataset')
parser.add_argument('--debug', default=False, type=eval)
parser.add_argument('--real_value', default=False, type=eval)
parser.add_argument('--log_dir', default="expe", type=str)
parser.add_argument('--model', default="v2_", type=str)## ourModel
parser.add_argument('--pred_model', default="gat", type=str)
parser.add_argument('--gpu_id', default="0", type=str)
parser.add_argument('--temp_method', default="SAttn", type=str)


### graph constructure
parser.add_argument('--nnodes', type=int, default=45, help='number of nodes')
parser.add_argument('--top_k', type=int, default=10, help='top-k')
parser.add_argument('--em_dim', type=int, default=32, help='embedding dimension')
parser.add_argument('--alpha', type=int, default=3, help='alpha')
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden_dim')
parser.add_argument('--att_option', type=int, default=1, help='att_option')

### pred model
parser.add_argument('--window_size', type=int, default=15, help='window_size')
parser.add_argument('--n_pred', type=int, default=3, help='n_pred')
parser.add_argument('--temp_kernel', type=int, default=5, help='temp_kernel')
parser.add_argument('--in_channels', type=int, default=1, help='in_channels')
parser.add_argument('--out_channels', type=int, default=1, help='out_channels')

parser.add_argument('--layer_num', type=int, default=2, help='layer_num')
parser.add_argument('--act_func', type=str, default="GLU", help='act_func')
parser.add_argument('--pred_lr_init', type=float, default=0.001, help='pred_lr_init')

### Attention
parser.add_argument('--embed_size', type=int, default=64, help='embed_size')
parser.add_argument('--num_heads', type=int, default=8, help='num_heads')
parser.add_argument('--num_layers', type=int, default=1, help='num_attn_layers')
parser.add_argument('--ffwd_size', type=int, default=32, help='feed_foward_layer_size')
parser.add_argument('--is_conv', type=eval, default=False)
parser.add_argument('--return_weight', type=eval, default=False)

### AE
parser.add_argument('--latent_size', type=int, default=1, help='latent_size')
parser.add_argument('--ae_lr_init', type=float, default=0.001, help='ae_lr_init')

parser.add_argument('--seed', type=int, default=666, help='seed')
parser.add_argument('--val_ratio', type=float, default=0.2, help='val_ratio')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=30, help='number of epoch')
parser.add_argument('--test_alpha', type=float, default=.5, help='test_alpha')
parser.add_argument('--test_beta', type=float, default=.0, help='test_beta')
parser.add_argument('--test_gamma', type=float, default=0.5, help='test_gamma')
parser.add_argument('--is_down_sample', type=eval, default=True, help='is_down_sample')
parser.add_argument('--down_len', type=int, default=100, help='down_len')

parser.add_argument('--early_stop', default=True, type=eval)
parser.add_argument('--early_stop_patience', type=int, default=10, help='early_stop_patience')

parser.add_argument('--lr_decay', default=True, type=eval)
parser.add_argument('--lr_decay_rate', default=0.5, type=float)
parser.add_argument('--lr_decay_step', default="5,20,40,70", type=str)
parser.add_argument('--search_steps', default=50, type=int)
parser.add_argument('--is_mas', default=True, type=eval)
args = parser.parse_args()

args.model = args.model + args.pred_model

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from model.net import *
from trainer import Tester
from lib.logger import get_logger
from lib.dataloader_smd import load_data,load_data2,load_data3,load_data_unsup_train
from lib.utils import *
from lib.metrics import *
from model.utils import *


DEVICE = get_default_device()

args.device = DEVICE

base_dir = os.getcwd()

smd_unsup_data = np.load("...npz")
attack_train = smd_unsup_data['a']
train_labels = smd_unsup_data['b']
attack_test = smd_unsup_data['c']
test_labels = smd_unsup_data['d']
print(len(test_labels))


_, _, test_loader, y_test_labels, _ = load_data3(attack_train, attack_test, test_labels,
                                                                        device=DEVICE,
                                                                        window_size=args.window_size,
                                                                        val_ratio=args.val_ratio,
                                                                        batch_size=args.batch_size,
                                                                        is_down_sample=args.is_down_sample,
                                                                        down_len=args.down_len)

train_loader, val_loader, min_max_scaler = load_data_unsup_train(attack_train, train_labels,
                                                                    device=DEVICE,
                                                                    window_size=args.window_size,
                                                                    val_ratio=0.05,
                                                                    batch_size=args.batch_size,
                                                                    is_down_sample=args.is_down_sample,
                                                                    down_len=args.down_len)

## set seed
init_seed(args.seed)

channels_list = [[16,8,32],[32,8,64]]
# channels_list = [[32,16,64],[64,16,64]]
# channels_list = [[16,8,16],[16,8,32],[32,8,64]]

AE_IN_CHANNELS = args.window_size * args.nnodes * args.in_channels
latent_size = args.window_size * args.latent_size

pred_model = STATModel(args, DEVICE, args.window_size - args.n_pred, channels_list, static_feat=None)
# pred_model = to_device(pred_model, DEVICE)


ae_model = EncoderDecoder(AE_IN_CHANNELS, latent_size, AE_IN_CHANNELS, not args.real_value)
# ae_model = to_device(ae_model, DEVICE)


logger = get_logger(args.log_dir, name=args.model, debug=args.debug, data = args.data)

model_path = "./expe/"+'best_model_unsup_weights_init_' + args.data + "_" + args.model + '.pth'
tester = Tester(pred_model, ae_model, args, min_max_scaler, logger, path = model_path, alpha=args.test_alpha, beta=args.test_beta, gamma= args.test_gamma)

map_location = torch.device(DEVICE)
# map_location = lambda storage.cuda(0), loc: storage
##[val_gt_list, val_pred_list, val_construct_list]

def get_train_results(train_loader, map_location):# for POT computing, adopt the trained STAMP to compute the errors in the training set 
    train_results = tester.testing(train_loader, map_location)
    train_y_pred, train_loss1_list, train_loss2_list, train_pred_list, train_gt_list, train_origin_list, train_construct_list, train_generate_list, train_generate_construct_list = concate_results(train_results)
    
    train_pred_results = [train_pred_list, train_gt_list]

    train_ae_results = [train_construct_list, train_origin_list]

    train_generate_results = [train_generate_list, train_generate_construct_list]
    
    return train_pred_results, train_ae_results, train_generate_results

train_pred_results, train_ae_results, train_generate_results = get_train_results(train_loader, map_location)
    

test_results = tester.testing(test_loader, map_location)


test_y_pred, test_loss1_list, test_loss2_list, test_pred_list, test_gt_list, test_origin_list, test_construct_list, test_generate_list, test_generate_construct_list = concate_results(test_results)


print("scores: ", len(test_y_pred), test_y_pred.mean())
print("loss1: ", len(test_loss1_list), test_loss1_list.mean())
print("loss2: ", len(test_loss2_list), test_loss2_list.mean())
print("y_pred: ", len(test_y_pred))
print("y_test_labels: ", len(y_test_labels))


start = test_y_pred.min()
end = test_y_pred.max()


test_pred_results = [test_pred_list, test_gt_list]

test_ae_results = [test_construct_list, test_origin_list]

test_generate_results = [test_generate_list, test_generate_construct_list]


"""
1. option: Normalization 
    if option == 1:
        n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)
    elif option == 2:
        n_err_mid, n_err_iqr = get_err_mean_and_std(test_predict, test_gt)
    elif option == 3:
        n_err_mid, n_err_iqr = get_err_mean_and_quantile(test_predict, test_gt, 0.25)
    else:
        n_err_mid, n_err_iqr = get_err_median_and_quantile(test_predict, test_gt, 0.25)
        
2. method: [mean, max, sum]
    if method in ["max", "MAX"]:
        return np.array(test_ae_scores).max(axis=0)
    elif method in ["mean", "MEAN"]:
        return np.array(test_ae_scores).mean(axis=0)
    else:
        return np.array(test_ae_scores).sum(axis=0)
"""

print("================= Find best f1 from score: normalization loss ... =================")
info = get_final_result_POT(test_pred_results, test_ae_results, test_generate_results, train_pred_results, train_ae_results, train_generate_results, y_test_labels, topk = 1, option = 2, method="max", alpha =args.test_alpha, beta=args.test_beta, gamma = args.test_gamma, search_steps=args.search_steps)
print(info)
print()

print("================= Find best f1 from score: normalization loss ... =================")
info = get_final_result_POT(test_pred_results, test_ae_results, test_generate_results, train_pred_results, train_ae_results, train_generate_results, y_test_labels, topk = 1, option = 2, method="sum", alpha =args.test_alpha, beta=args.test_beta, gamma = args.test_gamma, search_steps=args.search_steps)
print(info)


print()
print("================= Find best f1 from score: normalization loss ... =================")
info = get_final_result_POT(test_pred_results, test_ae_results, test_generate_results, train_pred_results, train_ae_results, train_generate_results, y_test_labels, topk = 1, option = 2, method="mean", alpha =args.test_alpha, beta=args.test_beta, gamma = args.test_gamma, search_steps=args.search_steps)
print(info)



