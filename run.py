

import  os
import argparse


parser = argparse.ArgumentParser(description='PyTorch RNN Prediction Model on Time-series Dataset')
parser.add_argument('--data', type=str, default='SWaT',
                    help='type of the dataset (SWaT, WADI, ...)')
parser.add_argument('--filename', type=str, default='SWaT_Dataset_Normal_v1.csv',
                    help='filename of the dataset')
parser.add_argument('--debug', default=False, type=eval)
parser.add_argument('--real_value', default=False, type=eval)
parser.add_argument('--log_dir', default="expe", type=str)
parser.add_argument('--log_dir_transfer', default="expe/transfer_learning", type=str)#迁移学习模型存放地址
parser.add_argument('--model', default="v2_", type=str)
parser.add_argument('--pred_model', default="gat", type=str)
parser.add_argument('--gpu_id', default="0", type=str)
parser.add_argument('--temp_method', default="Conv", type=str)

### graph constructure
parser.add_argument('--nnodes', type=int, default=45, help='number of nodes')
parser.add_argument('--top_k', type=int, default=10, help='top-k')
parser.add_argument('--em_dim', type=int, default=32, help='embedding dimension')
parser.add_argument('--alpha', type=int, default=3, help='alpha')
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden_dim')
parser.add_argument('--att_option', type=int, default=1, help='att_option')

### GATModel
parser.add_argument('--window_size', type=int, default=15, help='window_size')
parser.add_argument('--n_pred', type=int, default=3, help='n_pred')
parser.add_argument('--temp_kernel', type=int, default=5, help='temp_kernel')
parser.add_argument('--in_channels', type=int, default=1, help='in_channels')
parser.add_argument('--out_channels', type=int, default=1, help='out_channels')

parser.add_argument('--layer_num', type=int, default=2, help='layer_num')
parser.add_argument('--act_func', type=str, default="GLU", help='act_func')
parser.add_argument('--pred_lr_init', type=float, default=0.001, help='pred_lr_init')
parser.add_argument('--pred_lr_init_trans', type=float, default=0.001, help='pred_lr_init_transfer')#迁移学习学习率

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
parser.add_argument('--ae_lr_init_trans', type=float, default=0.001, help='ae_lr_init_transfer')#迁移学习学习率

parser.add_argument('--seed', type=int, default=666, help='seed')
parser.add_argument('--val_ratio', type=float, default=.2, help='val_ratio')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=30,  help='number of epoch')
parser.add_argument('--is_down_sample', type=bool, default=True, help='is_down_sample')
parser.add_argument('--down_len', type=int, default=100, help='down_len')

parser.add_argument('--early_stop', default=True, type=eval)
parser.add_argument('--early_stop_patience', type=int, default=10, help='early_stop_patience')

parser.add_argument('--lr_decay', default=True, type=eval)
parser.add_argument('--lr_decay_rate', default=0.5, type=float)
parser.add_argument('--lr_decay_step', default="5,20,40,70", type=str)
parser.add_argument('--largest_loss_diff', default=0.2, type=float)
parser.add_argument('--is_graph', default=True, type=eval)
parser.add_argument('--is_mas', default=True, type=eval)
parser.add_argument('--transfer_option', default=1, type=int, help='trasfer learning option')#迁移学习选项，1为重新训练模型，2为加载原始全部模型再微调，3为只加载pred_model并固定，4为只加载ae_model并固定
parser.add_argument('--transfer_epochs', type=int, default=40,  help='number of transfer learning epoch')#迁移训练epoch

args = parser.parse_args()

args.model = args.model + args.pred_model

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

from model.net import *
from trainer import Trainer
from lib.logger import get_logger
#from lib.dataloader_swat import load_data, load_data2
from lib.dataloader_smd import load_data, load_data2
#from lib.dataloader_wadi import load_data, load_data2
#from lib.dataloader_msl_smap import load_data, load_data2
from lib.utils import *
from lib.metrics import *
from model.utils import *


print(torch.cuda.is_available())

DEVICE = get_default_device()

base_dir = os.getcwd()
'''
train_filename = base_dir + "/data/" + args.data + "/SWaT_Dataset_normal.csv"
test_filename = base_dir + "/data/" + args.data + "/SWaT_Dataset_attack.csv"


if args.is_mas:
    train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = load_data2(train_filename, test_filename,
                                                                                     device=DEVICE,
                                                                                     window_size=args.window_size,
                                                                                     val_ratio=args.val_ratio,
                                                                                     batch_size=args.batch_size,
                                                                                     is_down_sample=args.is_down_sample,
                                                                                     down_len=args.down_len)
else:
    train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = load_data(args.data, device = DEVICE,
                                                                                 window_size = args.window_size, val_ratio = args.val_ratio,
                                                                                 batch_size = args.batch_size,
                                                                                 is_down_sample=args.is_down_sample, down_len=args.down_len)
'''

if args.is_mas:
    train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = load_data2('SMD', 'non_trival',
                                                                                     device=DEVICE,
                                                                                     window_size=args.window_size,
                                                                                     val_ratio=args.val_ratio,
                                                                                     batch_size=args.batch_size,
                                                                                     is_down_sample=args.is_down_sample,
                                                                                     down_len=args.down_len)
else:
    train_loader, val_loader, test_loader, y_test_labels, min_max_scaler = load_data('SMD', 'non_trival', device = DEVICE,
                                                                                 window_size = args.window_size, val_ratio = args.val_ratio,
                                                                                 batch_size = args.batch_size,
                                                                                 is_down_sample=args.is_down_sample, down_len=args.down_len)

## set seed
init_seed(args.seed)

channels_list = [[16,8,32],[32,8,64]]
# channels_list = [[32,16,64],[64,16,64]]
# channels_list = [[16,8,16],[16,8,32],[32,8,64]]

AE_IN_CHANNELS = args.window_size * args.nnodes * args.in_channels
latent_size = args.window_size * args.latent_size
'''
graph = np.load('smd_gc.npy')
graph = torch.from_numpy(graph).to(DEVICE)
'''
if args.pred_model in ["gat","GAT"]:
    pred_model = STATModel(args, DEVICE, args.window_size - args.n_pred, channels_list, static_feat=None)
elif args.pred_model in ["cnn", "CNN"]:
    channels_list = [[16, 8, 32], [32, 8, 32]]
    pred_model = OurCNN(args, DEVICE, args.window_size - args.n_pred, channels_list, static_feat=None)
elif args.pred_model in ["lstm", "LSTM"]:
    args.hidden_dim = 16
    pred_model = OurLSTM(args, DEVICE, args.window_size - args.n_pred, hidden_dim=args.hidden_dim)
else:
    raise "model Error ..."
'''
model_path = "./expe/"+'best_model_' + args.data + "_" + args.model + '.pth'
map_location = torch.device(DEVICE)
check_point = torch.load(model_path, map_location = map_location)
pred_state_dict = check_point['pred_state_dict']
print("load model: ", model_path)
pred_model.load_state_dict(pred_state_dict)

# 当需要设置特定网络层不参与训练时
no_grad_params = ['gc.emb1.weight', 'gc.emb2.weight', 'gc.lin1.weight', 'gc.lin1.bias', 'gc.lin2.weight', 'gc.lin2.bias', 
                  'st_layers.0.spat_attn.self_attn.0.in_proj_weight', 'st_layers.0.spat_attn.self_attn.0.in_proj_bias', 
                  'st_layers.0.spat_attn.self_attn.0.out_proj.weight', 'st_layers.0.spat_attn.self_attn.0.out_proj.bias', 
                  'st_layers.0.spat_attn.position_ffwd.0.0.weight', 'st_layers.0.spat_attn.position_ffwd.0.0.bias', 
                  'st_layers.0.spat_attn.position_ffwd.0.2.weight', 'st_layers.0.spat_attn.position_ffwd.0.2.bias', 
                  'st_layers.0.bn.weight', 'st_layers.0.bn.bias', 'st_layers.0.gat.Theta.weight', 
                  'st_layers.1.spat_attn.self_attn.0.in_proj_weight', 'st_layers.1.spat_attn.self_attn.0.in_proj_bias', 
                  'st_layers.1.spat_attn.self_attn.0.out_proj.weight', 'st_layers.1.spat_attn.self_attn.0.out_proj.bias', 
                  'st_layers.1.spat_attn.position_ffwd.0.0.weight', 'st_layers.1.spat_attn.position_ffwd.0.0.bias', 
                  'st_layers.1.spat_attn.position_ffwd.0.2.weight', 'st_layers.1.spat_attn.position_ffwd.0.2.bias', 
                  'st_layers.1.bn.weight', 'st_layers.1.bn.bias', 'st_layers.1.gat.Theta.weight']#在这里写入不参与训练（固定）的层
    
for name_pred, param_pred in pred_model.named_parameters():
    if name_pred in no_grad_params:
        param_pred.requires_grad = False
'''
pred_model = to_device(pred_model, DEVICE)
pred_optimizer = torch.optim.Adam(params=pred_model.parameters(), lr=args.pred_lr_init, eps=1.0e-8, weight_decay=0.0001, amsgrad=False)
#pred_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, pred_model.parameters()), lr=args.pred_lr_init, eps=1.0e-8, weight_decay=0.0001, amsgrad=False)
pred_loss = masked_mse_loss(mask_value = -0.01)


ae_model = EncoderDecoder(AE_IN_CHANNELS, latent_size, AE_IN_CHANNELS, not args.real_value)
ae_model = to_device(ae_model, DEVICE)
ae_optimizer = torch.optim.Adam(params=ae_model.parameters(), lr=args.ae_lr_init, eps=1.0e-8, weight_decay=0.0001, amsgrad=False)
ae_loss = masked_mse_loss(mask_value = -0.01)

#### ====== 查看模型参数 ========
print_model_parameters(pred_model)
print_model_parameters(ae_model)

trainer = Trainer(pred_model, pred_loss, pred_optimizer, ae_model, ae_loss, ae_optimizer, train_loader, val_loader, test_loader, args, min_max_scaler, lr_scheduler=None)

train_history, val_history = trainer.train()

plot_history(train_history, model = args.model, mode="train", data=args.data)
plot_history(val_history, model = args.model, mode="val", data=args.data)
plot_history2(val_history, model = args.model, mode="val", data=args.data)

