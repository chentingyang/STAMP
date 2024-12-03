# STAMP

## 1.dataset

Please refer to '/lib/dataloader_...' for more details of data loading and preprocessing

1.1 SWaT and WADI
- train and test data in '.csv' with the last column as labels (0 for normal and 1 for abnormal)

1.2 SMD and MSL
- train and test data in '..._train.pkl' and '..._test.pkl' formed as arrays with the shape (samples, features), for example: 'machine-1-1_train.pkl' or 'MSL_train.pkl'
- test labels also in '...test_label.pkl formed as arrays with the shape (samples,)

1.3 Unsupervised Datasets
- the train and test data should be concatenated together and saved in '.....npz', where label 'a' are samples with the shape (samples, features) and label 'b' are labels with the shape (samples,), you can refer to '/data/unsupervised_data/test_data_smd_unsup.npz'

1.4 Your Datasets
- you can use '/lib/dataloader_...' according to your data format or create new dataloaders

## 2. key parameters

- data: dataset name, such as 'SWaT', 'WADI', ...
- temp_methods: for STAMP, keep it as 'SAttn'

2.1 Graph Structure
- nnodes：number of features, which varies across datasets
- top-k：number of neighbor nodes


2.2 Pred Model
- window_size：time window length
- n_pred：prediction step
- temp_kernel：the kernel size of Convolutional Input-Output Layer
- layer_num：number of TLL and SLL Layers
- act_func：activation function

2.3 Attention
- embed_size: embedding size
- num_heads: number of attention heads
- is_conv: while True, FFN use Linear; else, FFN use Conv1D
 
2.4 AE
- latent_size：dimension of L-space


2.5 Training

- is_down_sample：whether performing down-sampling to the original samples
- down_len：down-sampling ratio
- is_mas：whether performing a moving average operation by sub-windows to extend channel

2.6 Testing
- test_alpha：weight of prediction error
- test_beta：weight of reconstruction error
- test_gamma：weight of adversarial error

2.7 params in get_final_result()
- topk: number of features to calculate the anomaly score
- option: set to 2
- method: ['sum', 'max', 'mean'], types of aggragation operators

## 3. Semi-Supervised Detecting

3.1 Run STAMP

python run.py --down_len 1 --epoch 5 --data SMD --nnodes 38 --window_size 15 --n_pred 3

3.2 
check the saved model weights in '/expe'

3.3 Evaluation

python test.py --down_len 1 --epoch 5 --data SMD --nnodes 38 --window_size 15 --n_pred 3 --test_alpha 0.5 --test_beta 0.1 --test_gamma 0.4 

## 4. Unsupervised Detecting

4.1 Get Model-Derived information

python get_model_information.py --down_len 1 --epoch 5 --data SMD --nnodes 38 --window_size 15 --n_pred 3 --test_alpha 0.5 --test_beta 0.1 --test_gamma 0.4

4.2 
check the saved model information in '/expe'

4.3 Screening Based on Model Information

python /unsupervise/Screening.py

4.4 Run STAMP

python run_unsup.py --down_len 1 --epoch 5 --data SMD --nnodes 38 --window_size 15 --n_pred 3

4.5 Evaluation

python test_unsup.py --down_len 1 --epoch 5 --data SMD --nnodes 38 --window_size 15 --n_pred 3 --test_alpha 0.5 --test_beta 0.1 --test_gamma 0.4 





