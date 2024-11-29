# STAMP

***Note: The codes and docs are being updated and optimized continuously.***
## 1.dataset
1.1 SWAT
## 1.常用参数说明
- pred_model: 预测模型 [lstm,cnn,gru,gat]

1.1 图结构参数
- nnodes：节点数目（KPI个数），因数据集不同而不同
- top-k：邻居节点个数
- em_dim：节点嵌入向量维度
- alpha：缩放系数
- hidden_dim：隐藏层维度

1.2 预测模型参数
- window_size：时间窗口大小
- n_pred：预测步数
- temp_kernel：最大时间卷积核大小，如多尺度卷积核[2,3,5]，则`temp_kernel=5`
- in_channels：输入数据维度
- out_channels：输出数据维度
- layer_num：block（时间学习、时空学习模块）数量
- act_func：激活函数
- pred_lr_init：预测模型初始学习率

1.3 自编码器参数
- latent_size：潜在向量维度
- ae_lr_init：自编码器初始学习率

1.4 训练参数
- seed：随机种子
- val_ratio：验证集比例
- is_down_sample：是否采用下采样策略
- down_len：下采样长度
- is_mas：是否用历史移动平均分量特征

1.5 测试参数
- test_alpha：预测误差权重
- test_beta：重构误差权重
- test_gamma：对抗误差权重
- search_steps：网格搜索步数
- top_kErr：输出topk根因KPI

1.6 仅预测模型

run_pred.py
