"""引入一些模块"""
# 数据操作
import numpy as np
import math
# 读取、写入数据
import pandas as pd
import os
import csv
# 进度条
from tqdm import tqdm
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from torch.utils.data import random_split
# 绘图
from torch.utils.tensorboard import SummaryWriter
# --------------------------------------------------------------
"""一些操作"""
# 设置随机种子,保证实验的可重复性
def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed) # 运用 CPU 训练
    # 运用 GPU 训练
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 划分数据集：训练集、验证集、测试集
# 由于 HW1 只给出了训练集和测试集，我们需要将训练集中的一部分划为验证集

# def train_valid_split(data_set, val_ratio, seed):
#     valid_data_size = int(len(data_set) * val_ratio)
#     train_data_size = len(data_set) - valid_data_size
#     train_set, valid_set = random_split(data_set, [train_data_size, valid_data_size], torch.Generator().manual_seed(seed))
#     return np.array(train_set), np.array(valid_set)
    # Q。你使用 np.array(train_set) 将 PyTorch 的 Subset 对象转换为了 NumPy 数组，但这样只会创建一个包含 Subset 对象的 0 维数组，而不是实际的数据数组
    # 当你尝试使用 train_data[:, -1] 访问这个 0 维数组时，就会出现索引错误。
    # 获取实际的数据
    # train_indices = train_set.indices
    # valid_indices = valid_set.indices
    #
    # # 如果data_set是DataFrame，使用iloc获取数据并转换为numpy数组
    # train_data = data_set.iloc[train_indices].values
    # valid_data = data_set.iloc[valid_indices].values
    #
    # return train_data, valid_data
def train_valid_split(data_set, val_ratio, seed):
    np.random.seed(seed)
    n = len(data_set)
    indices = np.random.permutation(n)
    valid_size = int(n * val_ratio)

    # 转换为 numpy 数组后再划分
    data_np = data_set.values
    train_data = data_np[indices[valid_size:]]
    valid_data = data_np[indices[:valid_size]]

    return train_data, valid_data

# 预测
def predict(test_loader, model, device):
    model.eval()
    pred_list = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad(): # 不需要更新梯度
            pred = model(x)
            pred_list.append(pred.detach().cpu()) # 将数据从GPU 移动到 CPU，转换到 numpy 数组
    preds = torch.cat(pred_list, dim=0).numpy()
    return preds
#-----------------------------------------------------------------------------------------
# 选择特征
# 最后一列数据是 label(y), 剩余的列是 feature(x)
def select_feature(train_data, valid_data, test_data, select_all=True):
    train_y = train_data[:, -1] # 行取全部，列取最后一列
    train_x = train_data[:, :-1] # 行取全部，列取除了最后一列

    valid_y = valid_data[:, -1]
    valid_x = valid_data[:, :-1]

    test_x = test_data
    if select_all:
        feat_idx = list(range(train_x.shape[1])) # train_x.shape = (2160, 117)
    else:
        feat_idx = [0, 1, 2, 3, 4] # 只使用前5个特征
    # 依次返回：训练集、验证集、测试集特征 和 训练集、验证集标签
    return train_x[:, feat_idx] , valid_x[:, feat_idx], test_x[:, feat_idx], train_y, valid_y


# 数据集
class COVID19Dataset(Dataset):
    def __init__(self, features, target = None):
        # 将数据转换为 torch.FloatTensor 类型
        if target is None: # 测试集
            self.target = None
        else:
            self.target = torch.FloatTensor(target) # 训练集
        self.features = torch.FloatTensor(features)

    def __getitem__(self, idx):
        if self.target is None:
            return self.features[idx] # 只需要特征
        else:
            return self.features[idx], self.target[idx]

    def __len__(self):
        return len(self.features)

# Dataloader
# 神经网络
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # 序列容器
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # 删除维度为 1 的维度 eg.(batch_size, 1) -> (batch_size)
        #? 此处为什么要删除维度为 1 的维度？
        # MSE损失函数要求输入的维度为 1
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu' # 选择设备 cpu or gpu
# 设置参数
config = {
    'seed': 5201314, # 1122408
    'select_all': True,
    'valid_ratio': 0.2,
    'n_epochs': 3000,
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400, # 重复训练 400 轮模型没有更新,则停止训练 epoch??
    'save_path':'./models/model.ckpt'
}

# 训练过程
def train(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean') # MSEloss,取平均值
    # L2正则化 https://pytorch.org/docs/stable/optim.html
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    
    # 创建TensorBoard日志目录
    log_dir = './runs/covid_experiment'
    writer = SummaryWriter(log_dir)
    
    # 记录模型结构
    sample_input = torch.randn(1, train_loader.dataset.features.shape[1]).to(device)
    writer.add_graph(model, sample_input)
    
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs = config['n_epochs']
    best_loss = math.inf # 初始化为无穷大
    step = 0
    early_stop_count = 0

    # 开始训练
    for epoch in range(n_epochs):
        model.train()
        train_loss_record = []
        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss_record.append(loss.detach().item())
            step += 1

            # 显示训练过程
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix(loss=loss.detach().item()) # 生成副本

        mean_train_loss = sum(train_loss_record) / len(train_loss_record)
        writer.add_scalar('Loss/Train', mean_train_loss, epoch)
        
        # valid process
        model.eval()
        valid_loss_record = []

        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)
            valid_loss_record.append(loss.item())

        mean_valid_loss = sum(valid_loss_record) / len(valid_loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/Validation', mean_valid_loss, epoch)
        
        # 记录学习率
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 记录模型参数的直方图
        if epoch % 100 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)
        
        # 记录损失差异
        loss_diff = abs(mean_train_loss - mean_valid_loss)
        writer.add_scalar('Loss/Difference', loss_diff, epoch)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            break
    
    # 关闭writer
    writer.close()
    print(f"\nTensorBoard日志已保存到: {log_dir}")
    print("启动TensorBoard查看训练过程:")
    print(f"命令行运行: tensorboard --logdir={log_dir}")
    print("然后在浏览器中打开: http://localhost:6006")

"""准备工作"""
# 设置随机数种子
same_seed(config['seed'])
# 读取数据
train_df = pd.read_csv('./dataset/covid.train.csv').drop(columns=['id'])
test_df = pd.read_csv('./dataset/covid.test.csv').drop(columns=['id'])
train_data, valid_data = train_valid_split(train_df, config['valid_ratio'], config['seed'])
test_data = test_df.values
print(f"""train_data_size:{train_data.shape}, valid_data_size:{valid_data.shape}, test_data_size:{test_data.shape}""")
# 选择特征
train_x, valid_x, test_x, train_y, valid_y = select_feature(train_data, valid_data, test_data, config['select_all']) #? 要加.values吗？
print(f"the number of features:{train_x.shape[1]}")
# 构造数据集
train_dataset, valid_dataset, test_dataset = COVID19Dataset(train_x, train_y), COVID19Dataset(valid_x, valid_y), COVID19Dataset(test_x)
# 构造 dataloader
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory = True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory = True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory = True) # 测试集不需要打乱

"""开始训练！"""
model = My_Model(input_dim= train_x.shape[1]).to(device)
trainer = train(train_loader, valid_loader, model, config, device)

# 以.csv文件形式保存预测结果
def save_pred(preds, file):
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

# 预测并保存结果
model = My_Model(input_dim= train_x.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'pred.csv')




