# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from network import N, N1, N2
from dataprocessor import Mydata
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os, argparse, pytz
from datetime import datetime
from collections import OrderedDict

# ===== collate 函数 =====
def custom_collate(batch):
    """
    过滤掉长度不对或为 None 的样本，然后 stack 成 batch
    """
    batch = [
        item for item in batch
        if item[0] is not None and item[1] is not None
        and len(item[0]) == 4096 and len(item[1]) == 4096
    ]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.stack(labels)

# ===== 统一模型 =====
class UnifiedModel(nn.Module):
    def __init__(self, model, model1, model2):
        super(UnifiedModel, self).__init__()
        self.model = model
        self.model1 = model1
        self.model2 = model2
        
    def forward(self, inputs, labels):
        # model: 主分支，输入 noisy / strain
        output = self.model(inputs)
        # model1: 可能是中间变换，输入 label + inputs
        y = self.model1(labels, inputs)
        # model2: 最后一个网络
        y2 = self.model2(y, inputs)
        return output, y, y2

# ===== RMSE损失（目前没用到，先保留）=====
class RMSELoss(nn.Module):
    def forward(self, input, target):
        return torch.sqrt(F.mse_loss(input, target))

# ===== 从 ckpt 加载权重（兼容 DataParallel 保存的 module.*）=====
def load_unified_checkpoint(unified_model, optimizer, checkpoint_path, device):
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # 如果 key 里带有 'module.' 前缀，说明是 DataParallel 存的
        if any(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # 只去掉最前面的一个 'module.'
                new_k = k.replace('module.', '', 1)
                new_state_dict[new_k] = v
            state_dict = new_state_dict

        # 加载到当前 unified_model（此时还是“裸模型”，不是 DataParallel）
        missing, unexpected = unified_model.load_state_dict(state_dict, strict=False)
        print(f"✅ 已加载 unified 模型权重: {checkpoint_path}")
        if missing:
            print("⚠️ Missing keys(前10个):", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("⚠️ Unexpected keys(前10个):", unexpected[:10], "..." if len(unexpected) > 10 else "")

        # 如果有 optimizer 状态，也一并恢复
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ 已加载 optimizer 状态")
            except Exception as e:
                print("⚠️ 加载 optimizer 状态失败:", e)
    else:
        print(f"⚠️ 未找到权重文件: {checkpoint_path}")

# ===== 训练 =====
def train_model(unified_model, train_loader, writer1, writer2, writer3,
                optimizer, start_epoch, num_epochs, model_dir_model):
    os.makedirs(model_dir_model, exist_ok=True)
    device = next(unified_model.parameters()).device
    unified_model.train()
    global_step = 0

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        total_loss_main = 0.0
        total_loss1 = 0.0
        count_batches = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader, start=1):

            # 可能出现空 batch（collate 过滤掉了所有样本）
            if inputs.nelement() == 0 or labels.nelement() == 0:
                continue

            inputs = inputs.float().to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            optimizer.zero_grad()
            output, y, y2 = unified_model(inputs, labels)

            # labels reshape 成 [B, 1, T]
            labels = labels.view(labels.size(0), 1, -1)

            # ===== 权重矩阵：按标签最大幅值分档 =====
            max_vals = torch.max(torch.abs(labels), dim=2).values.squeeze(1)  # [B]
            weights = torch.ones_like(max_vals)
            
            weights[(max_vals < 1.0)] = 20.0
            weights[(max_vals >= 1.0) & (max_vals < 3.0)] = 5.0
            weights[(max_vals >= 3.0) & (max_vals < 10.0)] = 2.0
            weights[(max_vals >= 10.0)] = 1.0

            # ===== 三个 loss 计算逻辑 =====
            # output / y 都是列表: [out0, out1, out2]
            lossa = torch.sqrt(F.mse_loss(output[0], y[0], reduction='none').mean(dim=(1, 2)))
            lossb = torch.sqrt(F.mse_loss(output[1], y[1], reduction='none').mean(dim=(1, 2)))
            lossc = torch.sqrt(F.mse_loss(output[2], y[2], reduction='none').mean(dim=(1, 2)))
            loss = lossa + lossb + lossc          # [B]
            loss_avg = loss.mean()                # 标准平均 loss（统计用）

            loss1 = torch.sqrt(F.mse_loss(y2, labels, reduction='none').mean(dim=(1, 2)))  # [B]
            loss1_avg = loss1.mean()

            # ===== 总损失：0.95 * 主损失 + 0.05 * loss1，然后按样本权重加权 =====
            los = loss * 0.95 + loss1 * 0.05      
            weighted_los = los * weights          
            los = torch.mean(weighted_los)       

            los.backward()
            optimizer.step()

            total_loss += los.item()
            total_loss_main += loss_avg.item()
            total_loss1 += loss1_avg.item()
            count_batches += 1
            global_step += 1

        # 一个 epoch 结束，统计平均
        average_loss = total_loss / count_batches
        average_loss_main = total_loss_main / count_batches
        average_loss1 = total_loss1 / count_batches

        # 写到 TensorBoard（按 epoch 记录）
        writer1.add_scalar('Loss', average_loss, epoch)
        writer2.add_scalar('Loss', average_loss_main, epoch)
        writer3.add_scalar('Loss', average_loss1, epoch)
        writer1.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # ===== 保存模型 =====
        model_path = os.path.join(model_dir_model, f'unified_model_epoch_{epoch}.pth')
        torch.save({
            'model_state_dict': unified_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)

        print(f"模型已保存到 {model_path}, Epoch {epoch} ave loss:{average_loss:.8f}")
        print(f"模型1 ave loss:{average_loss_main:.8f}")
        print(f"模型2 ave loss:{average_loss1:.8f}")
        print('OK')

# ===== 主程序 =====
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    parse = argparse.ArgumentParser()
    parse.add_argument('--traindir', type=str, default='/raid/zkg/Data/NewData/strain')
    parse.add_argument('--labeldir', type=str, default='/raid/zkg/Data/NewData/signal')
    parse.add_argument('--window', type=int, default=1024)
    parse.add_argument('--epochs', type=int, default=20000)
    parse.add_argument('--multi_gpu', action='store_true', help='使用多卡 DataParallel')
    args = parse.parse_args()  # ✅ 这里不要再传 args=[] 了

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== 数据集 & DataLoader =====
    dataset = Mydata(dataset_dir=args.traindir,
                     label_dir=args.labeldir,
                     window=args.window)

    train_loader = DataLoader(dataset,
                              batch_size=400,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=custom_collate)

    # ===== 模型（先建“裸模型”）=====
    model = N(64, 128, 256, 4, 8, 2048).to(device)
    model1 = N1(64, 128, 256, 4, 8, 2048).to(device)
    model2 = N2(64, 128, 256, 4, 8, 2048).to(device)
    unified_model = UnifiedModel(model, model1, model2).to(device)

    # ===== 优化器（针对“裸模型”）=====
    optimizer = torch.optim.Adam(unified_model.parameters(), lr=0.001)

    # ===== 先加载 ckpt 到“非 DataParallel”模型 =====
    checkpoint_path = '/raid/zkg/Projects/Double/v_010526/weight/unified_model_epoch_348.pth'
    # checkpoint_path = ''
    load_unified_checkpoint(unified_model, optimizer, checkpoint_path, device=device)

    # ===== 强制把继续训练的 lr 改成 1e-4（不影响 m/v 的加载）=====
    new_lr = 1e-4
    for pg in optimizer.param_groups:
        pg['lr'] = new_lr
    print("✅ 已强制设置继续训练 lr =", [pg['lr'] for pg in optimizer.param_groups])

    # ===== 再根据需要包 DataParallel（这之后不再重新 load ckpt）=====
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"使用多卡训练, DataParallel 包装模型, 可见 GPU 数量 = {torch.cuda.device_count()}")
        unified_model = nn.DataParallel(unified_model)
    unified_model = unified_model.to(device)

    # ===== 从 epoch 号继续 =====
    start_epoch = int(checkpoint_path.split('_epoch_')[-1].split('.')[0]) + 1
    # start_epoch = 0

    print(f"从 epoch {start_epoch} 开始继续训练")

    # ===== TensorBoard =====
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(tz)
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    writer1 = SummaryWriter(os.path.join("/raid/zkg/Projects/Double/v_010526/run_log", f'{timestamp}_TotalLoss'))
    writer2 = SummaryWriter(os.path.join("/raid/zkg/Projects/Double/v_010526/run_log", f'{timestamp}_MainLoss'))
    writer3 = SummaryWriter(os.path.join("/raid/zkg/Projects/Double/v_010526/run_log", f'{timestamp}_SecondaryLoss'))

    # ===== 开始训练 =====
    train_model(
        unified_model, train_loader, writer1, writer2, writer3,
        optimizer, start_epoch=start_epoch, num_epochs=args.epochs,
        model_dir_model='/raid/zkg/Projects/Double/v_010526/weight'
    )

    writer1.close()
    writer2.close()
    writer3.close()
