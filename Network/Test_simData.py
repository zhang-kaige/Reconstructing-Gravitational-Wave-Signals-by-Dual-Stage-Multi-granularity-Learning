# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from network import N, N1, N2
from dataprocessor import Mydata


def custom_collate(batch):
    # batch item: (x, y, meta) 或 None
    batch = [
        item for item in batch
        if item is not None
        and item[0] is not None and item[1] is not None and item[2] is not None
        and len(item[0]) == 4096 and len(item[1]) == 4096
    ]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor(), []

    inputs, labels, metas = zip(*batch)
    return torch.stack(inputs), torch.stack(labels), list(metas)



class MinimalUnifiedModel(nn.Module):
    def __init__(self, model_N, model_N1, model_N2):
        super().__init__()
        self.model_N = model_N
        self.model_N1 = model_N1
        self.model_N2 = model_N2

    def forward(self, x, y):
        out_N = self.model_N(x)
        out_N1 = self.model_N1(y, x)   # 你原来的结构
        result = out_N                 # 你原来就是用 out_N，不用 out_N1
        out_N2 = self.model_N2(result, x)
        return out_N2


def inv_signed_sqrt(y: torch.Tensor) -> torch.Tensor:
    # y = sign(x)*sqrt(|x|)  ->  x = sign(y)*(|y|^2)
    return torch.sign(y) * (torch.abs(y) ** 2)


def load_submodule_weights(model: MinimalUnifiedModel, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)

    # 去掉 DataParallel 的 module.
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    # 取出 unified_model 里三段的前缀（和你训练代码保持一致）
    n_prefix = 'model.'
    n1_prefix = 'model1.'
    n2_prefix = 'model2.'

    n_dict = {k.replace(n_prefix, ''): v for k, v in sd.items() if k.startswith(n_prefix)}
    n1_dict = {k.replace(n1_prefix, ''): v for k, v in sd.items() if k.startswith(n1_prefix)}
    n2_dict = {k.replace(n2_prefix, ''): v for k, v in sd.items() if k.startswith(n2_prefix)}

    missing_n, unexpected_n = model.model_N.load_state_dict(n_dict, strict=False)
    missing_n1, unexpected_n1 = model.model_N1.load_state_dict(n1_dict, strict=False)
    missing_n2, unexpected_n2 = model.model_N2.load_state_dict(n2_dict, strict=False)

    print(f"✅ loaded weights from: {ckpt_path}")
    if missing_n or unexpected_n:
        print("  [N] missing:", missing_n[:10], "..." if len(missing_n) > 10 else "")
        print("  [N] unexpected:", unexpected_n[:10], "..." if len(unexpected_n) > 10 else "")
    if missing_n1 or unexpected_n1:
        print("  [N1] missing:", missing_n1[:10], "..." if len(missing_n1) > 10 else "")
        print("  [N1] unexpected:", unexpected_n1[:10], "..." if len(unexpected_n1) > 10 else "")
    if missing_n2 or unexpected_n2:
        print("  [N2] missing:", missing_n2[:10], "..." if len(missing_n2) > 10 else "")
        print("  [N2] unexpected:", unexpected_n2[:10], "..." if len(unexpected_n2) > 10 else "")


def run_inference(model, dataloader, save_dir: str, signal_scale: float):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, labels, metas in dataloader:
            if inputs.nelement() == 0 or labels.nelement() == 0 or len(metas) == 0:
                continue

            inputs = inputs.to(device)   # [B,4096]
            labels = labels.to(device)   # [B,4096]

            outputs = model(inputs, labels)

            labels_3d = labels.view(labels.size(0), 1, -1)

            signal_gt = inv_signed_sqrt(labels_3d) / signal_scale
            signal_hat = inv_signed_sqrt(outputs) / signal_scale

            inputs_np = inputs.cpu().numpy()
            signal_gt_np = signal_gt.cpu().numpy()
            signal_hat_np = signal_hat.cpu().numpy()

            # ====== 用 meta 命名 ======
            # batch_size=1 时 metas[0] 就是这一条
            meta = metas[0]
            idx_val = meta["idx"]
            which = meta["which"]  # 0/1/2
            which_1based = which + 1

            # ========= 画图 =========
            plt.figure(figsize=(12, 7))

            plt.subplot(3, 1, 1)
            plt.plot(inputs_np.squeeze())
            plt.grid(True)
            plt.title(f"strain (normalized) | idx={idx_val} seg={which_1based}")

            plt.subplot(3, 1, 2)
            plt.plot(signal_gt_np.squeeze())
            plt.grid(True)
            plt.title("signal GT (inverse to original scale)")

            plt.subplot(3, 1, 3)
            plt.plot(signal_hat_np.squeeze())
            plt.grid(True)
            plt.title("signal Pred (inverse to original scale)")

            save_path = os.path.join(save_dir, f"plot_result_step_{idx_val}_{which_1based}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[idx={idx_val} seg={which_1based}] saved -> {save_path}")



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()

#20_2048 
    # 测试集
    # parser.add_argument('--traindir', type=str,
    #                     default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/SNR_1_30x10/strain')
    # parser.add_argument('--labeldir', type=str,
    #                     default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/SNR_1_30x10/signal')

    # parser.add_argument('--traindir', type=str,
    #                     default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data_noise_only_Test/strain')
    # parser.add_argument('--labeldir', type=str,
    #                     default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data_noise_only_Test/signal')

    parser.add_argument('--traindir', type=str,
                        default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/5/strain')
    parser.add_argument('--labeldir', type=str,
                        default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/5/signal')


    
    parser.add_argument('--window', type=int, default=1024)

    parser.add_argument('--checkpoint', type=str,
                        default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/weight/Threeseg/unified_model_epoch_152.pth')

    parser.add_argument('--save_dir', type=str,
                        # default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Result/RealEvent_20_1024_newNorm/GW231223_032836')
                        # default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Result/TestData/50epoch_newNorm/1000_OneCrop_20_1024_newNorm_inv')
                        # default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/SNR_1_30x10/result')
                        # default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Result/SimuEvent_20_2048_PureNoise')
                        default='/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/5/test_5_samples')
    
    
    

    # 必须和 dataprocessor_newNorm 里的 signal_scale 一致（你现在用 1e23）
    parser.add_argument('--signal_scale', type=float, default=1e23)

    # 你之前写 parse_args(args=[]) 我这里保留同样行为（脚本内跑）
    args = parser.parse_args(args=[])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Mydata(
        dataset_dir=args.traindir,
        label_dir=args.labeldir,
        window=args.window,
        return_meta=True,
        debug_verify=True,          # ✅开启验证
        debug_print_first=18,       # 打印前18条（正好6个idx的顺序）
        debug_tail_check_k=5        # 抽样检查 tail 一致性
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate
    )

    N_model = N(dim1=64, dim2=128, dim3=256, depth=4, heads=8, mlp_dim=2048)
    N1_model = N1(dim1=64, dim2=128, dim3=256, depth=4, heads=8, mlp_dim=2048)
    N2_model = N2(dim1=64, dim2=128, dim3=256, depth=4, heads=8, mlp_dim=2048)
    model = MinimalUnifiedModel(N_model, N1_model, N2_model).to(device)

    load_submodule_weights(model, args.checkpoint, device=device)

    run_inference(
        model=model,
        dataloader=dataloader,
        save_dir=args.save_dir,
        signal_scale=args.signal_scale
    )
