# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network import N, N1, N2
from dataset_real4096_event import RealEvent4096Dataset


def custom_collate(batch):
    batch = [
        item for item in batch
        if item is not None
        and item[0] is not None and item[1] is not None and item[2] is not None
        and len(item[0]) == 4096 and len(item[1]) == 4096
    ]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor(), []
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)


class MinimalUnifiedModel(nn.Module):
    def __init__(self, model_N, model_N1, model_N2):
        super().__init__()
        self.model_N = model_N
        self.model_N1 = model_N1
        self.model_N2 = model_N2

    def forward(self, x, y):
        out_N = self.model_N(x)
        _ = self.model_N1(y, x)
        out_N2 = self.model_N2(out_N, x)
        return out_N2


def inv_signed_sqrt(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * (torch.abs(y) ** 2)


def load_submodule_weights(model: MinimalUnifiedModel, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    def pick(prefix):
        return {k.replace(prefix, ""): v for k, v in sd.items() if k.startswith(prefix)}

    model.model_N.load_state_dict(pick("model."), strict=False)
    model.model_N1.load_state_dict(pick("model1."), strict=False)
    model.model_N2.load_state_dict(pick("model2."), strict=False)

    print(f"✅ loaded weights from: {ckpt_path}")


@torch.no_grad()
def run_one_event(event_name, strain_dir, signal_dir, save_dir, model, device, signal_scale):
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 注意：真实事件 dataset 没有 return_meta 参数
    dataset = RealEvent4096Dataset(
        event_name=event_name,
        strain_dir=strain_dir,
        signal_dir=signal_dir,
        debug_verify=True,
        debug_print_first=10
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,         # ❗顺序绝对不乱
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate
    )

    model.eval()

    saved = 0
    skipped_batches = 0

    for inputs, labels, metas in loader:
        if inputs.nelement() == 0 or labels.nelement() == 0 or len(metas) == 0:
            skipped_batches += 1
            continue

        meta = metas[0]
        center_index = meta["center_index"]
        save_path = os.path.join(save_dir, f"plot_center_{center_index}.png")

        # ✅ 断点续跑：已存在就跳过
        if os.path.exists(save_path):
            continue

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs, labels)

        labels_3d = labels.view(labels.size(0), 1, -1)
        signal_gt = inv_signed_sqrt(labels_3d) / signal_scale
        signal_hat = inv_signed_sqrt(outputs) / signal_scale

        x_np = inputs.cpu().numpy().squeeze()
        gt_np = signal_gt.cpu().numpy().squeeze()
        pr_np = signal_hat.cpu().numpy().squeeze()

        plt.figure(figsize=(12, 7))
        plt.subplot(3, 1, 1); plt.plot(x_np); plt.grid(True); plt.title(f"{event_name} | strain | center={center_index}")
        plt.subplot(3, 1, 2); plt.plot(gt_np); plt.grid(True); plt.title("signal GT")
        plt.subplot(3, 1, 3); plt.plot(pr_np); plt.grid(True); plt.title("signal Pred")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        saved += 1

        if saved <= 5:
            print(f"[{event_name}] saved -> {save_path}")

    print(f"[DONE] {event_name}: total saved {saved} plots | skipped_empty_batches={skipped_batches}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_base",
        default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/RealData_20_2048_RealGpsTime")

    parser.add_argument("--out_base",
        default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Result/RealEvent_20_2048_RealGpsTime_405epoch")

    parser.add_argument("--checkpoint",
        default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/weight/Threeseg/unified_model_epoch_405.pth")

    # parser.add_argument("--real_base",
    #     default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/RealData_20_2048_NoSignalPar")

    # parser.add_argument("--out_base",
    #     default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Result/RealEvent_20_2048_NoSignalPar")

    # parser.add_argument("--checkpoint",
    #     default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/weight/Threeseg/unified_model_epoch_74.pth")
    

    # parser.add_argument("--real_base",
    #     default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/RealData_20_2048_RealGpsTime_DifficultEvent/HavePar")

    # parser.add_argument("--out_base",
    #     default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Result/RealEvent_20_2048_DifficultEvent/HavePar")

    # parser.add_argument("--checkpoint",
    #     default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/weight/Threeseg/unified_model_epoch_152.pth")

    # parser.add_argument("--real_base",
    #     default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/RealData_20_2048_RealGpsTime_DifficultEvent/NoPar")

    # parser.add_argument("--out_base",
    #     default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Result/RealEvent_20_2048_DifficultEvent/NoPar")

    # parser.add_argument("--checkpoint",
    #     default="/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/weight/Threeseg/unified_model_epoch_322.pth")

    parser.add_argument("--signal_scale", type=float, default=1e23)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MinimalUnifiedModel(
        N(dim1=64, dim2=128, dim3=256, depth=4, heads=8, mlp_dim=2048),
        N1(dim1=64, dim2=128, dim3=256, depth=4, heads=8, mlp_dim=2048),
        N2(dim1=64, dim2=128, dim3=256, depth=4, heads=8, mlp_dim=2048),
    ).to(device)

    load_submodule_weights(model, args.checkpoint, device=device)

    events = sorted([
        d for d in os.listdir(args.real_base)
        if os.path.isdir(os.path.join(args.real_base, d))
    ])

    print(f"Found {len(events)} events under {args.real_base}")

    for ev in events:
        ev_dir = os.path.join(args.real_base, ev)
        strain_dir = os.path.join(ev_dir, "strain")
        signal_dir = os.path.join(ev_dir, "signal")

        if not (os.path.isdir(strain_dir) and os.path.isdir(signal_dir)):
            print(f"[SKIP] {ev} missing strain/signal dirs")
            continue

        save_dir = os.path.join(args.out_base, ev)
        run_one_event(ev, strain_dir, signal_dir, save_dir, model, device, args.signal_scale)

    print("ALL EVENTS DONE.")


if __name__ == "__main__":
    main()
