import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset


def signed_sqrt_compress(x, eps=0.0):
    return torch.sign(x) * torch.sqrt(torch.abs(x) + eps)


def rms_normalize(x, eps=1e-8):
    rms = torch.sqrt(torch.mean(x * x) + eps)
    return x / rms


def _extract_idx(path: str):
    """
    从文件名里提取 idx：
      signal_123.npy / strain_123.npy  -> 123
    """
    base = os.path.basename(path)
    m = re.search(r'_(\d+)\.npy$', base)
    return int(m.group(1)) if m else None


class Mydata(Dataset):
    def __init__(self, dataset_dir, label_dir, window=None,
                 strain_mean=-5.067700e-05,
                 strain_std=4.512173e+01,
                 signal_scale=1e23,
                 sqrt_eps=0.0,
                 return_meta=False):
        """
        适配你现在生成的数据格式：

        每个文件长度固定：12291 = 4096*3 + 3
        尾巴 3 个 float： [idx, snr, shift]
        文件中前三段是三个 4096 段拼接的时域序列。

        处理方案（沿用你原逻辑）：
          - strain: (x - mean) / std  +  RMS normalize
          - signal: y * signal_scale，然后 signed sqrt compress
          - 默认忽略尾巴 3 个值（可选 return_meta=True 返回）
        """
        self.datadir = dataset_dir   # strain 根目录
        self.labeldir = label_dir    # signal 根目录

        self.window = window  # 这里暂时不使用（你训练当前是用全 4096）
        self.return_meta = return_meta

        # 统计量/缩放参数（你可以换成新数据重新统计的值）
        self.strain_mean = float(strain_mean)
        self.strain_std = float(strain_std)
        self.signal_scale = float(signal_scale)
        self.sqrt_eps = float(sqrt_eps)

        self.segment_length = 4096
        self.num_segments = 3
        self.extra_tail = 3
        self.expected_len = self.segment_length * self.num_segments + self.extra_tail  # 12291

        # 按 idx 对齐配对
        self.pairs = self._build_pairs()

    def _collect_npy(self, folder):
        paths = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".npy"):
                    paths.append(os.path.join(root, f))
        return paths

    def _build_pairs(self):
        strain_files = self._collect_npy(self.datadir)
        signal_files = self._collect_npy(self.labeldir)

        strain_map = {}
        for p in strain_files:
            idx = _extract_idx(p)
            if idx is not None:
                strain_map[idx] = p

        signal_map = {}
        for p in signal_files:
            idx = _extract_idx(p)
            if idx is not None:
                signal_map[idx] = p

        common = sorted(set(strain_map.keys()) & set(signal_map.keys()))
        pairs = [(strain_map[i], signal_map[i], i) for i in common]

        if len(pairs) == 0:
            raise RuntimeError(
                f"No paired files found. Check folders:\n  strain={self.datadir}\n  signal={self.labeldir}"
            )

        # 可选：打印一下配对数量
        print(f"[Mydata] paired files: {len(pairs)}")
        return pairs

    def __len__(self):
        return len(self.pairs) * self.num_segments

    def __getitem__(self, idx):
        file_idx = idx // self.num_segments
        which = idx % self.num_segments  # 0/1/2

        strain_path, signal_path, real_idx = self.pairs[file_idx]

        try:
            data = np.load(strain_path).astype(np.float32).reshape(-1)
            label = np.load(signal_path).astype(np.float32).reshape(-1)
        except Exception:
            return (None, None)

        # 长度检查（必须是 12291）
        if data.shape[0] != self.expected_len or label.shape[0] != self.expected_len:
            return (None, None)

        seg = self.segment_length

        # 取前三段（忽略尾巴3）
        data_main = data[:seg * self.num_segments]
        label_main = label[:seg * self.num_segments]

        # 切 3 段
        x_np = data_main[which * seg:(which + 1) * seg]
        y_np = label_main[which * seg:(which + 1) * seg]

        # strain 处理：mean/std + RMS normalize
        x = torch.from_numpy(x_np).float()
        x = (x - self.strain_mean) / (self.strain_std + 1e-12)
        x = rms_normalize(x, eps=1e-8)

        # signal 处理：scale + signed sqrt
        y = torch.from_numpy(y_np).float()
        y = y * self.signal_scale
        y = signed_sqrt_compress(y, eps=self.sqrt_eps)

        if not self.return_meta:
            return x, y

        # 读取尾巴 meta（idx, snr, shift）
        tail = data[-3:]  # strain 和 signal 尾巴理论一致，这里取 strain 的
        meta = {
            "idx": int(round(float(tail[0]))),
            "snr": float(tail[1]),
            "shift": int(round(float(tail[2]))),
            "file_idx": real_idx,
            "which": which,
            "strain_path": strain_path,
            "signal_path": signal_path,
        }
        return x, y
