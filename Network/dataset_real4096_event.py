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

def _extract_center_index(path: str):
    """
    segment_123456.npy -> 123456
    """
    base = os.path.basename(path)
    m = re.search(r'segment_(\d+)\.npy$', base)
    return int(m.group(1)) if m else None

class RealEvent4096Dataset(Dataset):
    """
    真实事件专用：每个事件文件夹里：
      strain/segment_*.npy
      signal/segment_*.npy
    每个 npy 长度就是 4096（没有 3段拼接、没有尾巴 meta）
    """
    def __init__(self,
                 event_name: str,
                 strain_dir: str,
                 signal_dir: str,
                 strain_mean=-5.067700e-05,
                 strain_std=4.512173e+01,
                 signal_scale=1e23,
                 sqrt_eps=0.0,
                 debug_verify=False,
                 debug_print_first=10):
        self.event_name = event_name
        self.strain_dir = strain_dir
        self.signal_dir = signal_dir

        self.strain_mean = float(strain_mean)
        self.strain_std = float(strain_std)
        self.signal_scale = float(signal_scale)
        self.sqrt_eps = float(sqrt_eps)

        self.debug_verify = bool(debug_verify)
        self.debug_print_first = int(debug_print_first)

        # 建立配对：按 center_index 对齐
        self.pairs = self._build_pairs()

        if self.debug_verify:
            self._debug_print_first_items(self.debug_print_first)

    def _collect_segment_files(self, folder):
        paths = []
        for f in os.listdir(folder):
            if f.endswith(".npy") and f.startswith("segment_"):
                paths.append(os.path.join(folder, f))
        return paths

    def _build_pairs(self):
        strain_files = self._collect_segment_files(self.strain_dir)
        signal_files = self._collect_segment_files(self.signal_dir)

        if len(strain_files) == 0 or len(signal_files) == 0:
            print(f"[SKIP] [{self.event_name}] Empty strain or signal directory.")
            return []

        strain_map = {}
        for p in strain_files:
            idx = _extract_center_index(p)
            if idx is not None:
                strain_map[idx] = p

        signal_map = {}
        for p in signal_files:
            idx = _extract_center_index(p)
            if idx is not None:
                signal_map[idx] = p

        common = sorted(set(strain_map.keys()) & set(signal_map.keys()))
        if len(common) == 0:
            # raise RuntimeError(f"[{self.event_name}] No paired segment_*.npy found")
            print(f"[SKIP] [{self.event_name}] No paired segment_*.npy found, skip this event.")
            return []

        pairs = [(strain_map[i], signal_map[i], i) for i in common]
        print(f"[{self.event_name}] paired segments: {len(pairs)} | center_index min={common[0]} max={common[-1]}")
        return pairs

    def _debug_print_first_items(self, n_items=10):
        n_items = min(n_items, len(self.pairs))
        print(f"\n===== {self.event_name} ORDER CHECK (first {n_items}) =====")
        for i in range(n_items):
            _, _, center = self.pairs[i]
            print(f"pairs[{i}] center_index={center}")
        print("====================================================\n")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        strain_path, signal_path, center_index = self.pairs[idx]

        try:
            x_np = np.load(strain_path).astype(np.float32).reshape(-1)
            y_np = np.load(signal_path).astype(np.float32).reshape(-1)
        except Exception:
            return None

        if x_np.shape[0] != 4096 or y_np.shape[0] != 4096:
            return None

        # strain: mean/std + RMS normalize
        x = torch.from_numpy(x_np).float()
        x = (x - self.strain_mean) / (self.strain_std + 1e-12)
        x = rms_normalize(x, eps=1e-8)

        # signal: scale + signed sqrt
        y = torch.from_numpy(y_np).float()
        y = y * self.signal_scale
        y = signed_sqrt_compress(y, eps=self.sqrt_eps)

        meta = {
            "event": self.event_name,
            "center_index": int(center_index),
            "strain_path": strain_path,
            "signal_path": signal_path,
        }
        return x, y, meta