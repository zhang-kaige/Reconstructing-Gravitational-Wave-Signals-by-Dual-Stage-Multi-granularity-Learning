import os
import gc
import numpy as np
import pandas as pd

from pycbc.waveform import get_td_waveform
from pycbc.psd.analytical import aLIGOAdVO4T1800545
from pycbc.noise import noise_from_psd
from pycbc.filter import make_frequency_series, sigmasq
from pycbc.filter import highpass, lowpass
from pycbc.types import TimeSeries
from pycbc.detector import Detector


# =============================
# PSD 0 修复
# =============================
def replace_zeros_with_neighbors(data, min_non_zero_value):
    for i in range(len(data)):
        if data[i] == 0:
            if i > 0 and data[i - 1] > 0:
                data[i] = data[i - 1]
            elif i < len(data) - 1 and data[i + 1] > 0:
                data[i] = data[i + 1]
            else:
                data[i] = min_non_zero_value
    return data


# =============================
# 白化：按你的逻辑
# strain: highpass -> crop(2,2) -> /sqrt(psd) -> highpass -> lowpass
# signal: 只 crop(2,2)，不做 highpass（与你原函数一致）
# =============================
def whiten_like_yours(strain_ts: TimeSeries,
                      signal_ts: TimeSeries,
                      sample_rate=4096,
                      low_frequency_cutoff=20,
                      bandpass_low=20,
                      bandpass_high=2047,
                      crop_sec=2):
    # strain
    strain_hp = highpass(strain_ts, low_frequency_cutoff)
    conditioned_strain = strain_hp.crop(crop_sec, crop_sec)

    # ostrain：原始 strain 只 crop，不做 highpass（按你上面代码逻辑）
    ostrain = strain_ts.crop(crop_sec, crop_sec)

    # signal：只 crop，不 highpass（与你原代码一致）
    ori = signal_ts.crop(crop_sec, crop_sec)

    # PSD 长度要匹配频域 N//2+1
    N = len(conditioned_strain)
    duration = N * (1.0 / sample_rate)
    psd = aLIGOAdVO4T1800545(
        N // 2 + 1,
        delta_f=1.0 / duration,
        low_freq_cutoff=low_frequency_cutoff
    )

    # 防 0
    min_non_zero = np.min(psd.data[psd.data > 0])
    psd.data = replace_zeros_with_neighbors(psd.data, min_non_zero)

    # whiten strain
    white = (conditioned_strain.to_frequencyseries() / psd**0.5).to_timeseries()
    white = highpass(white, bandpass_low)
    white = lowpass(white, bandpass_high)

    return white, ostrain, ori, psd


# =============================
# 第一种：用 antenna_pattern 做投影（不做 dt 平移）
# ht = fp * hp + fc * hc
# =============================
def project_to_ifo_simple(ifo: str,
                          hp: TimeSeries,
                          hc: TimeSeries,
                          ra: float,
                          dec: float,
                          pol: float,
                          t0: float = 1368975639.0):
    det = Detector(ifo)

    # antenna_pattern 一定要 time（t_gps）
    fp, fc = det.antenna_pattern(ra, dec, pol, t0)

    ht = fp * hp + fc * hc

    # 保证时间轴和 hp/hc 一致（不引入 dt）
    ht.start_time = hp.start_time
    return ht


# =============================
# 三段 4096：重叠版，且间隔一致
# overlap_ratio 随机 ∈ [2/3, 1)
# shift = 4096 - overlap
# starts: start0, start0+shift, start0+2*shift
# 尾部追加 [idx, snr, shift]  —— 注意：这里 idx 保持 CSV idx（不做 offset）
# =============================
def make_3_segments_concat_overlap(signal_ts: TimeSeries,
                                  strain_ts: TimeSeries,
                                  idx_val: int,
                                  snr_val: float,
                                  segment_length=4096,
                                  min_peak_pos=1500,
                                  max_peak_pos=2500,
                                  min_overlap_ratio=2/3,
                                  rng=None,
                                  ostrain_ts: TimeSeries = None
                                  ):
    if rng is None:
        rng = np.random.default_rng()

    sig = np.asarray(signal_ts)
    stn = np.asarray(strain_ts)
    if len(sig) != len(stn):
        raise ValueError("signal and strain lengths mismatch after preprocessing")

    if ostrain_ts is not None:
            ostn = np.asarray(ostrain_ts)
            if len(sig) != len(ostn):
                raise ValueError("signal and ostrain lengths mismatch after preprocessing")

    N = len(sig)
    if N < segment_length:
        raise ValueError(f"too short: N={N} < {segment_length}")

    # overlap_length in [ceil(2/3*L), L-1]
    min_overlap = int(np.ceil(segment_length * min_overlap_ratio))
    if min_overlap >= segment_length:
        min_overlap = segment_length - 1

    overlap_length = int(rng.integers(min_overlap, segment_length))  # [min_overlap, segment_length-1]
    shift = segment_length - overlap_length                          # > 0

    total_needed = segment_length + 2 * shift
    if N < total_needed:
        raise ValueError(f"too short for 3 overlapped segments: N={N} < needed={total_needed}")

    # 峰定位（用 signal）
    peak_index = int(np.argmax(np.abs(sig)))
    desired_pos = int(rng.integers(min_peak_pos, max_peak_pos + 1))

    start0 = peak_index - desired_pos
    start0 = max(0, start0)

    if start0 + total_needed > N:
        start0 = N - total_needed

    starts = [start0 + k * shift for k in range(3)]
    ends = [s + segment_length for s in starts]

    sig_segs = [sig[s:e] for s, e in zip(starts, ends)]
    stn_segs = [stn[s:e] for s, e in zip(starts, ends)]

    # 尾部：idx, snr, shift（idx 保持 CSV idx，不做 offset）
    tail = np.array([float(idx_val), float(snr_val), float(shift)], dtype=float)

    sig_concat = np.concatenate(sig_segs + [tail])
    stn_concat = np.concatenate(stn_segs + [tail])

    if ostrain_ts is not None:
        ostn_segs = [ostn[s:e] for s, e in zip(starts, ends)]
        ostn_concat = np.concatenate(ostn_segs + [tail])
        return sig_concat, stn_concat, ostn_concat, shift, overlap_length

    return sig_concat, stn_concat, shift, overlap_length


def main():
    # ============ 你需要改的路径 ============
    PARAMS_CSV = ""

    OUT_BASE = "/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/150_mass_low_high_snr30/low"
    strain_dir = os.path.join(OUT_BASE, "strain")
    signal_dir = os.path.join(OUT_BASE, "signal")
    ostrain_dir = os.path.join(OUT_BASE, "ostrain")
    os.makedirs(strain_dir, exist_ok=True)
    os.makedirs(signal_dir, exist_ok=True)
    os.makedirs(ostrain_dir, exist_ok=True)

    # ============ 固定参数 ============
    sample_rate = 4096
    delta_t = 1.0 / sample_rate
    f_lower = 20.0
    approximant = "IMRPhenomD"

    bandpass_high = 2047
    crop_sec = 2
    segment_length = 4096

    # ✅ 全局随机发生器（保持你原逻辑）
    rng = np.random.default_rng(27)

    df = pd.read_csv(PARAMS_CSV)

    # 固定一个时间（只用于 antenna_pattern 的地球自转角）
    t0 = 1368975639.0

    # =========================================================
    # ✅ 只影响“保存文件名”的 idx 偏移（CSV idx 不变）
    # 例如你要从 300W 开始命名：3_000_000
    # 这样 CSV idx=0 -> 文件名 signal_3000000.npy
    # =========================================================
    IDX_OFFSET = 0

    # =========================================================
    # ✅ 从 CSV 的哪个 idx 开始生成 & 断点续跑（按 CSV idx 生效）
    # =========================================================
    START_CSV_IDX = 0        # 只处理 CSV idx >= START_CSV_IDX

    # 断点续跑模式：
    # - "auto" : 扫描输出文件名，推回 CSV idx，自动继续
    # - "manual": 手动指定 RESUME_FROM_CSV_IDX（从它的下一个继续）
    RESUME_MODE = "manual"
    RESUME_FROM_CSV_IDX = -1    # manual 模式生效：例如 12345 表示从 12346 继续

    def parse_saved_fileidx(fn: str, prefix: str):
        # 识别 signal_123.npy / strain_123.npy
        if not (fn.startswith(prefix) and fn.endswith(".npy")):
            return None
        s = fn[len(prefix):-4]
        if not s.isdigit():
            return None
        return int(s)

    def find_max_completed_csv_idx(signal_dir, strain_dir, idx_offset):
        """
        扫描 signal/strain 目录的文件名编号（file_idx），换算成 csv_idx=file_idx-idx_offset，
        返回已完成的最大 csv_idx。
        若目录里存在不带 offset 的旧文件名，也能兼容（自动尝试两种解释）。
        """
        def scan_dir(d, prefix):
            vals = []
            if not os.path.isdir(d):
                return vals
            for fn in os.listdir(d):
                v = parse_saved_fileidx(fn, prefix)
                if v is not None:
                    vals.append(v)
            return vals

        sig_fileidx = set(scan_dir(signal_dir, "signal_"))
        stn_fileidx = set(scan_dir(strain_dir, "strain_"))

        if not sig_fileidx and not stn_fileidx:
            return -1

        # 只把两边都存在的当“完成”，更安全
        common_fileidx = sig_fileidx & stn_fileidx
        if not common_fileidx:
            # 如果两边没有交集，就退而求其次用各自最大值（不太建议，但防止你目录不齐）
            common_fileidx = sig_fileidx or stn_fileidx

        # 将 file_idx -> csv_idx：优先认为 file_idx 是 offset 后的
        csv_candidates = []
        for file_idx in common_fileidx:
            csv1 = file_idx - idx_offset          # offset 命名
            csv2 = file_idx                        # 兼容旧命名（无 offset）
            if csv1 >= 0:
                csv_candidates.append(csv1)
            csv_candidates.append(csv2)

        # 只保留合理范围（>=0）
        csv_candidates = [c for c in csv_candidates if c >= 0]
        return max(csv_candidates) if csv_candidates else -1

    if RESUME_MODE.lower() == "auto":
        RESUME_FROM = find_max_completed_csv_idx(signal_dir, strain_dir, IDX_OFFSET)
        print(f"[resume:auto] detected max completed CSV idx = {RESUME_FROM}, will start from CSV idx > {RESUME_FROM}")
    else:
        RESUME_FROM = int(RESUME_FROM_CSV_IDX)
        print(f"[resume:manual] will start from CSV idx > {RESUME_FROM}")

    print(f"[csv-range] START_CSV_IDX={START_CSV_IDX}, RESUME_FROM={RESUME_FROM}, IDX_OFFSET(for filename)={IDX_OFFSET}")

    # ✅ 原子保存：修复 np.save 自动补 .npy 导致 replace 找不到文件的问题
    def atomic_save(path, arr):
        tmp = path + ".tmp"
        np.save(tmp, arr)                 # 实际写入 tmp + ".npy"
        os.replace(tmp + ".npy", path)    # 原子替换

    for _, row in df.iterrows():
        idx_val = int(row["idx"])  # CSV idx（0..999999）

        # =====================================================
        # ✅ 只按 CSV idx 控制从哪里开始/断点续跑
        # =====================================================
        if idx_val < START_CSV_IDX:
            continue
        if idx_val <= RESUME_FROM:
            continue
        # =====================================================

        m1 = float(row["mass1"])
        m2 = float(row["mass2"])
        distance = float(row["distance"])
        ra = float(row["ra"])
        dec = float(row["dec"])
        pol = float(row["pol"])
        ifo = str(row["ifo"]).strip()

        try:
            # 1) 生成 hp, hc
            hp, hc = get_td_waveform(
                approximant=approximant,
                mass1=m1, mass2=m2,
                distance=distance,
                delta_t=delta_t,
                f_lower=f_lower
            )

            # 2) 第一种投影：ht = fp*hp + fc*hc
            h_ifo = project_to_ifo_simple(ifo, hp, hc, ra, dec, pol, t0=t0)

            # 3) 构造 PSD 并计算 rho0
            N = len(h_ifo)
            duration = N * delta_t
            psd = aLIGOAdVO4T1800545(
                N // 2 + 1,
                delta_f=1.0 / duration,
                low_freq_cutoff=f_lower
            )

            h_tilde = make_frequency_series(h_ifo)
            rho0_sq = sigmasq(
                h_tilde,
                psd=psd,
                low_frequency_cutoff=f_lower,
                high_frequency_cutoff=sample_rate / 2.0
            )
            rho0 = float(np.sqrt(rho0_sq))
            if rho0 <= 0:
                raise ValueError("rho0 <= 0")

            # 4) 目标 SNR：整数 1..30
            # target_snr = int(rng.integers(1, 31))
            target_snr = 30


            # 5) 缩放信号幅度
            alpha = target_snr / rho0
            signal_ts = h_ifo * alpha

            # 6) 生成噪声并叠加 raw strain
            noise = noise_from_psd(N, delta_t, psd)
            noise.start_time = signal_ts.start_time
            strain_raw = signal_ts + noise

            # 7) 白化（按你的逻辑）
            strain_white, ostrain, ori, _psd_w = whiten_like_yours(
                strain_raw, signal_ts,
                sample_rate=sample_rate,
                low_frequency_cutoff=f_lower,
                bandpass_low=20,
                bandpass_high=bandpass_high,
                crop_sec=crop_sec
            )

            # 8) 长度对齐保险
            L = min(len(ori), len(strain_white))
            ori = ori[:L]
            strain_white = strain_white[:L]
            ostrain = ostrain[:L]

            # 9) 三段重叠拼接 + 尾部[idx, snr, shift]（idx 仍用 CSV idx）
            sig_concat, stn_concat, ostn_concat,  shift, overlap = make_3_segments_concat_overlap(
                ori, strain_white,
                idx_val=idx_val,
                snr_val=target_snr,
                segment_length=segment_length,
                rng=rng,
                ostrain_ts=ostrain
            )

            # 10) 保存：只影响文件名编号（CSV idx + offset）
            file_idx = idx_val + IDX_OFFSET
            sig_path = os.path.join(signal_dir, f"signal_{file_idx}.npy")
            stn_path = os.path.join(strain_dir, f"strain_{file_idx}.npy")
            ostn_path = os.path.join(ostrain_dir, f"ostrain_{file_idx}.npy")


            atomic_save(sig_path, sig_concat.astype(np.float32))
            atomic_save(stn_path, stn_concat.astype(np.float32))
            atomic_save(ostn_path, ostn_concat.astype(np.float32))


            # 最终长度：3*4096 + 3 = 12291
            print(f"Saved CSV idx={idx_val} -> file idx={file_idx} ifo={ifo} snr={target_snr} "
                  f"alpha={alpha:.3e} shift={shift} overlap={overlap} len={len(sig_concat)}")

        except Exception as e:
            print(f"[skip] CSV idx={idx_val} failed: {e}")

        gc.collect()

    print("Done.")


if __name__ == "__main__":
    main()
