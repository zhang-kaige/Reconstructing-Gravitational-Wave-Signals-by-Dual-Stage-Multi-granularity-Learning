import os
import h5py
import numpy as np
import pandas as pd

from pycbc.waveform import get_td_waveform
from pycbc.filter import highpass, lowpass
from pycbc.psd import interpolate, inverse_spectrum_truncation, welch
from pycbc.types import TimeSeries
from pycbc.detector import Detector


# =============================
# PSD 0 修复（保留你的逻辑）
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
# 从 filename 解析 IFO
# 例：H-H1_... -> H1, L-L1_... -> L1, V-V1_... -> V1
# =============================
def ifo_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    head = base.split("_")[0]  # e.g. "H-H1"
    parts = head.split("-")
    if len(parts) >= 2 and parts[1]:
        return parts[1]        # "H1" / "L1" / "V1"
    raise ValueError(f"Cannot parse ifo from filename: {filename}")


# =============================
# 投影：ht = fp*hp + fc*hc
# =============================
def project_to_ifo_simple(
    ifo: str,
    hp: TimeSeries,
    hc: TimeSeries,
    ra: float,
    dec: float,
    pol: float,
    t0: float
):
    det = Detector(ifo)
    fp, fc = det.antenna_pattern(ra, dec, pol, t0)
    ht = fp * hp + fc * hc
    ht.start_time = hp.start_time
    return ht


# =============================
# 把模板信号截成固定 4096（以峰值为中心）
# =============================
def crop_template_to_4096(signal_ts: TimeSeries, segment_length=4096) -> TimeSeries:
    sig = np.asarray(signal_ts)
    if len(sig) < segment_length:
        raise ValueError(f"Template too short: len={len(sig)} < {segment_length}")

    peak_index = int(np.argmax(np.abs(sig)))
    half = segment_length // 2

    start = max(0, peak_index - half)
    end = start + segment_length
    if end > len(sig):
        end = len(sig)
        start = end - segment_length

    return signal_ts[start:end]


def _get_first_present_float(row, candidates, default=None):
    """
    从一组可能的列名中，找到第一个存在且可转 float 的值。
    """
    for k in candidates:
        if k in row and pd.notna(row[k]):
            try:
                return float(row[k])
            except Exception:
                pass
    return default


def main():
    # ==========================================================
    # 你需要改的 3 个路径
    # ==========================================================
    PARAMS_CSV = "/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/updated_pe.csv"
    HDF5_BASE_DIR = "/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Data/RealEventAllSeq"
    OUT_BASE = "/mnt/3.6T/zkg/Development/VSCode/Projects/DoubleSeg/Util/更新版/Data/RealData_20_2048_RealGpsTime_ostrain"
    # ==========================================================

    # 固定参数（按你原逻辑）
    f_lower = 20.0
    crop_sec = 2
    bandpass_high = 2047
    segment_length = 4096
    half_length = segment_length // 2

    df = pd.read_csv(PARAMS_CSV)

    # 逐行处理 CSV
    for _, row in df.iterrows():
        event = str(row["event"]).strip()

        m1 = float(row["mass1_source_Msun"])
        m2 = float(row["mass2_source_Msun"])
        distance = float(row["luminosity_distance_Mpc"])

        # 从 CSV 读 ra/dec/pol（单位：弧度）
        ra = _get_first_present_float(row, ["ra", "ra_rad", "right_ascension"])
        dec = _get_first_present_float(row, ["dec", "dec_rad", "declination"])
        pol = _get_first_present_float(row, ["psi", "pol", "psi_rad", "pol_rad", "polarization"])

        if ra is None or dec is None or pol is None:
            print(f"[SKIP] {event} missing ra/dec/pol in CSV")
            continue

        # ====== 【唯一新增】从 CSV 读取每个事件的 GPS 时间 ======
        t0 = _get_first_present_float(row, ["GPSTime", "gps_time", "gps"])
        if t0 is None:
            print(f"[SKIP] {event} missing GPSTime in CSV")
            continue
        # ========================================================

        # 你已确认：start_idx/end_idx 是 crop(2,2) 后可直接用的索引
        start_idx = int(row["start_idx"])
        end_idx = int(row["end_idx"])

        filename = str(row["filename"]).strip()
        file_path = filename if os.path.isabs(filename) else os.path.join(HDF5_BASE_DIR, filename)

        # 输出目录（按 event 分目录）
        out_strain_dir = os.path.join(OUT_BASE, event, "strain")
        out_signal_dir = os.path.join(OUT_BASE, event, "signal")
        out_ostrain_dir = os.path.join(OUT_BASE, event, "ostrain")
        os.makedirs(out_strain_dir, exist_ok=True)
        os.makedirs(out_signal_dir, exist_ok=True)
        os.makedirs(out_ostrain_dir, exist_ok=True)

        try:
            # 1) 读 hdf5 strain
            with h5py.File(file_path, "r") as f:
                strain_data = f["strain/Strain"][:]

            sample_rate = 4096
            delta_t = 1 / sample_rate

            ts_strain = TimeSeries(strain_data, delta_t=delta_t)

            # 2) 生成 hp, hc
            hp, hc = get_td_waveform(
                approximant="IMRPhenomD",
                mass1=m1,
                mass2=m2,
                distance=distance,
                delta_t=delta_t,
                f_lower=f_lower
            )

            # 3) IFO 从 filename 解析 + 投影（使用事件自己的 GPSTime）
            ifo = ifo_from_filename(filename)
            h_ifo = project_to_ifo_simple(
                ifo=ifo,
                hp=hp,
                hc=hc,
                ra=ra,
                dec=dec,
                pol=pol,
                t0=t0
            )

            # 4) 转成 TimeSeries，并且只 crop(2,2)
            ts_signal = TimeSeries(h_ifo, delta_t=delta_t)

            # ============ 白化：完全保留你的 welch 流程 ============
            strain_hp = highpass(ts_strain, 20)
            conditioned_strain = strain_hp.crop(crop_sec, crop_sec)

            ostrain = ts_strain.crop(crop_sec, crop_sec)
            ori = ts_signal.crop(crop_sec, crop_sec)

            seg_len = int(4 * sample_rate)
            seg_stride = seg_len // 2

            psd = welch(conditioned_strain, seg_len=seg_len, seg_stride=seg_stride)
            psd = interpolate(psd, conditioned_strain.delta_f)
            psd = inverse_spectrum_truncation(
                psd,
                int(4 * conditioned_strain.sample_rate),
                low_frequency_cutoff=20.0
            )

            if (psd.data == 0).any():
                min_non_zero = np.min(psd.data[psd.data > 0])
                psd.data = replace_zeros_with_neighbors(psd.data, min_non_zero)

            white_strain = (conditioned_strain.to_frequencyseries() / psd**0.5).to_timeseries()
            white_strain = highpass(white_strain, 20)
            white_strain = lowpass(white_strain, bandpass_high)
            # =======================================================

            # 5) 对齐长度保险
            L = min(len(white_strain), len(ostrain))
            white_strain = white_strain[:L]
            ostrain = ostrain[:L]

            # 6) 模板信号固定裁成 4096
            ori_4096 = crop_template_to_4096(ori, segment_length=segment_length)

            # 7) 逐点滑窗保存
            saved = 0
            for center_index in range(start_idx, end_idx + 1):
                segment_start = center_index - half_length
                segment_end = center_index + half_length

                if segment_start < 0 or segment_end > len(white_strain):
                    continue

                seg_white = white_strain[segment_start:segment_end]
                seg_ostrain = ostrain[segment_start:segment_end]

                np.save(os.path.join(out_strain_dir, f"segment_{center_index}.npy"),
                        np.asarray(seg_white))
                np.save(os.path.join(out_signal_dir, f"segment_{center_index}.npy"),
                        np.asarray(ori_4096))
                np.save(os.path.join(out_ostrain_dir, f"segment_{center_index}.npy"),
                        np.asarray(seg_ostrain))

                saved += 1

            print(f"[OK] {event} ifo={ifo} saved={saved} "
                  f"idx=[{start_idx},{end_idx}] file={os.path.basename(file_path)} "
                  f"ra={ra:.6f} dec={dec:.6f} psi={pol:.6f} t0={t0:.3f}")

        except Exception as e:
            print(f"[SKIP] {event} failed: {e}")


if __name__ == "__main__":
    main()
