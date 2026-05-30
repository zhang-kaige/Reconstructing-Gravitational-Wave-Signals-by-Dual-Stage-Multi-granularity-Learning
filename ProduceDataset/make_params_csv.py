import os
import numpy as np
import pandas as pd

OUT_CSV = ""

# sample size
N_SAMPLES = 10000
# seed number
SEED = 988

# 固定范围
MASS_MIN, MASS_MAX = 3.0, 150.0        # Msun
DIST_MIN, DIST_MAX = 100.0, 10000.0    # Mpc

def sample_sky_isotropic(n: int, rng: np.random.Generator):
    """
    返回弧度制：
      ra  ~ Uniform(0, 2pi)
      dec: 天球均匀 => sin(dec) ~ Uniform(-1, 1)
      pol: 极化角 psi 有 pi 周期 => Uniform(0, pi)
    """
    ra = rng.uniform(0.0, 2.0 * np.pi, size=n)
    u = rng.uniform(-1.0, 1.0, size=n)          # u = sin(dec)
    dec = np.arcsin(u)
    pol = rng.uniform(0.0, np.pi, size=n)       # 0..pi 覆盖不重复物理
    return ra, dec, pol

def main():
    rng = np.random.default_rng(SEED)

    # 质量独立均匀采样后排序，保证 mass1 >= mass2（可选但建议）
    m1 = rng.uniform(MASS_MIN, MASS_MAX, size=N_SAMPLES)
    m2 = rng.uniform(MASS_MIN, MASS_MAX, size=N_SAMPLES)
    mass1 = np.maximum(m1, m2)
    mass2 = np.minimum(m1, m2)

    # 距离线性均匀
    distance = rng.uniform(DIST_MIN, DIST_MAX, size=N_SAMPLES)

    # 天球参数（弧度）
    ra, dec, pol = sample_sky_isotropic(N_SAMPLES, rng)

    # IFO：保证一半 H1，一半 L1（奇数时 L1 多 1）
    n_h1 = N_SAMPLES // 2
    n_l1 = N_SAMPLES - n_h1
    ifos = np.array(["H1"] * n_h1 + ["L1"] * n_l1)
    rng.shuffle(ifos)

    out = pd.DataFrame({
        "idx": np.arange(N_SAMPLES, dtype=int),
        "mass1": mass1.astype(float),
        "mass2": mass2.astype(float),
        "distance": distance.astype(float),
        "ra": ra.astype(float),
        "dec": dec.astype(float),
        "pol": pol.astype(float),
        "ifo": ifos
    })

    out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {os.path.abspath(OUT_CSV)}")
    print(f"H1={np.sum(out['ifo']=='H1')}, L1={np.sum(out['ifo']=='L1')}")
    print(f"mass1 range: [{out['mass1'].min():.3f}, {out['mass1'].max():.3f}]")
    print(f"mass2 range: [{out['mass2'].min():.3f}, {out['mass2'].max():.3f}]")
    print(f"distance range: [{out['distance'].min():.3f}, {out['distance'].max():.3f}]")
    print("Angles are in radians.")

if __name__ == "__main__":
    main()
