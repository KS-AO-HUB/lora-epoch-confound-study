"""Reproduce every summary statistic quoted in README.md from data/lora_metrics.csv.

Usage:  python analysis/robustness_checks.py
Needs:  numpy, pandas
"""
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "lora_metrics.csv"
METRICS = ["clip_identity", "lpips_identity", "diversity_lpips", "fid_vs_refs"]
N_PERM = 20_000
SEED = 0

df = pd.read_csv(CSV)
df["log_data"] = np.log2(df["data_size"])
df["log_batch"] = np.log2(df["batch_size"])
df["log_epochs"] = np.log2(df["epochs"])

print("== Group means by dataset size ==")
print(df.groupby("data_size")[METRICS].mean().sort_index(ascending=False).round(4).to_string())
print("\n== Group means by batch size ==")
print(df.groupby("batch_size")[METRICS].mean().round(4).to_string())

print("\n== Pearson r with log(epochs) and log(batch size) ==")
for m in METRICS:
    print(f"{m:<16} epochs {df['log_epochs'].corr(df[m]):+.3f}   batch {df['log_batch'].corr(df[m]):+.3f}")

def ols_r2(cols, target):
    """Ordinary least squares with an intercept; returns (R2, adjusted R2)."""
    X = np.column_stack([np.ones(len(df))] + [c.to_numpy() for c in cols])
    yv = df[target].to_numpy()
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    r2 = 1 - resid @ resid / np.sum((yv - yv.mean()) ** 2)
    n, k = X.shape
    return r2, 1 - (1 - r2) * (n - 1) / (n - k)


print("\n== Regressions (R2 / adjusted R2) ==")
for m in METRICS:
    a = ols_r2([df["log_data"], df["log_batch"], df["log_data"] * df["log_batch"]], m)
    b = ols_r2([df["log_epochs"]], m)
    print(f"{m:<16} full grid {a[0]:.3f} / {a[1]:+.3f}   epochs only {b[0]:.3f} / {b[1]:+.3f}")

y = df["clip_identity"].to_numpy()
x = df["log_epochs"].to_numpy()
r = np.corrcoef(x, y)[0, 1]
rng = np.random.default_rng(SEED)
perm = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(N_PERM)])
p = (np.sum(np.abs(perm) >= abs(r)) + 1) / (N_PERM + 1)
print(f"\n== CLIP vs log(epochs): r = {r:+.3f}, two-sided permutation p = {p:.5f} ({N_PERM} shuffles) ==")

for size in (25, 400):
    keep = df["data_size"] != size
    print(f"r without the {size}-image runs: {np.corrcoef(x[keep], y[keep])[0, 1]:+.3f}")

g = df.groupby("data_size")["clip_identity"]
within_sd = g.std(ddof=1)
gap = g.mean()[25] - g.mean()[400]
print(f"\nCLIP gap, 25 vs 400 images: {gap:.4f}")
print("Within-group SD:", within_sd.round(4).to_dict(), f"(mean {within_sd.mean():.4f})")
print(f"Gap / mean within-group SD: {gap / within_sd.mean():.1f}")

print("\n== Run 14 rank on each metric (1 = most similar / most diverse) ==")
for m, higher_better in [("clip_identity", True), ("lpips_identity", False), ("fid_vs_refs", False), ("diversity_lpips", True)]:
    order = df.sort_values(m, ascending=not higher_better)["run_id"].tolist()
    print(f"{m:<16} rank {order.index('run_14') + 1}")
