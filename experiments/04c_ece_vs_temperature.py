"""Figure: choosing the temperature.

Sweep the applied temperature T and plot ECE and NLL of the recalibrated
confidences. Both bottom out near T = beta (the true sharpness): below it the
model stays overconfident, above it it becomes underconfident. NLL is the
smooth objective temperature scaling actually minimizes; ECE is the
calibration metric we care about, and they bottom out together.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.calibration import CalibrationDataset, apply_temperature, ece, nll


def main() -> None:
    beta = 2.5
    ds = CalibrationDataset(n_items=16000, sharpness=beta, seed=0)
    Ts = np.linspace(0.5, 5.0, 91)
    eces = [ece(apply_temperature(ds.report_logit, T), ds.correct, n_bins=15)
            for T in Ts]
    nlls = [nll(apply_temperature(ds.report_logit, T), ds.correct) for T in Ts]

    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    ax2 = ax1.twinx()
    l1, = ax1.plot(Ts, eces, "-", color="#c44e52", linewidth=2, label="ECE")
    l2, = ax2.plot(Ts, nlls, "-", color="#3a7ebf", linewidth=2, label="NLL")
    ax1.axvline(beta, color="#888", linestyle=":", linewidth=2,
                label=f"true sharpness β={beta}")
    T_best = Ts[int(np.argmin(nlls))]
    ax1.axvline(T_best, color="#55a467", linestyle="--", linewidth=2,
                label=f"NLL-optimal T={T_best:.2f}")

    ax1.set_xlabel("applied temperature T")
    ax1.set_ylabel("ECE", color="#c44e52")
    ax2.set_ylabel("NLL", color="#3a7ebf")
    ax1.tick_params(axis="y", labelcolor="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#3a7ebf")
    ax1.set_title(
        "ECE and NLL both bottom out near T = β. Temperature scaling minimizes\n"
        "the smooth NLL; the calibration error follows it down.",
        fontsize=11)
    lines = [l1, l2]
    ax1.legend(handles=lines + [plt.Line2D([], [], color="#888", ls=":"),
                                plt.Line2D([], [], color="#55a467", ls="--")],
               labels=["ECE", "NLL", f"true β={beta}", f"NLL-optimal T={T_best:.2f}"],
               loc="upper right")
    ax1.grid(alpha=0.3); ax1.set_axisbelow(True)

    print(f"  NLL-optimal T={T_best:.3f} (true beta={beta})")
    fig.tight_layout()
    out = Path("figures/04c_ece_vs_temperature.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
