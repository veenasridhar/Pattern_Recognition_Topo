import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def _flow_to_rgb(fx, fy, max_mag):
    """Direction -> hue, magnitude -> brightness (Middlebury-style)."""
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(fy, fx)                      # -pi..pi
    h = (ang + np.pi) / (2 * np.pi)              # 0..1
    s = np.ones_like(h)
    v = np.clip(mag / (max_mag + 1e-8), 0, 1)
    return hsv_to_rgb(np.stack([h, s, v], axis=-1))


def _colorwheel(ax, n=128):
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r = np.sqrt(xx ** 2 + yy ** 2)
    rgb = _flow_to_rgb(xx, yy, 1.0)
    rgb[r > 1] = 1.0                              # white outside disk
    ax.imshow(rgb, origin="lower")
    ax.set_title("flow key\n(hue=dir, bright=mag)", fontsize=8)
    ax.axis("off")


def plot_registration_steps(warped, flows, template, save_path, batch):
    """
    warped   : [N, 1, H, W]  aligned images
    flows    : [N, 2, H, W]  pixel displacements (ch0=dx, ch1=dy)
    template : [1, H, W]      emergent template
    """
    os.makedirs(save_path, exist_ok=True)

    warped = np.asarray(warped)
    flows = np.asarray(flows)
    template = np.asarray(template)

    N = warped.shape[0]
    w = warped[:, 0]                              # [N, H, W]
    fx, fy = flows[:, 0], flows[:, 1]             # [N, H, W] each
    tmpl = template[0]                            # [H, W]
    H, W = tmpl.shape

    max_mag = float(np.sqrt(fx ** 2 + fy ** 2).max())
    mag_floor = max(max_mag, 1e-3)               # avoid /0 when flow is all-zero
    step = max(H // 14, 1)
    ys, xs = np.mgrid[0:H:step, 0:W:step]

    fig, axes = plt.subplots(2, N + 1, figsize=(2.3 * (N + 1), 4.8))

    for i in range(N):
        axes[0, i].imshow(w[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"warped {i}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(_flow_to_rgb(fx[i], fy[i], max_mag), origin="upper")
        if max_mag > 1e-6:                        # nothing to draw for zero flow
            axes[1, i].quiver(xs, ys, fx[i][::step, ::step], fy[i][::step, ::step],
                              color="k", scale=mag_floor * len(xs[0]) * 1.2,
                              width=0.004, alpha=0.6)
        axes[1, i].set_title(f"flow {i}  |max|={np.sqrt(fx[i]**2+fy[i]**2).max():.1f}px",
                             fontsize=8)
        axes[1, i].axis("off")

    # last column: template on top, flow key below
    axes[0, N].imshow(tmpl, cmap="gray", vmin=0, vmax=1)
    axes[0, N].set_title("template\n(emergent)", fontsize=9)
    axes[0, N].axis("off")
    _colorwheel(axes[1, N])

    fig.suptitle(f"Registration steps — batch {batch}   (max disp {max_mag:.1f}px)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{save_path}/registration_{batch}.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # smoke test with a synthetic aligned stack + smooth flows
    np.random.seed(0)
    N, H, W = 5, 96, 128
    yy, xx = np.mgrid[0:H, 0:W]
    base = ((np.abs(xx - W / 2) < 8) | (np.abs(yy - H / 2) < 8)).astype(np.float32)
    warped = np.stack([base + 0.05 * np.random.randn(H, W) for _ in range(N)])[:, None]
    warped = warped.clip(0, 1)
    flows = np.zeros((N, 2, H, W), np.float32)
    for i in range(N):
        flows[i, 0] = (i - 2) * 1.5 * np.sin(yy / 20)   # dx
        flows[i, 1] = (i - 2) * 1.0 * np.cos(xx / 25)   # dy
    template = warped.mean(0)                            # [1, H, W]

    plot_registration_steps(warped, flows, template, "/home/claude/out", batch=0)
    print("saved:", os.path.exists("/home/claude/out/registration_0.png"))