"""
Fig 5: α–β cost model fit for MPI_Allreduce.
Uses P=4, mpi_builtin, steady-state (epochs 3-5 avg) measurements at three N.

T_allreduce ≈ α + β·N   (ms, floats)

Data points:
  Network 784→64→10:             N = 50,890  floats,  T = 5.97 ms
  Network 784→512→256→10:        N = 535,818 floats,  T = 50.1 ms
  Network 784→2048→2048→2048→10: N = 5,824,522 floats, T = 823.2 ms
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats    # linear regression with confidence

# ── Data ────────────────────────────────────────────────────────────
N_floats = np.array([50_890, 535_818, 5_824_522], dtype=float)
T_ms     = np.array([5.97,   50.1,    823.2])

network_labels = [
    '784→64→10\n(51K)',
    '784→512→256→10\n(536K)',
    '784→2048→…→10\n(5.82M)',
]

# ── Linear fit T = α + β·N ─────────────────────────────────────────
slope, intercept, r_value, p_value, std_err = stats.linregress(N_floats, T_ms)
beta_ms_per_float = slope          # ms / float
alpha_ms          = intercept      # ms (latency)

# Effective bandwidth: each float is 4 bytes; AllReduce sends 2(P-1)/P*N*4 bytes total
# For simplicity of the α-β model we report "bytes / slope" as effective end-to-end rate
P = 4
factor = 2 * (P-1) / P              # ≈ 1.5 for P=4 (ring formula)
bw_MBs = factor * 4.0 / (beta_ms_per_float * 1e-3) / 1e6   # MB/s

N_fit = np.linspace(0, N_floats.max() * 1.08, 400)
T_fit = intercept + slope * N_fit

# ── Plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

# Fit line
ax.plot(N_fit / 1e6, T_fit, 'r--', lw=2.0, alpha=0.8,
        label=f'Fit: T = {alpha_ms:.1f} + {slope*1e6:.1f}·N  ms  (N in M floats)\n'
              f'R² = {r_value**2:.4f}')

# Data points
scatter_colors = ['#1f78b4', '#33a02c', '#e31a1c']
for i, (n, t, lbl, c) in enumerate(zip(N_floats, T_ms, network_labels, scatter_colors)):
    ax.scatter(n/1e6, t, s=120, color=c, zorder=5,
               label=f'{lbl}  →  {t:.1f} ms')

ax.set_xlabel('Gradient Size  N  (million floats)', fontsize=12)
ax.set_ylabel('MPI AllReduce Time  (ms)', fontsize=12)
ax.set_title('α–β Bandwidth Model  (P=4, MPI Built-in, cross-node)',
             fontsize=12, fontweight='bold')

# Annotation box
textstr = (f'α (latency)  ≈ {alpha_ms:.1f} ms\n'
           f'β (BW term)  ≈ {slope*1e6:.2f} ms / M floats\n'
           f'Eff. BW      ≈ {bw_MBs:.0f} MB/s  (ring formula, P=4)')
ax.text(0.97, 0.05, textstr, transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff9e6',
                  edgecolor='#ccc', alpha=0.95))

ax.legend(fontsize=9.5, framealpha=0.9, loc='upper left')
ax.grid(alpha=0.3, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(-0.15, N_floats.max()/1e6 * 1.1)
ax.set_ylim(-50, T_ms.max() * 1.15)

plt.tight_layout()
plt.savefig('fig5_alpha_beta.pdf', bbox_inches='tight')
plt.savefig('fig5_alpha_beta.png', dpi=150, bbox_inches='tight')
print("Saved fig5_alpha_beta.pdf / .png")
print(f"\nFit results:")
print(f"  α = {alpha_ms:.2f} ms   β = {slope:.4e} ms/float")
print(f"  Effective bandwidth ≈ {bw_MBs:.1f} MB/s")
print(f"  R² = {r_value**2:.4f}")
plt.show()
