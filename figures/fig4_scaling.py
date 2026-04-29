"""
Fig 4: Strong-scaling experiment.
Network: 784→512→256→10   (N = 535,818 gradient floats)
Metric: per-epoch wall time (average of epochs 2-5, excludes warmup).
algo: mpi_builtin

P=1  is missing (srun rejected on 2-node allocation).
     → fill in P1_TIME once teammate provides the number,
       or run: salloc -N1 -n1 --gpus=1 -t5:00 -A m4341 -q interactive -C gpu
               srun ... mlp_train --layers 784,512,256,10 --algo mpi_builtin ...
"""
import numpy as np
import matplotlib.pyplot as plt

# ── Measured data ───────────────────────────────────────────────────
# epoch 2-5 averages (seconds)
P_measured   = np.array([2, 4, 8])
T_measured   = np.array([0.1185, 0.1038, 0.0623])

# P=1 measured by teammate on nid001169 (epochs 2-5 avg)
P1_TIME = 0.1665

# ── Ideal linear speedup anchored at P=1 ───────────────────────────
T_at_P1  = P1_TIME                # 0.1665 s
P_ideal  = np.array([1, 2, 4, 8])
T_ideal  = T_at_P1 / P_ideal

# ── Compute timing breakdown (for stacked plot reference) ───────────
# compute / d2h+h2d / mpi  (epoch 2-5 avg, s)
breakdown = {
    1: dict(compute=0.0566, transfer=0.0596, mpi=0.0000),
    2: dict(compute=0.0284, transfer=0.0301, mpi=0.0379),
    4: dict(compute=0.0143, transfer=0.0151, mpi=0.0501),
    8: dict(compute=0.0072, transfer=0.0080, mpi=0.0357),
}

# ── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---- Left: epoch time + ideal line --------------------------------
ax = axes[0]

if P1_TIME is not None:
    P_all = np.array([1, 2, 4, 8])
    T_all = np.array([P1_TIME, *T_measured])
else:
    P_all = P_measured
    T_all = T_measured

ax.plot(P_ideal, T_ideal, 'k--', lw=1.5, alpha=0.55, label='Ideal (linear speedup)')
ax.plot(P_all,   T_all,   'bo-', lw=2.0, ms=8, label='Measured')

# speedup labels (relative to P=1)
for P, T in zip(P_all, T_all):
    sp = T_at_P1 / T
    ax.text(P, T - 0.008, f'{sp:.1f}×', ha='center', fontsize=8.5, color='#333')

ax.set_xlabel('Number of GPUs (P)', fontsize=12)
ax.set_ylabel('Epoch Time (s)', fontsize=12)
ax.set_title('Strong Scaling: 784→512→256→10', fontsize=12, fontweight='bold')
ax.set_xticks([1, 2, 4, 8])
ax.set_xlim(0.5, 9.5)
ax.set_ylim(0, 0.22)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

# ---- Right: stacked bar – time breakdown per P --------------------
ax = axes[1]
P_bar = [1, 2, 4, 8]
comp   = [breakdown[p]['compute']  for p in P_bar]
trans  = [breakdown[p]['transfer'] for p in P_bar]
comm   = [breakdown[p]['mpi']      for p in P_bar]

x = np.arange(len(P_bar))
w = 0.45
b1 = ax.bar(x, comp,  w, label='Compute',      color='#4393c3', edgecolor='#333', lw=0.7)
b2 = ax.bar(x, trans, w, bottom=comp,
            label='D↔H Transfer',               color='#92c5de', edgecolor='#333', lw=0.7)
b3 = ax.bar(x, comm,  w,
            bottom=[c+t for c,t in zip(comp,trans)],
            label='MPI AllReduce',               color='#d6604d', edgecolor='#333', lw=0.7)

ax.set_xticks(x); ax.set_xticklabels([f'P={p}' for p in P_bar], fontsize=10)
ax.set_xlabel('Number of GPUs (P)', fontsize=12)
ax.set_ylabel('Epoch Time (s)', fontsize=12)
ax.set_title('Time Breakdown per Epoch', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('Strong Scaling: MPI Overhead Limits Speedup',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig4_scaling.pdf', bbox_inches='tight')
plt.savefig('fig4_scaling.png', dpi=150, bbox_inches='tight')
print("Saved fig4_scaling.pdf / .png")
plt.show()
