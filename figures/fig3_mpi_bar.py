"""
Fig 3: Per-epoch MPI AllReduce time by algorithm, network size, and rank count.
Data: epoch 3-5 average (steady state, excludes warmup).

Network sizes:
  Small: 784→64→10   N = 50,890 floats
  Large: 784→2048→2048→2048→10   N = 5,824,522 floats
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Data (epoch 3-5 average, ms) ───────────────────────────────────
# Layout: [mpi_builtin, ring, tree]
mpi_data = {
    ('Small', 'P=4'): [5.97,  6.10,  13.67],
    ('Small', 'P=8'): [5.27,  7.27,   8.37],
    ('Large', 'P=4'): [823.2, 688.5, 2696.7],
    ('Large', 'P=8'): [603.8, 482.1, 1569.9],
}

algos  = ['MPI Built-in', 'Ring AllReduce', 'Tree Reduce']
colors = ['#4393c3', '#4dac26', '#d6604d']

# ── Two-panel layout (left=Small, right=Large) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

for col, net in enumerate(['Small', 'Large']):
    ax = axes[col]
    P_labels = ['P=4', 'P=8']
    x = np.arange(len(P_labels))
    width = 0.25

    for i, (algo, color) in enumerate(zip(algos, colors)):
        vals = [mpi_data[(net, p)][i] for p in P_labels]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=algo, color=color, alpha=0.88, edgecolor='#333', linewidth=0.7)
        # value labels on bars
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            if net == 'Large':
                label = f'{v/1000:.2f}s' if v >= 1000 else f'{v:.0f}ms'
            else:
                label = f'{v:.1f}ms'
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.04,
                    label, ha='center', va='bottom', fontsize=7.5, color='#222')

    ax.set_xticks(x)
    ax.set_xticklabels(P_labels, fontsize=11)
    ax.set_xlabel('# GPUs', fontsize=11)
    ax.set_ylabel('MPI Time per Epoch (ms)', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    if net == 'Small':
        ax.set_title(f'Small Network  (N ≈ 51K floats)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 22)
    else:
        ax.set_title(f'Large Network  (N ≈ 5.8M floats)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 3200)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f'{v/1000:.1f}s' if v >= 1000 else f'{v:.0f}ms'))

    if col == 0:
        ax.legend(fontsize=10, framealpha=0.9)

fig.suptitle('AllReduce MPI Time: Ring vs Tree vs MPI Built-in',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig3_mpi_bar.pdf', bbox_inches='tight')
plt.savefig('fig3_mpi_bar.png', dpi=150, bbox_inches='tight')
print("Saved fig3_mpi_bar.pdf / .png")
plt.show()
