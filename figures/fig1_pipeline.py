"""
Fig 1: Data-parallel SGD pipeline diagram.
Shows how data is split across GPUs and how gradients are aggregated.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis('off')

# ── color palette ──────────────────────────────────────────────────
C_DATA   = '#AED6F1'   # light blue  – dataset / shard
C_GPU    = '#A9DFBF'   # light green – GPU compute
C_COMM   = '#F9E79F'   # yellow      – AllReduce
C_UPD    = '#F1948A'   # salmon      – weight update

def box(ax, x, y, w, h, color, label, fontsize=9):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='#444', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', wrap=True)

def arrow(ax, x1, y1, x2, y2, label='', color='#555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.18, label, ha='center', fontsize=8, color=color)

# ── Step 0: Dataset ────────────────────────────────────────────────
box(ax, 0.15, 1.5, 1.5, 2.0, C_DATA, 'Fashion-\nMNIST\n60K samples', fontsize=8)

# ── Step 1: Scatter (partition) ────────────────────────────────────
arrow(ax, 1.65, 2.5, 2.1, 2.5, '')
ax.text(1.87, 2.72, 'Scatter', ha='center', fontsize=8, color='#555')

# Four GPU shards
shard_y = [3.4, 2.5, 1.6, 0.7]
shard_labels = ['GPU 0\n15K', 'GPU 1\n15K', 'GPU 2\n15K', 'GPU 3\n15K']
for i, (y, lbl) in enumerate(zip(shard_y, shard_labels)):
    box(ax, 2.1, y, 1.1, 0.7, C_DATA, lbl, fontsize=8)

# bracket
ax.annotate('', xy=(2.1, 0.65), xytext=(2.1, 4.15),
            arrowprops=dict(arrowstyle='-', color='#888', lw=1.5))

# ── Step 2: Forward + Backward on each GPU ─────────────────────────
arrow(ax, 3.2, 2.5, 3.6, 2.5, '')
ax.text(4.15, 4.55, 'Forward + Backward\n(cuBLAS GEMM, ReLU/Softmax)', ha='center', fontsize=9, color='#222')
ax.axhline(4.35, xmin=3.6/12, xmax=5.7/12, color='#aaa', lw=0.8, linestyle='--')

fwd_y = [3.4, 2.5, 1.6, 0.7]
for y in fwd_y:
    box(ax, 3.6, y, 1.55, 0.7, C_GPU, 'fwd+bwd\n∇W, ∇b', fontsize=8)

# ── Step 3: AllReduce ──────────────────────────────────────────────
arrow(ax, 5.15, 2.5, 5.6, 2.5, '')
ax.text(6.25, 4.55, 'AllReduce\n(ring / tree / MPI built-in)', ha='center', fontsize=9, color='#222')
ax.axhline(4.35, xmin=5.6/12, xmax=6.9/12, color='#aaa', lw=0.8, linestyle='--')

box(ax, 5.6, 1.0, 1.3, 3.0, C_COMM,
    'AllReduce\n∑ gradients\n÷ P', fontsize=9)

# ── Step 4: Weight Update ──────────────────────────────────────────
arrow(ax, 6.9, 2.5, 7.35, 2.5, '')
ax.text(8.1, 4.55, 'SGD Update\n(W -= lr · ∇W)', ha='center', fontsize=9, color='#222')
ax.axhline(4.35, xmin=7.35/12, xmax=9.5/12, color='#aaa', lw=0.8, linestyle='--')

upd_y = [3.4, 2.5, 1.6, 0.7]
for y in upd_y:
    box(ax, 7.35, y, 1.55, 0.7, C_UPD, 'update\nW, b', fontsize=8)

# ── Step labels at top ────────────────────────────────────────────
for x, lbl in [(0.9, '① Data\nPartition'),
               (3.87, '② Compute\nGradients'),
               (6.25, '③ Aggregate\nGradients'),
               (8.12, '④ Update\nWeights')]:
    ax.text(x, 4.75, lbl, ha='center', fontsize=8.5,
            color='#222', fontweight='bold')

ax.set_title('Data-Parallel MLP Training Pipeline  (P = 4 GPUs)',
             fontsize=13, pad=6, fontweight='bold')

plt.tight_layout()
plt.savefig('fig1_pipeline.pdf', bbox_inches='tight')
plt.savefig('fig1_pipeline.png', dpi=150, bbox_inches='tight')
print("Saved fig1_pipeline.pdf / .png")
plt.show()
