"""
Fig 2: Ring AllReduce vs Tree Reduce communication topology.
Left: ring (P=4 nodes in a circle with reduce-scatter + allgather arrows)
Right: binomial tree reduce + broadcast
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# ── helper ─────────────────────────────────────────────────────────
def draw_node(ax, x, y, label, color='#AED6F1', r=0.38):
    circ = plt.Circle((x, y), r, facecolor=color, edgecolor='#333', linewidth=1.5, zorder=3)
    ax.add_patch(circ)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

def curved_arrow(ax, x1, y1, x2, y2, color='#2c7bb6', rad=0.25, lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    connectionstyle=f'arc3,rad={rad}'))

# ══════════════════════════════════════════════════════════════════
# LEFT: Ring AllReduce (P=4)
# ══════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(-2, 2); ax.set_ylim(-2.3, 2.6); ax.axis('off')
ax.set_title('Ring AllReduce  (P = 4)', fontsize=13, fontweight='bold', pad=8)

# node positions (square ring)
positions = [(0, 1.5), (1.5, 0), (0, -1.5), (-1.5, 0)]
node_colors = ['#AED6F1', '#A9DFBF', '#F9E79F', '#F1948A']
labels = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3']

for (x, y), c, lbl in zip(positions, node_colors, labels):
    draw_node(ax, x, y, lbl, color=c, r=0.42)

# Phase 1: reduce-scatter arrows (clockwise, blue)
for i in range(4):
    x1, y1 = positions[i]
    x2, y2 = positions[(i+1) % 4]
    curved_arrow(ax, x1*0.6, y1*0.6, x2*0.6, y2*0.6,
                 color='#2166ac', rad=0.15, lw=2.0)

# Phase 2: allgather arrows (counter-clockwise, green)
for i in range(4):
    x1, y1 = positions[i]
    x2, y2 = positions[(i-1) % 4]
    curved_arrow(ax, x1*0.55, y1*0.55, x2*0.55, y2*0.55,
                 color='#1a9641', rad=-0.15, lw=2.0)

# legend
p1 = mpatches.Patch(color='#2166ac', label='Phase 1: Reduce-Scatter (P−1 steps)')
p2 = mpatches.Patch(color='#1a9641', label='Phase 2: All-Gather (P−1 steps)')
ax.legend(handles=[p1, p2], loc='lower center', fontsize=9,
          bbox_to_anchor=(0.5, -0.08))

ax.text(0, -2.2,
        'Each rank sends/recvs N/P floats per step\n'
        'Total: 2(P−1)/P · N floats sent per rank',
        ha='center', fontsize=9, color='#444',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', alpha=0.8))

# ══════════════════════════════════════════════════════════════════
# RIGHT: Binomial Tree Reduce + Bcast (P=4)
# ══════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.8, 4.0); ax.axis('off')
ax.set_title('Tree Reduce + Broadcast  (P = 4)', fontsize=13, fontweight='bold', pad=8)

# Tree layout (logical ranks 0-3 mapped to root=0):
#   level 0 (root): rank 0 at top
#   level 1: rank 0 left child, rank 2 right
#   level 2: rank 0, rank 1, rank 2, rank 3 at bottom
tree_pos = {0: (1.5, 3.2), 1: (0.5, 1.6), 2: (2.5, 1.6), 3: (0.5, 0.0), 4: (1.5, 0.0),
            5: (2.5, 0.0), 6: (3.5, 0.0)}

# Actual P=4 binomial tree positions
nodes = {
    'rank 0': (1.5, 3.0),
    'rank 2': (1.5, 1.5),
    'rank 1': (0.3, 1.5),
    'rank 3': (2.7, 1.5),
}
leaf_positions = {'rank 0': (1.5, 0.1), 'rank 1': (0.3, 0.1),
                  'rank 2': (1.5, 0.1), 'rank 3': (2.7, 0.1)}

# Draw leaf nodes
node_color_map = {'rank 0':'#AED6F1','rank 1':'#A9DFBF','rank 2':'#F9E79F','rank 3':'#F1948A'}
for lbl, (x, y) in nodes.items():
    draw_node(ax, x, y, lbl, color=node_color_map[lbl], r=0.38)

# Phase 1: Reduce arrows (bottom → root, red)
# Step d=0: rank 1 → rank 0, rank 3 → rank 2
reduce_edges = [
    (nodes['rank 1'], nodes['rank 0'], 'step 1'),
    (nodes['rank 3'], nodes['rank 2'], 'step 1'),
    (nodes['rank 2'], nodes['rank 0'], 'step 2'),
]
reduce_color = '#d73027'
bcast_color  = '#1a9641'

for (x1,y1),(x2,y2),_ in reduce_edges[:2]:
    ax.annotate('', xy=(x2, y2+0.4), xytext=(x1, y1+0.4),
                arrowprops=dict(arrowstyle='->', color=reduce_color, lw=2.0,
                                connectionstyle='arc3,rad=0.0'))
# step 2: rank 2 → rank 0
x1,y1 = nodes['rank 2']; x2,y2 = nodes['rank 0']
ax.annotate('', xy=(x2+0.1, y2-0.4), xytext=(x1+0.1, y1+0.4),
            arrowprops=dict(arrowstyle='->', color=reduce_color, lw=2.0,
                            connectionstyle='arc3,rad=-0.15'))

# Phase 2: Bcast arrows (root → all, green)
x1,y1 = nodes['rank 0']; x2,y2 = nodes['rank 2']
ax.annotate('', xy=(x2-0.1, y2+0.4), xytext=(x1-0.1, y1-0.4),
            arrowprops=dict(arrowstyle='->', color=bcast_color, lw=2.0,
                            connectionstyle='arc3,rad=-0.15'))
for (x2,y2) in [nodes['rank 1'], nodes['rank 3']]:
    x1,y1 = nodes['rank 0'] if (x2 < 1.5) else nodes['rank 2']
    if x2 < 1.5: x1, y1 = nodes['rank 0']
    else: x1, y1 = nodes['rank 2']
    ax.annotate('', xy=(x2, y2+0.4), xytext=(x1, y1-0.4) if x2 < 1.5 else (x1, y1+0.4),
                arrowprops=dict(arrowstyle='->', color=bcast_color, lw=2.0,
                                connectionstyle='arc3,rad=0.0'))

# Step annotations
ax.text(1.9, 2.25, 'step 2', fontsize=8, color=reduce_color)
ax.text(-0.1, 1.95, 'step 1', fontsize=8, color=reduce_color)
ax.text(2.85, 1.95, 'step 1', fontsize=8, color=reduce_color)

# legend
p1 = mpatches.Patch(color=reduce_color, label='Phase 1: Tree Reduce (⌈log₂P⌉ steps)')
p2 = mpatches.Patch(color=bcast_color,  label='Phase 2: MPI_Bcast (⌈log₂P⌉ steps)')
ax.legend(handles=[p1, p2], loc='lower center', fontsize=9,
          bbox_to_anchor=(0.5, -0.08))

ax.text(1.5, -0.65,
        'Each step sends full N floats\n'
        'Total: 2⌈log₂P⌉ · N floats sent per root path',
        ha='center', fontsize=9, color='#444',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', alpha=0.8))

fig.suptitle('AllReduce Communication Patterns', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig2_topology.pdf', bbox_inches='tight')
plt.savefig('fig2_topology.png', dpi=150, bbox_inches='tight')
print("Saved fig2_topology.pdf / .png")
plt.show()
