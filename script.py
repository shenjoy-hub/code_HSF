# hilbert_fragment_matrix.py
# ---------------------------------------------------------------
# Visualise fragmentation in a (D,S) sector of the weakly tilted
# Ising chain.  Requires: numpy, matplotlib, networkx

import itertools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from fragment import bitstring_to_tuple,krylov_fragment
# -------------- USER PARAMETERS --------------------------------
L         = 8     # chain length
D_target  = 3     # domain-wall number  (0 … L-1 for open chain)
S_target  = 4     # number of ↓ spins   (0 … L)
periodic  = False # True → periodic BCs
# ---------------------------------------------------------------


# ---------- helper functions -----------------------------------
def domain_wall_count(bits, periodic=False):
    D = sum(b1 != b2 for b1, b2 in zip(bits, bits[1:]))
    if periodic:
        D += bits[0] != bits[-1]          # extra bond for a ring
    return D

def spin_down_count(bits):
    return sum(bits)

def states_with_DS(L, D, S, periodic=False):
    """Generate all bitstrings with given (D,S)."""
    for bits in itertools.product((0, 1), repeat=L):
        if spin_down_count(bits) == S and domain_wall_count(bits, periodic) == D:
            yield bits

def projector_neighbors(state, periodic=False):
    """One-step projector-allowed swaps."""
    L = len(state)
    for j in range(L - 1):
        jp1 = j + 1
        jm1 = (j - 1) % L if periodic else j - 1
        jp2 = (j + 2) % L if periodic else j + 2
        if jm1 < 0 or jp2 >= L:
            if not periodic:              # projector ill-defined at edge
                continue
        # kinetic constraint
        if state[j] != state[jp1] and state[jm1] == state[jp2]:
            new_state = list(state)
            new_state[j], new_state[jp1] = new_state[jp1], new_state[j]
            yield tuple(new_state)

# -------- 1. enumerate states in the (D,S) sector --------------
states = sorted(states_with_DS(L, D_target, S_target, periodic))
N      = len(states)
orig2idx = {s: i for i, s in enumerate(states)}

# -------- 2. build graph using original ordering ---------------
G = nx.Graph()
G.add_nodes_from(range(N))
for i, s in enumerate(states):
    for t in projector_neighbors(s, periodic):
        j = orig2idx.get(t)
        if j is not None:
            G.add_edge(i, j)

# -------- 3. find connected components & permutation -----------
components = list(nx.connected_components(G))
# sort fragments: largest first; resolve ties by smallest old index
components.sort(key=lambda c: (-len(c), min(c)))

perm = []
for comp in components:
    perm.extend(sorted(comp))   # keep lexicographic order inside each fragment

newpos = {old: new for new, old in enumerate(perm)}

# -------- 4. build reordered adjacency matrix H ----------------
H = np.zeros((N, N), dtype=int)
np.fill_diagonal(H, 1)          # self-loops = 1
for i, j in G.edges():
    ni, nj = newpos[i], newpos[j]
    H[ni, nj] = H[nj, ni] = 1

# -------- 5. plot ----------------------------------------------
plt.figure(figsize=(6, 6))
plt.imshow(H, interpolation='nearest')
plt.title(f"H grouped by fragments\n(L={L}, D={D_target}, S={S_target}, N={N})")
plt.xlabel("state index (grouped)")
plt.ylabel("state index (grouped)")
plt.tight_layout()
plt.show()

# -------- 6. print index mapping -------------------------------
print("New index  ←  old index   bitstring")
for new_i, old_i in enumerate(perm):
    bits = ''.join(str(b) for b in states[old_i])
    print(f"{new_i:>3}     ←    {old_i:>3}      {bits}")

# -----------------------------------------------------------------
#  Enumerate ALL fragments and their sizes for a given L
# -----------------------------------------------------------------
def all_fragment_sizes(L, *, open_boundaries=True):
    """
    Return a dict {state_tuple: fragment_size} covering the entire Hilbert
    space of length L.  Uses a global BFS so each fragment is explored once.
    """
    from itertools import product

    unvisited = {tuple(bits) for bits in product((0, 1), repeat=L)}
    frag_size = {}

    while unvisited:
        root = unvisited.pop()           # pick a fresh basis state
        fragment = krylov_fragment(root,
                                   open_boundaries=open_boundaries)
        size = len(fragment)
        for s in fragment:
            frag_size[s] = size
        unvisited.difference_update(fragment)   # remove explored states
    return frag_size


if __name__ == "__main__":
    L = 8                   # chain length
    open_boundaries = True  # set False for periodic
    sizes = all_fragment_sizes(L, open_boundaries=open_boundaries)

    # -------- summary statistics ---------------------------------
    print(f"\nTotal Hilbert-space dimension: {2**L}")
    print(f"Unique fragments found       : {len(set(sizes.values()))}")
    print("Fragment-size histogram (size → count):")
    from collections import Counter
    hist = Counter(sizes.values())
    for sz, ct in sorted(hist.items()):
        print(f"  {sz:>5}  →  {ct:>6}")

    # OPTIONAL: examine a few sample states
    sample_states = ["01010101", "00011110", "11110000"]
    for s in sample_states:
        tup = bitstring_to_tuple(s)
        print(f"fragment size for {s}: {sizes[tup]}")

# -----------------------------------------------------------------
#  Enumerate ALL fragments and their sizes for a given L
# -----------------------------------------------------------------
def all_fragment_sizes(L, *, open_boundaries=True):
    """
    Return a dict {state_tuple: fragment_size} covering the entire Hilbert
    space of length L.  Uses a global BFS so each fragment is explored once.
    """
    from itertools import product

    unvisited = {tuple(bits) for bits in product((0, 1), repeat=L)}
    frag_size = {}

    while unvisited:
        root = unvisited.pop()           # pick a fresh basis state
        fragment = krylov_fragment(root,
                                   open_boundaries=open_boundaries)
        size = len(fragment)
        for s in fragment:
            frag_size[s] = size
        unvisited.difference_update(fragment)   # remove explored states
    return frag_size


if __name__ == "__main__":
    L = 8                   # chain length
    open_boundaries = True  # set False for periodic
    sizes = all_fragment_sizes(L, open_boundaries=open_boundaries)

    # -------- summary statistics ---------------------------------
    print(f"\nTotal Hilbert-space dimension: {2**L}")
    print(f"Unique fragments found       : {len(set(sizes.values()))}")
    print("Fragment-size histogram (size → count):")
    from collections import Counter
    hist = Counter(sizes.values())
    for sz, ct in sorted(hist.items()):
        print(f"  {sz:>5}  →  {ct:>6}")

    # OPTIONAL: examine a few sample states
    sample_states = ["01010101", "00011110", "11110000"]
    for s in sample_states:
        tup = bitstring_to_tuple(s)
        print(f"fragment size for {s}: {sizes[tup]}")

from itertools import product
from typing import Generator, Tuple
