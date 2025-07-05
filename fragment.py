# -----------------------------------------------
#  Weakly tilted Ising chain: fragmentation probe
# -----------------------------------------------
from collections import deque
from itertools import islice

__all__ = ["krylov_fragment"]

def bitstring_to_tuple(s):
    """'0101' -> (0,1,0,1)."""
    return tuple(int(c) for c in s)

def tuple_to_bitstring(t):
    """(0,1,0,1) -> '0101'."""
    return ''.join(str(x) for x in t)

def projector_allowed_swaps(state, open_boundaries=True):
    """
    Yield all states reachable in one application of the
    projected hopping term.
    
    Parameters
    ----------
    state : tuple[int]
        Current basis state as 0/1 tuple.
    open_boundaries : bool
        If True, require j-1 and j+2 to lie inside the chain.
        If False, use periodic boundary conditions.
    """
    L = len(state)
    for j in range(L - 1):  # pair (j, j+1)
        jp1 = j + 1
        jm1 = j - 1
        jp2 = j + 2
        # Handle boundaries
        if open_boundaries:
            if jm1 < 0 or jp2 >= L:
                continue  # projector ill-defined at edge
        else:  # periodic
            jm1 %= L
            jp2 %= L
        # Kinetic-constraint test
        if state[j] != state[jp1] and state[jm1] == state[jp2]:
            new_state = list(state)
            new_state[j], new_state[jp1] = new_state[jp1], new_state[j]
            yield tuple(new_state)

def krylov_fragment(initial, *, open_boundaries=True, max_size=None):
    """
    Breadth-first search to find all basis states connected to `initial`.
    
    Parameters
    ----------
    initial : str | tuple[int]
        Initial basis vector (e.g. '010011').
    open_boundaries : bool
        Open vs periodic BCs (see above).
    max_size : int | None
        Safety cap for huge fragments.  If given, stop after this many.
    
    Returns
    -------
    fragment : set[tuple[int]]
        All dynamically connected basis states.
    """
    if isinstance(initial, str):
        initial = bitstring_to_tuple(initial)
    visited = {initial}
    queue = deque([initial])

    while queue:
        state = queue.popleft()
        for neigh in projector_allowed_swaps(state, open_boundaries):
            if neigh not in visited:
                visited.add(neigh)
                queue.append(neigh)
                if max_size is not None and len(visited) >= max_size:
                    return visited
    return visited

def print_krylov_fragment(init,num=10):
    """_summary_

    Args:
        init (_type_): _description_
        num (int, optional): _description_. Defaults to 10.
    """
    L = len(init)
    # init = "0101110010"          # pick your favourite basis state
    frag = krylov_fragment(init, open_boundaries=True)
    print(f"System size L = {L}")
    print(f"Initial state : {init}")
    print(f"Fragment size : {len(frag)} connected basis vectors")
    # show the first few states
    print(f"First {num} states in this fragment:")
    for s in islice(sorted(map(tuple_to_bitstring, frag)), 10):
        print("  ", s)
    
# ----------------------  EXAMPLE USAGE  ----------------------
if __name__ == "__main__":
    L = 10
    # init = "0101110010"          # pick your favourite basis state
    init = "0101010101"          # pick your favourite basis state
    frag = krylov_fragment(init, open_boundaries=True)
    print(f"System size L = {L}")
    print(f"Initial state : {init}")
    print(f"Fragment size : {len(frag)} connected basis vectors")
    # show the first few states
    print("First 10 states in this fragment:")
    for s in islice(sorted(map(tuple_to_bitstring, frag)), 10):
        print("  ", s)
