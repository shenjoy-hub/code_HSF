from collections import deque, defaultdict, Counter
from itertools import product, islice
from typing import Generator, Set, Tuple, Dict, List, Union, Optional, Collection
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class FragmentedIsingChain:
    def __init__(self, L: int, open_boundaries: bool = True) -> None:
        """
        Initialize the Ising chain fragmentation analyzer.
        
        Args:
            L: Chain length
            open_boundaries: Use open boundaries (True) or periodic boundaries (False)
        """
        self.L = L
        self.open_boundaries = open_boundaries
        self._bit_cache: Dict[Tuple[int, ...], int] = {}
        
    @staticmethod
    def bitstring_to_tuple(s: str) -> Tuple[int, ...]:
        """Convert bitstring to tuple (0,1,0,1)"""
        return tuple(int(c) for c in s)

    @staticmethod
    def tuple_to_bitstring(t: Tuple[int, ...]) -> str:
        """Convert tuple to bitstring '0101'"""
        return ''.join(str(x) for x in t)

    def domain_wall_count(self, state: Tuple[int, ...]) -> int:
        """Count domain walls (D) for a state"""
        if self.open_boundaries:
            return sum(state[i] != state[i+1] for i in range(self.L-1))
        else:
            return sum(state[i] != state[(i+1) % self.L] for i in range(self.L))

    def spin_down_count(self, state: Tuple[int, ...]) -> int:
        """Count down spins (S) for a state"""
        return sum(state)

    def projector_neighbors(self, state: Tuple[int, ...]) -> Generator[Tuple[int, ...], None, None]:
        """
        Generate states reachable via one projected swap operation.
        
        Args:
            state: Current basis state
            
        Yields:
            Neighboring states satisfying the kinetic constraint
        """
        L = self.L
        for j in range(L - 1):
            jp1 = j + 1
            jm1 = j - 1
            jp2 = j + 2
            
            # Handle boundaries
            if self.open_boundaries:
                if jm1 < 0 or jp2 >= L:
                    continue
            else:
                jm1 = jm1 % L
                jp2 = jp2 % L

            # Kinetic constraint check
            if state[j] != state[jp1] and state[jm1] == state[jp2]:
                new_state = list(state)
                new_state[j], new_state[jp1] = new_state[jp1], new_state[j]
                yield tuple(new_state)

    def krylov_fragment(self, 
                        initial: Union[str, Tuple[int, ...]], 
                        max_size: Optional[int] = None) -> Set[Tuple[int, ...]]:
        """
        Find all basis states connected to an initial state (Krylov fragment).
        
        Args:
            initial: Starting basis state
            max_size: Safety cap for fragment size (optional)
            
        Returns:
            Set of connected basis states
        """
        state0 = self.bitstring_to_tuple(initial) if isinstance(initial, str) else initial
        visited: Set[Tuple[int, ...]] = {state0}
        queue: deque[Tuple[int, ...]] = deque([state0])

        while queue:
            state = queue.popleft()
            for neigh in self.projector_neighbors(state):
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append(neigh)
                    if max_size is not None and len(visited) >= max_size:
                        return visited
        return visited

    def states_with_DS(self, D_target: int, S_target: int) -> List[Tuple[int, ...]]:
        """
        Generate all states with given (D, S) sector.
        
        Args:
            D_target: Target domain wall count
            S_target: Target down-spin count
            
        Returns:
            List of states satisfying both conditions
        """
        states: List[Tuple[int, ...]] = []
        max_d = self.L if not self.open_boundaries else self.L - 1
        
        if not (0 <= D_target <= max_d):
            raise ValueError(f"Impossible D={D_target} for L={self.L} with open_boundaries={self.open_boundaries}")
        if not (0 <= S_target <= self.L):
            raise ValueError(f"S_target must be in [0, {self.L}], got {S_target}")

        for bits in product((0, 1), repeat=self.L):
            bits_tuple = tuple(bits)
            if (self.spin_down_count(bits_tuple) == S_target and 
                self.domain_wall_count(bits_tuple) == D_target):
                states.append(bits_tuple)
        return states

    def build_DS_sector_graph(self, D_target: int, S_target: int) -> nx.Graph:
        """
        Build connectivity graph for a (D, S) sector.
        
        Args:
            D_target: Domain wall count
            S_target: Down-spin count
            
        Returns:
            Connectivity graph of the sector
        """
        states = self.states_with_DS(D_target, S_target)
        G = nx.Graph()
        G.add_nodes_from(states)
        states_set = set(states)

        for s in states:
            for t in self.projector_neighbors(s):
                if t in states_set:
                    G.add_edge(s, t)
        return G

    def analyze_DS_sector(self, D_target: int, S_target: int) -> Dict[str, object]:
        """
        Analyze fragmentation in a (D, S) sector.
        
        Args:
            D_target: Domain wall count
            S_target: Down-spin count
            
        Returns:
            Dictionary containing:
                states: All states in sector
                graph: Connectivity graph
                components: Connected components
                sizes: Component sizes
                fragments: Number of fragments
        """
        states = self.states_with_DS(D_target, S_target)
        G = self.build_DS_sector_graph(D_target, S_target)
        components: List[Set[Tuple[int, ...]]] = list(nx.connected_components(G))
        sizes = sorted((len(c) for c in components), reverse=True)
        
        return {
            "states": states,
            "graph": G,
            "components": components,
            "sizes": sizes,
            "fragments": len(components)
        }

    def all_fragment_sizes(self) -> Dict[Tuple[int, ...], int]:
        """
        Compute fragment sizes for all states in Hilbert space.
        
        Returns:
            Mapping: state → fragment size
        """
        unvisited: Set[Tuple[int, ...]] = {bits for bits in product((0, 1), repeat=self.L)}
        frag_size: Dict[Tuple[int, ...], int] = {}

        while unvisited:
            root = unvisited.pop()
            fragment = self.krylov_fragment(root)
            size = len(fragment)
            for s in fragment:
                frag_size[s] = size
            unvisited.difference_update(fragment)
            
        return frag_size

    def plot_DS_sector_fragments(self, D_target: int, S_target: int) -> None:
        """
        Visualize fragmentation in a (D, S) sector.
        
        Args:
            D_target: Domain wall count
            S_target: Down-spin count
        """
        sector_data = self.analyze_DS_sector(D_target, S_target)
        states = sector_data["states"]
        G = nx.Graph(sector_data["graph"])  # Create a copy to avoid mutation
        components = sector_data["components"]
        
        # Create index mapping
        orig2idx: Dict[Tuple[int, ...], int] = {s: i for i, s in enumerate(states)}
        N = len(states)
        
        # Create permutation of states by component
        perm: List[Tuple[int, ...]] = []
        for comp in sorted(components, key=len, reverse=True):
            perm.extend(sorted(comp, key=lambda s: orig2idx[s]))
        newpos = {s: i for i, s in enumerate(perm)}
        
        # Build adjacency matrix
        H = np.zeros((N, N), dtype=int)
        np.fill_diagonal(H, 1)
        for i, j in G.edges():
            ni, nj = newpos[i], newpos[j]
            H[ni, nj] = H[nj, ni] = 1

        # Plot
        plt.figure(figsize=(8, 8))
        plt.imshow(H, interpolation='nearest')
        plt.title(f"Fragmentation (L={self.L}, D={D_target}, S={S_target}, N={N})")
        plt.xlabel("State index (grouped)")
        plt.ylabel("State index (grouped)")
        plt.tight_layout()
        plt.show()

    def print_fragment_info(self, 
                           initial: Union[str, Tuple[int, ...]], 
                           num: int = 10) -> None:
        """
        Print information about a Krylov fragment.
        
        Args:
            initial: Starting basis state
            num: Number of states to display
        """
        state0_str = initial if isinstance(initial, str) else self.tuple_to_bitstring(initial)
        frag = self.krylov_fragment(initial)
        print(f"System size L = {self.L}")
        print(f"Initial state : {state0_str}")
        print(f"Fragment size : {len(frag)} connected basis vectors")
        print(f"First {num} states in this fragment:")
        for s in islice(sorted(map(self.tuple_to_bitstring, frag)), num):
            print("  ", s)

    def global_fragmentation_report(self) -> None:
        """
        Generate fragmentation report for entire Hilbert space.
        """
        sizes = self.all_fragment_sizes()
        hist = Counter(sizes.values())
        total_dim = 2 ** self.L
        
        print(f"\nTotal Hilbert-space dimension: {total_dim}")
        print(f"Unique fragments found: {len(set(sizes.values()))}")
        print("Fragment-size histogram:")
        for sz, ct in sorted(hist.items()):
            print(f"  {sz:>5}  →  {ct:>6}")

# Example usage
if __name__ == "__main__":
    # Initialize system
    chain = FragmentedIsingChain(L=8, open_boundaries=True)
    
    # Analyze specific state
    chain.print_fragment_info("01010101")
    
    # Analyze (D, S) sector
    D_target, S_target = 3, 4
    sector_data = chain.analyze_DS_sector(D_target, S_target)
    print(f"\nD={D_target}, S={S_target} sector:")
    print(f"  States: {len(sector_data['states'])}")
    print(f"  Fragments: {sector_data['fragments']}")
    print(f"  Fragment sizes: {sector_data['sizes']}")
    
    # Visualize sector fragmentation
    chain.plot_DS_sector_fragments(D_target, S_target)
    
    # Global fragmentation report
    chain.global_fragmentation_report()