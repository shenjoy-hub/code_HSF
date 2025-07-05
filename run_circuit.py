import numpy as np
import qutip as qt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator
import matplotlib.pyplot as plt
from tqdm import tqdm

class TiltedIsingEvolution:
    def __init__(self, num_qubits, h_z, J, h_y, epsilon=0.01):
        """
        Initialize the weakly tilted Ising chain evolution simulator
        
        Args:
            num_qubits: Number of spins in the chain
            h_z: Strength of longitudinal field (sigma_z term)
            J: Coupling strength between nearest neighbors
            h_y: Strength of transverse field (sigma_y term)
            epsilon: Small parameter for spin-flip precision
        """
        self.num_qubits = num_qubits
        self.h_z = h_z
        self.J = J
        self.h_y = h_y
        self.epsilon = epsilon
        self.T = 2.0  # Total period time
        
        # Create basis states
        self.basis_0 = qt.basis(2, 0)
        self.basis_1 = qt.basis(2, 1)
        
        # Build the static Hamiltonian H1
        self.H1 = self._build_H1()
        
    def _build_H1(self):
        """Construct the static Hamiltonian H1"""
        H_terms = []
        
        # Add sigma_z terms (longitudinal field)
        for i in range(self.num_qubits):
            op_list = [qt.qeye(2)] * self.num_qubits
            op_list[i] = qt.sigmaz()
            H_terms.append(self.h_z * qt.tensor(op_list))
        
        # Add sigma^z_i sigma^z_{i+1} terms (Ising coupling)
        for i in range(self.num_qubits - 1):
            op_list = [qt.qeye(2)] * self.num_qubits
            op_list[i] = qt.sigmaz()
            op_list[i+1] = qt.sigmaz()
            H_terms.append(self.J * qt.tensor(op_list))
        
        # Add sigma_y terms (transverse field)
        for i in range(self.num_qubits):
            op_list = [qt.qeye(2)] * self.num_qubits
            op_list[i] = qt.sigmay()
            H_terms.append(self.h_y * qt.tensor(op_list))
        
        return sum(H_terms)
    
    def H_flip(self):
        """Construct the spin-flip Hamiltonian"""
        H_terms = []
        
        # Add sigma_x terms for spin-flip
        for i in range(self.num_qubits):
            op_list = [qt.qeye(2)] * self.num_qubits
            op_list[i] = qt.sigmax()
            H_terms.append(qt.tensor(op_list))
        
        coefficient = (2 / self.T) * (np.pi / 2 - self.epsilon)
        return coefficient * sum(H_terms)

    def exact_time_evolution(self, initial_state, num_cycles=1):
        """
        Compute exact time evolution of magnetizations using QuTiP
        
        Args:
            initial_state(qt.Qobj,str): Initial quantum state(e.g., "1010")
            num_cycles: Number of evolution cycles
            
        Returns:
            Final state after evolution
        """
        
        # Build full Hamiltonian operators
        H1 = self.H1
        H_flip = self.H_flip()
        
        total_time = num_cycles * self.T
        time_points = np.arange(0,total_time,self.T)
        # Convert string input to quantum state
        if isinstance(initial_state, str):
            # Check length matches number of qubits
            if len(initial_state) != self.num_qubits:
                raise ValueError(f"Bitstring length ({len(initial_state)}) doesn't match"
                                f" number of qubits ({self.num_qubits})")
            
            # Create computational basis state
            basis_states = []
            for bit in initial_state:
                if bit == '0':
                    basis_states.append(qt.basis(2, 0))
                elif bit == '1':
                    basis_states.append(qt.basis(2, 1))
                else:
                    raise ValueError(f"Invalid character '{bit}' in bitstring - must be 0 or 1")
            
            initial_state = qt.tensor(basis_states)

        def static_time_function(t, args):
            phase = t % self.T
            if phase < self.T / 2:
                return 1.0  # static hamiltonian during first half
            else:
                return 0.0
        def flip_time_function(t, args):
            phase = t % self.T
            if phase < self.T / 2:
                return 0.0 
            else:
                return 1.0  # flip hamiltonian during second half

        # Solve time-dependent evolution
        floquent_H = [[H1,static_time_function],[H_flip,flip_time_function]]
        result = qt.sesolve(floquent_H, initial_state, time_points)
        return result.states
    
    def plot_heatmap(self, initial_state, num_cycles=50, cmap='bwr'):
        """
        Plot magnetization heatmap in exact simulation
        
        Args:
            initial_state: Initial quantum state as qt.Qobj or bitstring (e.g., "1010")
            num_cycles: Number of evolution cycles
            cmap: Color map (default: blue-white-red)
        """
        # Get states data
        data = self.exact_time_evolution(initial_state, num_cycles)

        # expection operator
        e_ops = []
        for i in range(self.num_qubits):
            op_list = [qt.qeye(2)]*self.num_qubits
            op_list[i] = qt.sigmaz()
            e_ops.append(qt.tensor(op_list))
        mag_data = qt.expect(e_ops,data)

        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create the heatmap
        im = plt.imshow(mag_data, 
                    cmap=cmap,
                    aspect='auto',
                    origin='lower',
                    vmin=-1, 
                    vmax=1,
                    extent=[0, num_cycles, 0, self.num_qubits])
        
        # Add labels and title
        plt.xlabel("Period", fontsize=12)
        plt.ylabel("Site Index", fontsize=12)
        plt.title("Magnetization Evolution in Tilted Ising Chain", fontsize=14)
        
        # Add colorbar with label
        cbar = plt.colorbar(im)
        cbar.set_label(r'$\langle \sigma_z \rangle$', fontsize=12)
        
        # Set ticks
        plt.xticks(np.arange(0, num_cycles+1, 5))
        plt.yticks(np.arange(0, self.num_qubits+1, 1))
        
        # Add grid for better visibility of boundaries
        plt.grid(False)  # No grid lines within cells
        for pos in np.arange(0.5, self.num_qubits, 1):
            plt.axhline(y=pos, color='lightgray', linestyle='-', alpha=0.3)
        for pos in np.arange(0.5, num_cycles, 1):
            plt.axvline(x=pos, color='lightgray', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def build_trotter_step_circuit(self):
        """Build a single Trotter step circuit"""
        qc = QuantumCircuit(self.num_qubits)
        
        # First part: e^{-i H1 (T/2)} for half period
        qc.compose(self.build_H1_circuit(self.T / 2), inplace=True)
        
        # Second part: Spin-flip operators for half period
        qc.compose(self.build_flip_circuit(),inplace=True)
        
        return qc
    
    def build_flip_circuit(self):
        """Build spin flip evolution for half period (τ = T/2)"""
        qc = QuantumCircuit(self.num_qubits)
        
        # Compute effective rotation angle
        flip_angle = np.pi - 2 * self.epsilon
        for i in range(self.num_qubits):
            qc.rx(flip_angle, i)
            
        return qc

    def build_H1_circuit(self, tau):
        """
        Build symmetric Trotter decomposition for H₁ evolution for time τ
        following the scheme: e^{-i H_1 τ} ≈ e^{-i A τ/4} e^{-i B τ/2} e^{-i A τ/2} e^{-i B τ/2} e^{-i A τ/4}
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Time scaling factors
        t1 = tau / 4  # τ/4
        t2 = tau / 2  # τ/2
        
        # Apply symmetric decomposition
        qc.compose(self.build_A_circuit(t1), inplace=True)  # e^{-i A τ/4}
        qc.compose(self.build_B_circuit(t2), inplace=True)  # e^{-i B τ/2}
        qc.compose(self.build_A_circuit(t2), inplace=True)  # e^{-i A τ/2}
        qc.compose(self.build_B_circuit(t2), inplace=True)  # e^{-i B τ/2}
        qc.compose(self.build_A_circuit(t1), inplace=True)  # e^{-i A τ/4}
        
        return qc
    
    def build_A_circuit(self, time):
        """
        Build circuit for exp(-i A time) where
        A = ∑_i (h_y sigma_i^y)
        """
        qc = QuantumCircuit(self.num_qubits)

        # Apply rotation: Ry
        for i in range(self.num_qubits):
            qc.ry(2*self.h_y, i)
        qc.barrier()
        return qc
    
    def build_B_circuit(self, time):
        """
        Build circuit for exp(-i B time) where
        B = ∑_i J sigma_i^z sigma_{i+1}^z + h_z sigma_i^z
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Apply nearest-neighbor ZZ interactions
        for i in range(0,self.num_qubits - 1,2):
            qc.rzz(4 * self.J * time, i, i+1)  # Rzz(φ) = exp(-i φ/2 Z⊗Z)
                        
        for i in range(1,self.num_qubits - 1,2):
            qc.rzz(4 * self.J * time, i, i+1)  # Rzz(φ) = exp(-i φ/2 Z⊗Z)
                        
        # Apply rotation: Rz
        for i in range(self.num_qubits):
            qc.ry(2*self.h_z, i)
        qc.barrier()    
        return qc
    
    def build_full_circuit(self, initial_state, num_cycles=1, is_measure=True):
        """
        Build quantum circuit for full evolution
        
        Args:
            initial_state: Bitstring for initial state (e.g., '0101')
            num_cycles: Number of evolution cycles
            
        Returns:
            QuantumCircuit: Full circuit with state prep and evolution
        """
        # Create quantum register and circuit
        qr = QuantumRegister(self.num_qubits, 'q')
        cr = ClassicalRegister(self.num_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Initialize state from bitstring
        if initial_state:
            if len(initial_state) != self.num_qubits:
                raise ValueError(f"Initial state length must be {self.num_qubits}")
                
            for i, bit in enumerate(initial_state):
                if bit == '1':
                    qc.x(qr[i])
            qc.barrier()
        
        # build circuit in each period
        trotter_step_circ = self.build_trotter_step_circuit()
        flip_circ = self.build_flip_circuit()
        for _ in range(num_cycles):
            qc.compose(trotter_step_circ, qubits=qr, inplace=True)
            # qc.compose(flip_circ,inplace=True)

        # Add measurements for simulation
        qc.barrier()
        if is_measure:
            qc.measure(qr, cr)
        
        return qc

    def simulate_statevector_evolution(self, initial_state, num_cycles=50):
        """
        Simulate evolution using qiskit_aer.StatevectorSimulator
        
        Args:
            initial_state: Initial state as bitstring
            num_cycles: Number of evolution cycles
            
        Returns:
            List of Statevector objects at each cycle
        """

        simulator = StatevectorSimulator()
        statevectors = []
        for i in tqdm(range(num_cycles)):
            qc = self.build_full_circuit(initial_state,i,is_measure=False)
            job = simulator.run(qc)
            state = job.result().get_statevector()
            statevectors.append(state)
        return statevectors
    
    def calculate_magnetizations(self, statevectors):
        """
        Calculate ⟨sigma_z⟩ for each site over time
        
        Args:
            statevectors: List of statevector at each time point
            
        Returns:
            2D array: sites * time
        """
        num_steps = len(statevectors)
        mag_data = np.zeros((self.num_qubits, num_steps))
        
        # For each site, calculate magnetization at each time step
        for site in range(self.num_qubits):
            for t, state in enumerate(statevectors):
                # Calculate <sigma_z> for this site at this time
                sv = state.data
                # Direct computation for sigma_z expectation value
                plus_amp = 0.0
                minus_amp = 0.0
                
                # For each basis state
                for idx, amplitude in enumerate(sv):
                    # Get binary representation for basis state
                    bin_str = bin(idx)[2:].zfill(self.num_qubits)
                    # Check this site's spin
                    if bin_str[site] == '0':
                        plus_amp += abs(amplitude)**2
                    else:
                        minus_amp += abs(amplitude)**2
                
                mag_data[site, t] = plus_amp - minus_amp
                
        return mag_data
    
    def create_heatmap(self, mag_data, num_cycles):
        """
        Create magnetization heatmap matching reference figure
        
        Args:
            mag_data: 2D array of magnetizations
            num_cycles: Number of evolution cycles
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        im = plt.imshow(mag_data,
                       cmap='RdBu',
                       aspect='auto',
                       origin='lower',
                       vmin=-1.0,
                       vmax=1.0,
                       extent=[0, num_cycles, 0, self.num_qubits])
        
        # Labels and title
        plt.xlabel("Period", fontsize=12)
        plt.ylabel("Site Index", fontsize=12)
        plt.title("Magnetization Evolution in Tilted Ising Chain", fontsize=14)
        
        # Colorbar with correct labels and positions
        cbar = plt.colorbar(im)
        cbar.set_label(r'$\langle \sigma_z \rangle$', fontsize=12)
        
        # Set ticks and grid
        plt.xticks(np.arange(0, num_cycles + 1, 5))
        plt.yticks(np.arange(0, self.num_qubits + 1, 1))
        
        # Grid lines between cells
        for pos in np.arange(0.5, self.num_qubits, 1):
            plt.axhline(y=pos, color='lightgray', linestyle='-', alpha=0.3)
        for pos in np.arange(0.5, num_cycles, 1):
            plt.axvline(x=pos, color='lightgray', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('magnetization_heatmap.png', dpi=120)
        plt.show()

    def plot_magnetization(self, initial_state, num_cycles=50):
        """
        Simulate evolution and plot magnetization heatmap
        
        Args:
            initial_state: Initial state as bitstring
            num_cycles: Number of evolution cycles
        """
        
        # Simulate with statevector backend to get full state evolution
        statevectors = self.simulate_statevector_evolution(initial_state, num_cycles)
        
        # Calculate magnetizations
        mag_data = self.calculate_magnetizations(statevectors)
        
        # Create heatmap similar to reference image
        self.create_heatmap(mag_data, num_cycles)
