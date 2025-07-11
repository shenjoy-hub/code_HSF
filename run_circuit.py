import numpy as np
import qutip as qt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister,transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator,AerSimulator
import matplotlib.pyplot as plt
from tqdm import tqdm

class TiltedIsingChain:
    """Tilted Ising Chain exact simulation using qutip
    """
    def __init__(self, num_qubits, h_z, J, h_y, epsilon=0.01):
        """
        Initialize the weakly tilted Ising chain

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
        # from utils import smooth_square_wave
        # # Create time-dependent Hamiltonian
        # static_time_function = lambda t, args: smooth_square_wave(t, 0, 1, self.T, phase=0.5)
        # flip_time_function = lambda t, args: smooth_square_wave(t, 0, 1, self.T, phase=0.0)
        
        floquent_H = [[H1,static_time_function],[H_flip,flip_time_function]]
        result = qt.mesolve(floquent_H, initial_state, time_points,options={"progress_bar":"tqdm"})
        return result.states
    
    def calculate_megnetizations(self, initial_state, num_cycles=50):
        """
        Calculate magnetization versus time in exact simulation
        
        Args:
            initial_state: Initial quantum state as qt.Qobj or bitstring (e.g., "1010")
            num_cycles: Number of evolution cycles
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
        return mag_data

    def plot_heatmap(self, initial_state, num_cycles=50, cmap='bwr'):
        """
        Plot magnetization heatmap in exact simulation
        
        Args:
            initial_state: Initial quantum state as qt.Qobj or bitstring (e.g., "1010")
            num_cycles: Number of evolution cycles
            cmap: Color map (default: blue-white-red)
        """
        
        mag_data = self.calculate_megnetizations(initial_state,num_cycles)
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

    def correlation_one_site(self, initial_state, site_index, num_cycles=50):
        """
        Calculate one-site correlation function in exact simulation

        Args:
            initial_state: Initial quantum state as qt.Qobj or bitstring (e.g., "1010")
            num_cycles: Number of evolution cycles
        """
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

        assert 0 <= site_index < self.num_qubits, "Site index out of bounds"
        op_list = [qt.qeye(2)] * self.num_qubits
        op_list[site_index] = qt.sigmaz()
        ops = qt.tensor(op_list)

        # Build full Hamiltonian operators
        H1 = self.H1
        H_flip = self.H_flip()
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
        floquent_H = [[H1,static_time_function],[H_flip,flip_time_function]]
        # calculate correlation function
        t_list = np.arange(0, num_cycles * self.T, self.T)
        # result = qt.correlation_2op_1t(floquent_H, initial_state, t_list, [], ops, ops,options={"progress_bar":"tqdm"})
        result = qt.correlation_2op_1t(floquent_H, initial_state, t_list, [], ops, ops)
        return result

    def hamming_distance(self, initial_state, num_cycles=50):
        """
        Calculate Hamming distance between in exact simulation
        Args:
            initial_state: Initial quantum state as qt.Qobj or bitstring (e.g., "1010")
            num_cycles: Number of evolution cycles
        """
        distance = 0.0
        for i in tqdm(range(2,self.num_qubits-2)):
            one_site_corr = self.correlation_one_site(initial_state, i, num_cycles)
            distance += one_site_corr
        distance = (1- distance / (self.num_qubits - 4)) / 2.0
        return distance

    def expectation_z(self, initial_state, num_cycles=50):
        """
        Calculate expectation value of sigma_z for the system in exact simulation
        Args:
            initial_state: Initial quantum state as qt.Qobj or bitstring (e.g., "1010")
            num_cycles: Number of evolution cycles
        """
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

        ops = []
        for i in range(self.num_qubits):
            op_list = [qt.qeye(2)] * self.num_qubits
            op_list[i] = qt.sigmaz()
            ops.append(qt.tensor(op_list))
        
        # Build full Hamiltonian operators
        H1 = self.H1
        H_flip = self.H_flip()
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
        floquent_H = [[H1,static_time_function],[H_flip,flip_time_function]]
        # calculate correlation function
        t_list = np.arange(0, num_cycles * self.T, self.T)
        result = qt.mesolve(floquent_H, initial_state, t_list, [], ops, options={"progress_bar":"tqdm"})
        return np.array(result.expect)

    def life_time(self, initial_state, num_cycles=50):
        """
        Calculate life time of the system in exact simulation
        Args:
            initial_state: Initial quantum state as qt.Qobj or bitstring (e.g., "1010")
            num_cycles: Number of evolution cycles
        """
        expectation_z = self.expectation_z(initial_state, num_cycles)
        sign_z_0 = np.sign(expectation_z[:,0])
        lambda_t = np.zeros(num_cycles)
        for i in range(num_cycles):
            lambda_t[i] = np.sum(sign_z_0 * expectation_z[:,i]) / self.num_qubits
        return lambda_t

class IsingChainSimulation:
    """circuit simulation of Ising Chain using qiskit
    """
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

    def build_trotter_step_circuit(self, is_barrier=True):
        """Build a single Trotter step circuit"""
        qc = QuantumCircuit(self.num_qubits)
        
        # First part: e^{-i H1 (T/2)} for half period
        qc.compose(self.build_H1_circuit(self.T / 2, is_barrier=is_barrier), inplace=True)
        
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

    def build_H1_circuit(self, tau, is_barrier=True):
        """
        Build symmetric Trotter decomposition for H₁ evolution for time τ
        following the scheme: e^{-i H_1 τ} ≈ e^{-i A τ/4} e^{-i B τ/2} e^{-i A τ/2} e^{-i B τ/2} e^{-i A τ/4}
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Time scaling factors
        t1 = tau / 4  # τ/4
        t2 = tau / 2  # τ/2
        
        # Apply symmetric decomposition
        qc.compose(self.build_A_circuit(t1, is_barrier), inplace=True)  # e
        qc.compose(self.build_B_circuit(t2, is_barrier), inplace=True)  # e
        qc.compose(self.build_A_circuit(t2, is_barrier), inplace=True)  # e
        qc.compose(self.build_B_circuit(t2, is_barrier), inplace=True)  # e
        qc.compose(self.build_A_circuit(t1, is_barrier), inplace=True)  # e^{-i A τ/4}

        return qc
    
    def build_A_circuit(self, time,is_barrier=True):
        """
        Build circuit for exp(-i A time) where
        A = ∑_i (h_y sigma_i^y)
        """
        qc = QuantumCircuit(self.num_qubits)

        # Apply rotation: Ry
        for i in range(self.num_qubits):
            qc.ry(2*self.h_y*time, i)
        if is_barrier:
            qc.barrier()
        return qc
    
    def build_B_circuit(self, time,is_barrier=True):
        """
        Build circuit for exp(-i B time) where
        B = ∑_i J sigma_i^z sigma_{i+1}^z + h_z sigma_i^z
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Apply nearest-neighbor ZZ interactions
        for i in range(0,self.num_qubits - 1,2):
            qc.rzz(2 * self.J * time, i, i+1)  # Rzz(φ) = exp(-i φ/2 Z⊗Z)
                        
        for i in range(1,self.num_qubits - 1,2):
            qc.rzz(2 * self.J * time, i, i+1)  # Rzz(φ) = exp(-i φ/2 Z⊗Z)
                        
        # Apply rotation: Rz
        for i in range(self.num_qubits):
            qc.rz(2*self.h_z*time, i)
        if is_barrier:
            qc.barrier()    
        return qc

    def build_full_circuit(self, initial_state, num_cycles=1, is_barrier=True, is_measure=True):
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
        trotter_step_circ = self.build_trotter_step_circuit(is_barrier)
        for _ in range(num_cycles):
            qc.compose(trotter_step_circ, qubits=qr, inplace=True)

        # Add measurements for simulation
        qc.barrier()
        if is_measure:
            qc.measure(qr, cr)
        
        return qc

    def build_echo_circuit(self, initial_state, num_cycles=1, is_barrier=True, is_measure=True):
        """
        Build echo circuit for time-reversal symmetry

        Args:
            initial_state: Bitstring for initial state (e.g., '0101')
            num_cycles: Number of evolution cycles
            is_barrier: Whether to include barriers
            is_measure: Whether to include measurements

        Returns:
            QuantumCircuit: Echo circuit
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
        trotter_step_circ = self.build_trotter_step_circuit(is_barrier)
        for _ in range(num_cycles):
            qc.compose(trotter_step_circ, qubits=qr, inplace=True)
        # Add echo operation: reverse the circuit
        for _ in range(num_cycles):
            qc.compose(trotter_step_circ.inverse(), qubits=qr, inplace=True)
        # qc.compose(qc.inverse(), qubits=qr, inplace=True)  # Reverse the entire circuit
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
        mag_data = np.flipud(mag_data)  # Reverses rows so site0 becomes top row
        return mag_data
    
    def create_heatmap(self, mag_data, num_cycles,cmap='bwr'):
        """
        Create magnetization heatmap matching reference figure
        
        Args:
            mag_data: 2D array of magnetizations
            num_cycles: Number of evolution cycles
        """
        plt.figure(figsize=(10, 6))
        
        # Create heatmap
        im = plt.imshow(mag_data,
                       cmap=cmap,
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
        # plt.savefig('magnetization_heatmap.png', dpi=120)
        plt.show()

    def plot_heatmap(self, initial_state, num_cycles=50):
        """
        Simulate evolution without noise and plot magnetization heatmap
        
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

    def plot_heatmap_with_noise(self, initial_state, noise_model=None, num_cycles=50, shots=5000):
        """
        Simulate evolution with noise and plot magnetization heatmap
        
        Args:
            initial_state: Initial state as bitstring
            noise_model: Noise model for simulation, None uses default
            num_cycles: Number of evolution cycles
            shots: Number of shots for each cycle
            
        """
        if noise_model is None:
            from qiskit_noise_model import get_noise_model
            noise_model = get_noise_model()
        # Simulate with noise model
        magnetization_time_2d = self.simulate_with_noise(initial_state, noise_model, num_cycles, shots)
        
        # Create heatmap similar to reference image
        self.create_heatmap(magnetization_time_2d, num_cycles)

    def plot_heatmap_normalized(self, initial_state, noise_model=None, num_cycles=50, shots=5000):
        """
        Simulate evolution with noise and plot normalized magnetization heatmap.
        This normalizes the magnetization by the echo circuit evolution
        Args:
            initial_state: Initial state as bitstring
            noise_model: Noise model for simulation, None uses default
            num_cycles: Number of evolution cycles
            shots: Number of shots for each cycle
            
        """
        if noise_model is None:
            from qiskit_noise_model import get_noise_model
            noise_model = get_noise_model()
        
        # Simulate echo circuit with noise model
        magnetization_time_noise = self.simulate_with_noise(initial_state, noise_model, num_cycles, shots)
        magnetization_time_echo = self.simulate_echo_circuit(initial_state, noise_model, num_cycles, shots)
        magnetization_time_normalized = magnetization_time_noise/np.sqrt(np.abs(magnetization_time_echo))

        # Create heatmap similar to reference image
        self.create_heatmap(magnetization_time_normalized, num_cycles)

    def simulate_with_noise(self, initial_state, noise_model=None, num_cycles=50, shots=5000):
        if noise_model is None:
            from qiskit_noise_model import get_noise_model
            noise_model = get_noise_model()
        simulator = AerSimulator(noise_model=noise_model)
        magnetization_time_2d = np.zeros((self.num_qubits, num_cycles))
        for i in tqdm(range(num_cycles)):
            qc = self.build_full_circuit(initial_state, i, is_barrier=False, is_measure=True)
            qc_transpiled = transpile(qc, basis_gates=['u3', 'cp'], optimization_level=3)
            job = simulator.run(qc_transpiled,shots=shots)
            result = job.result()
            counts = result.get_counts()
            expectations = [0.0] * self.num_qubits
            for bitstring, freq in counts.items():
                for q in range(self.num_qubits):
                    # each qubit expection: ⟨Z⟩ = P(0) - P(1)
                    bit = int(bitstring[self.num_qubits - 1 - q])  # reverse order
                    expectations[q] += ((-1)**bit) * (freq / shots)
            magnetization_time_2d[:, i] = expectations
        
        return magnetization_time_2d

    def simulate_echo_circuit(self, initial_state, noise_model=None, num_cycles=50, shots=5000):
        """
        Simulate echo circuit evolution with noise
        
        Args:
            initial_state: Initial state as bitstring
            noise_model: Noise model for simulation
            num_cycles: Number of evolution cycles
            shots: Number of shots for each cycle
            
        Returns:
            2D array of magnetizations over time
        """
        if noise_model is None:
            from qiskit_noise_model import get_noise_model
            noise_model = get_noise_model()
        
        simulator = AerSimulator(noise_model=noise_model)
        magnetization_time_2d = np.zeros((self.num_qubits, num_cycles))
        
        for i in tqdm(range(num_cycles)):
            qc = self.build_echo_circuit(initial_state, i, is_barrier=False, is_measure=True)
            qc_transpiled = transpile(qc, basis_gates=['u3', 'cp'], optimization_level=3)
            job = simulator.run(qc_transpiled, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
            expectations = [0.0] * self.num_qubits
            for bitstring, freq in counts.items():
                for q in range(self.num_qubits):
                    # each qubit expectation: ⟨Z⟩ = P(0) - P(1)
                    bit = int(bitstring[self.num_qubits - 1 - q])
                    expectations[q] += ((-1)**bit) * (freq / shots)
            magnetization_time_2d[:, i] = expectations

        return magnetization_time_2d

    def plot_heatmap_echo(self, initial_state, num_cycles=50):
        """
        Plot magnetization heatmap
        """
        magnetization_time_2d = self.simulate_echo_circuit(initial_state, num_cycles=num_cycles)
        self.create_heatmap(magnetization_time_2d, num_cycles)

def plot_heatmap_theory(num_qubits, h_z, J, h_y, epsilon, initial_state, num_cycles=50):
    """
    Plot magnetization heatmap for theoretical simulation
    
    Args:
        num_qubits: Number of spins in the chain
        h_z: Strength of longitudinal field (sigma_z term)
        J: Coupling strength between nearest neighbors
        h_y: Strength of transverse field (sigma_y term)
        epsilon: Small parameter for spin-flip precision
        initial_state: Initial quantum state as bitstring (e.g., "1010")
        num_cycles: Number of evolution cycles
    """
    
    # Create exact simulation object
    exact_sim = TiltedIsingChain(num_qubits, h_z, J, h_y, epsilon)
    
    # Calculate magnetizations
    mag_data = exact_sim.calculate_megnetizations(initial_state, num_cycles)
    
    # Plot heatmap
    exact_sim.plot_heatmap(initial_state, num_cycles)

def plot_heatmap_circuit(num_qubits, h_z, J, h_y, epsilon, initial_state, num_cycles=50):
    """
    Plot magnetization heatmap for circuit simulation
    
    Args:
        num_qubits: Number of spins in the chain
        h_z: Strength of longitudinal field (sigma_z term)
        J: Coupling strength between nearest neighbors
        h_y: Strength of transverse field (sigma_y term)
        epsilon: Small parameter for spin-flip precision
        initial_state: Initial quantum state as bitstring (e.g., "1010")
        num_cycles: Number of evolution cycles
    """
    
    # Create circuit simulation object
    circuit_sim = IsingChainSimulation(num_qubits, h_z, J, h_y, epsilon)
    
    # Simulate and plot heatmap
    circuit_sim.plot_heatmap(initial_state, num_cycles)

def plot_heatmap_circuit_with_noise(num_qubits, h_z, J, h_y, epsilon, initial_state, num_cycles=50, noise_model=None):
    """
    Plot magnetization heatmap for circuit simulation with noise
    
    Args:
        num_qubits: Number of spins in the chain
        h_z: Strength of longitudinal field (sigma_z term)
        J: Coupling strength between nearest neighbors
        h_y: Strength of transverse field (sigma_y term)
        epsilon: Small parameter for spin-flip precision
        initial_state: Initial quantum state as bitstring (e.g., "1010")
        num_cycles: Number of evolution cycles
        noise_model: Noise model for simulation (default: None)
    """
    
    # Create circuit simulation object
    circuit_sim = IsingChainSimulation(num_qubits, h_z, J, h_y, epsilon)
    
    # Simulate and plot heatmap with noise
    circuit_sim.plot_heatmap_with_noise(initial_state, noise_model, num_cycles)

def compare_difference(num_qubits, h_z, J, h_y, epsilon, initial_state, num_cycles=50):
    """
    Compare exact simulation with circuit simulation
    
    Args:
        num_qubits: Number of spins in the chain
        h_z: Strength of longitudinal field (sigma_z term)
        J: Coupling strength between nearest neighbors
        h_y: Strength of transverse field (sigma_y term)
        epsilon: Small parameter for spin-flip precision
        initial_state: Initial quantum state as bitstring (e.g., "1010")
        num_cycles: Number of evolution cycles
    """
    
    # Exact simulation using QuTiP
    exact_sim = TiltedIsingChain(num_qubits, h_z, J, h_y, epsilon)
    exact_magnetizations = exact_sim.calculate_megnetizations(initial_state, num_cycles)
    
    # Circuit simulation using Qiskit
    circuit_sim = IsingChainSimulation(num_qubits, h_z, J, h_y, epsilon)
    circuit_state = circuit_sim.simulate_statevector_evolution(initial_state, num_cycles)
    circuit_magnetizations = circuit_sim.calculate_magnetizations(circuit_state)

    # Plot both results side by side for comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(exact_magnetizations,
               cmap='bwr',
               aspect='auto',
               origin='lower',
               vmin=-1.0,
               vmax=1.0,
               extent=[0, num_cycles, 0, num_qubits])
    plt.title("Exact Simulation Magnetization")
    plt.xlabel("Period")
    plt.ylabel("Site Index")
    plt.colorbar(label=r'$\langle \sigma_z \rangle$')
    
    plt.subplot(1, 3, 2)
    plt.imshow(circuit_magnetizations,
               cmap='bwr',
               aspect='auto',
               origin='lower',
               vmin=-1.0,
               vmax=1.0,
               extent=[0, num_cycles, 0, num_qubits])
    plt.title("Circuit Simulation Magnetization")
    plt.xlabel("Period")
    plt.ylabel("Site Index")
    plt.colorbar(label=r'$\langle \sigma_z \rangle$')

    plt.subplot(1, 3, 3)
    plt.imshow(circuit_magnetizations- exact_magnetizations,
               cmap='bwr',
               aspect='auto',
               origin='lower',
               vmin=-1.0,
               vmax=1.0,
               extent=[0, num_cycles, 0, num_qubits])
    plt.title("Difference (Circuit - Exact)")
    plt.xlabel("Period")
    plt.ylabel("Site Index")
    plt.colorbar(label=r'$\langle \sigma_z \rangle$')

    plt.tight_layout()
    plt.show()

    print("Max difference:", np.max(np.abs(circuit_magnetizations - exact_magnetizations)))

def compare_difference_with_noise(num_qubits, h_z, J, h_y, epsilon, initial_state, num_cycles=50, noise_model=None):
    """
    Compare exact simulation with circuit simulation with noise
    
    Args:
        num_qubits: Number of spins in the chain
        h_z: Strength of longitudinal field (sigma_z term)
        J: Coupling strength between nearest neighbors
        h_y: Strength of transverse field (sigma_y term)
        epsilon: Small parameter for spin-flip precision
        initial_state: Initial quantum state as bitstring (e.g., "1010")
        num_cycles: Number of evolution cycles
        noise_model: Noise model for simulation (default: None)
    """
    
    # Exact simulation using QuTiP
    exact_sim = TiltedIsingChain(num_qubits, h_z, J, h_y, epsilon)
    exact_magnetizations = exact_sim.calculate_megnetizations(initial_state, num_cycles)
    
    # Circuit simulation using Qiskit
    circuit_sim = IsingChainSimulation(num_qubits, h_z, J, h_y, epsilon)
    circuit_state = circuit_sim.simulate_statevector_evolution(initial_state, num_cycles)
    circuit_magnetizations = circuit_sim.calculate_magnetizations(circuit_state)

    # Circuit simulation with noise using Qiskit
    # circuit_sim_noisy = IsingChainSimulation(num_qubits, h_z, J, h_y, epsilon)
    circuit_magnetizations_noisy = circuit_sim.simulate_with_noise(initial_state, noise_model, num_cycles)
    # Plot both results side by side for comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(exact_magnetizations,
                cmap='bwr',
                aspect='auto',
                origin='lower',
                vmin=-1.0,
                vmax=1.0,
                extent=[0, num_cycles, 0, num_qubits])
    plt.title("Exact Simulation Magnetization")
    plt.xlabel("Period")
    plt.ylabel("Site Index")
    plt.colorbar(label=r'$\langle \sigma_z \rangle$')

    plt.subplot(1, 3, 2)
    plt.imshow(circuit_magnetizations,
               cmap='bwr',
               aspect='auto',
               origin='lower',
               vmin=-1.0,
               vmax=1.0,
               extent=[0, num_cycles, 0, num_qubits])
    plt.title("Circuit Simulation Magnetization")
    plt.xlabel("Period")
    plt.ylabel("Site Index")
    plt.colorbar(label=r'$\langle \sigma_z \rangle$')

    plt.subplot(1, 3, 3)
    plt.imshow(circuit_magnetizations_noisy,
               cmap='bwr',
               aspect='auto',
               origin='lower',
               vmin=-1.0,
               vmax=1.0,
               extent=[0, num_cycles, 0, num_qubits])
    plt.title("Circuit Simulation with Noise")
    plt.xlabel("Period")
    plt.ylabel("Site Index")
    plt.colorbar(label=r'$\langle \sigma_z \rangle$')
    plt.tight_layout()
    plt.show()

def compare_difference_with_noise_echo(num_qubits, h_z, J, h_y, epsilon, initial_state, num_cycles=50, noise_model=None):
    """
    Compare exact simulation with circuit simulation with noise and echo circuit
    
    Args:
        num_qubits: Number of spins in the chain
        h_z: Strength of longitudinal field (sigma_z term)
        J: Coupling strength between nearest neighbors
        h_y: Strength of transverse field (sigma_y term)
        epsilon: Small parameter for spin-flip precision
        initial_state: Initial quantum state as bitstring (e.g., "1010")
        num_cycles: Number of evolution cycles
        noise_model: Noise model for simulation (default: None)
    """
    
    # Exact simulation using QuTiP
    exact_sim = TiltedIsingChain(num_qubits, h_z, J, h_y, epsilon)
    exact_magnetizations = exact_sim.calculate_megnetizations(initial_state, num_cycles)
    
    # Circuit simulation using Qiskit
    circuit_sim = IsingChainSimulation(num_qubits, h_z, J, h_y, epsilon)
    circuit_state = circuit_sim.simulate_statevector_evolution(initial_state, num_cycles)
    circuit_magnetizations = circuit_sim.calculate_magnetizations(circuit_state)

    # Circuit simulation with noise using Qiskit
    circuit_magnetizations_noisy = circuit_sim.simulate_with_noise(initial_state, noise_model, num_cycles)

    # Circuit simulation normalized with echo circuit
    circuit_magnetizations_echo = circuit_sim.simulate_echo_circuit(initial_state, noise_model, num_cycles)
    circuit_magnetizations_normalized = circuit_magnetizations_noisy / np.sqrt(np.abs(circuit_magnetizations_echo))
    # Plot both results side by side for comparison
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(exact_magnetizations,
                cmap='bwr',
                aspect='auto',
                origin='lower',
                vmin=-1.0,
                vmax=1.0,
                extent=[0, num_cycles, 0, num_qubits])
    plt.title("Exact Simulation Magnetization")
    plt.xlabel("Period")
    plt.ylabel("Site Index")
    plt.colorbar(label=r'$\langle \sigma_z \rangle$')
    plt.subplot(1, 4, 2)
    plt.imshow(circuit_magnetizations,
               cmap='bwr',
               aspect='auto',
               origin='lower',
               vmin=-1.0,
               vmax=1.0,
               extent=[0, num_cycles, 0, num_qubits])
    plt.title("Circuit Simulation Magnetization")
    plt.xlabel("Period")
    plt.ylabel("Site Index")
    plt.colorbar(label=r'$\langle \sigma_z \rangle$')
    plt.subplot(1, 4, 3)
    plt.imshow(circuit_magnetizations_noisy,
               cmap='bwr',
               aspect='auto',
               origin='lower',
               vmin=-1.0,
               vmax=1.0,
               extent=[0, num_cycles, 0, num_qubits])
    plt.title("Circuit Simulation with Noise")
    plt.xlabel("Period")
    plt.ylabel("Site Index")
    plt.colorbar(label=r'$\langle \sigma_z \rangle$')
    plt.subplot(1, 4, 4)
    plt.imshow(circuit_magnetizations_normalized,
               cmap='bwr',
               aspect='auto',
               origin='lower',
               vmin=-1.0,
               vmax=1.0,
               extent=[0, num_cycles, 0, num_qubits])
    plt.title("Circuit Simulation Normalized")
    plt.xlabel("Period")
    plt.ylabel("Site Index")
    plt.colorbar(label=r'$\langle \sigma_z \rangle$')
    plt.tight_layout()
    plt.show()