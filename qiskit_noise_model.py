import copy
import numpy as np
from qiskit_aer.noise import NoiseModel,ReadoutError,amplitude_damping_error,phase_damping_error,depolarizing_error,thermal_relaxation_error,pauli_error
from qiskit.circuit.library import IGate


class MyNoiseModel:
    '''
    custom noise model
    '''
    def __init__(self):

        self.noise_model = NoiseModel(
            basis_gates=['u3', 'cz','measure']
        )
        
    def add_sq_pauli_error(self, error=0.0):
        D = 2**1
        depo_error = depolarizing_error(error / (1 - 1 / D**2), 1)
        self.noise_model.add_all_qubit_quantum_error(depo_error, ['u3','h','z','x'])
    
    def add_tq_pauli_error(self, error=0.0):
        D = 2**2
        depo_error = depolarizing_error(error / (1 - 1 / D**2), 2)
        self.noise_model.add_all_qubit_quantum_error(depo_error, ['cz','cp','unitary'])

    def add_readout_error(self, error=0.0):
        if np.isclose(error, 0):
            return
        probabilities = [[1-error, error], [error, 1-error]]
        error_meas = ReadoutError(np.array(probabilities).T)
        self.noise_model.add_all_qubit_readout_error(error_meas)

    def add_idle_error(self, gate_label, idle_error_dict):
        t1 = idle_error_dict['t1']
        t2 = idle_error_dict['t2']
        time = idle_error_dict['t_gate']
        error = thermal_relaxation_error(t1=t1, t2=t2, time=time)
        self.noise_model.add_all_qubit_quantum_error(error, [gate_label])

    def gen_noise_model(self, error_dict):
        if 'sq' in error_dict:
            self.add_sq_pauli_error(error_dict['sq'])
        if 'tq' in error_dict:
            self.add_tq_pauli_error(error_dict['tq'])
        if 'readout' in error_dict:
            self.add_readout_error(error_dict['readout'])
        if 'id' in error_dict:
            self.add_idle_error('id',error_dict['id'])

def get_noise_model(error_dict=None):
    my_noise_model = MyNoiseModel()
    error_dict = {
        "sq": 5.2e-4,
        "tq": 4.2e-3,
        "readout": 1e-2,
        "id": {
            "t1": 80e3,
            "t2": 20e3,
            "t_gate": 30,
        }
    } if error_dict is None else error_dict
    my_noise_model.gen_noise_model(error_dict)
    return my_noise_model.noise_model

if __name__ == "__main__":
    noise_model = get_noise_model()
    print(noise_model)
