import functools
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.circuit import Instruction
from qiskit.circuit.library import IGate
from error_model import ErrorModel
from qiskit import ClassicalRegister
from qiskit_aer.noise import (
    NoiseModel)

from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)


def correct_if_needed(func):
    """
    This wrapper will call the correction function on a specific line if it determines that, 
    given the rules explained in the rapport final, the line has had enough corrections.
    """
    @functools.wraps(func)
    def wrapper_correct(*args, **kwargs):
        selfe = args[0]

        affected_qubits = []

        if func.__name__ == "cx" or func.__name__ == "mcx":
            controls = args[1]
            if type(controls) != list:
                controls = [controls]
            applied = args[2]
            affected_qubits = controls + [applied]
            n = len(controls) + 1
            # Each line is allowed to have at most floor(T / n) previous gates.
            for c in affected_qubits:
                if selfe.gate_count[c] > selfe._error_model.T / n:
                    selfe._error_model.correct_line(c, selfe)
                    selfe.gate_count[c] = 0
            for c in controls:
                selfe.gate_count[applied] += selfe.gate_count[c]
        else:
            affected_qubits = [args[1]]


        func(*args, **kwargs)
        for q in affected_qubits:
            if selfe.gate_count[q] == selfe._error_model.T:
                selfe.gate_count[q] = 0
                selfe._error_model.correct_line(q, selfe)
            
            selfe.gate_count[q] += 1

    return wrapper_correct

class QuantumCircuitCorrected:
    """
    Used to build a quantum circuit with integrated bitflip error correction.
    """

    def __init__(self, num_quantum_lines : int, num_classical_lines : int, name : str, error_model : ErrorModel):
        """
        Docstring for __init__
        
        :param num_quantum_lines: Number of qubits (quantum lines) to set up in the circuit.
        :param num_classical_lines: The number of classical lines to set up.
        :param name: Title displayed when drawing the circuit.
        :param error_model: Error model to apply. The Error Model defines both the noise and the correction method.
        """
        self.n, self.m, self.name = num_quantum_lines, num_classical_lines, name 

        # The number of ancillas that are needed for the correction operation.
        self.ancilla = error_model.get_num_ancillas_needed()
        # The number of physical lines that encode a logical qubit.
        self.physical = error_model.get_num_physical_lines()
        # The number of classical lines needed for the error correction.
        self.classical = error_model.get_num_classical_needed()
        
        self.circuit = QuantumCircuit(self.n,self.m, name=name)
        self.circuit_corrected = QuantumCircuit(self.physical * self.n + self.ancilla, self.physical * self.m, name=name)
        self.synd = ClassicalRegister(self.classical, "synd")
        self.circuit_corrected.add_register(self.synd)

        # gate_count counts the number of gates applied consecutively (without error correction) on a line.
        self.gate_count = [0 for i in range(self.n)]

        self._error_model = error_model
        # Noise will be used when running the circuit.
        self.noise = self._error_model.get_noise_model_instance()


    @classmethod
    def fromExistingCircuit(cls, circuit : QuantumCircuit, error_model : ErrorModel):
        """
        Takes an existing qiskit quantum circuit and build an instance of `QuantomCircuitCorrected`.
        """
        qcc = QuantumCircuitCorrected(circuit.num_qubits, circuit.num_clbits,circuit.name, error_model)
        for inst, qargs, cargs in circuit.data:
            qubit_indices = [q._index for q in qargs]
            for c in cargs:
                qubit_indices.append(c._index)
            
            if inst.name == "mcx":
                getattr(qcc, inst.name)(qubit_indices[:-1], qubit_indices[-1])
            else:
                getattr(qcc, inst.name)(*qubit_indices)
        return qcc
    

    @correct_if_needed
    def h(self, q):
        self.circuit.h(q)

        base = self.physical * q
        for i in range(1, self.physical):
            self.circuit_corrected.cx(base, base + i)
        self.circuit_corrected.h(self.physical * q)
        for i in range(self.physical - 1):
            self.circuit_corrected.cx(self.physical * q, self.physical * q + i + 1)

    @correct_if_needed
    def cx(self, control: int, apply: int):
        self.circuit.cx(control, apply)
        for i in range(self.physical):
            self.circuit_corrected.cx(self.physical * control + i, self.physical * apply + i)
    
    def ccx(self, control1: int, control2: int, apply: int):
        self.mcx([control1, control2], apply)

    @correct_if_needed
    def x(self, q: int):
        self.circuit.x(q)
        for i in range(self.physical):
            self.circuit_corrected.x(self.physical * q + i)

    @correct_if_needed
    def mcx(self, controls: list[int], apply : int):
        self.circuit.mcx(controls, apply)
        for i in range(self.physical):
            self.circuit_corrected.mcx([self.physical * c + i for c in controls], apply * self.physical + i)

    def barrier(self, *args):
        self.circuit.barrier()
        self.circuit_corrected.barrier()

    def measure(self, i : int, j : int):
        """
        Measure qubit i into classical line j
        """
        self.circuit.measure(i, j)
        for k in range(self.physical):
            self.circuit_corrected.measure(self.physical * i + k, self.physical*j + k)


    def draw(self, corrected : bool = True) -> None:
        """
        Matplot lib display of the circuit
        
        :param corrected: If set to True will display the corrected version.
        """
        if corrected:
            self.circuit_corrected.draw(output="mpl")
        else:
            self.circuit.draw(output="mpl")
        plt.show()

    def correct_on_line(self, i):
        """
        Bypasses the automatic correction and correct the supplied line.
        """
        self._error_model.correct_line(i, self)

    def run_circuit(self, transpiling = True):
        noise = self.noise

        simulator = AerSimulator(method="automatic", noise_model = noise)
        circuit = self.circuit
        if transpiling:
            circuit = transpile(self.circuit, simulator, coupling_map=None)
        result = simulator.run(circuit).result()
        counts = result.get_counts(circuit)

        simulatorcc = AerSimulator(method="matrix_product_state", noise_model = noise)
        corrected_circuit = self.circuit_corrected 

        if transpiling:
            # we want to keep mcx gates (which explaines the basis gates and the optimization_level)
            #corrected_circuit = transpile(self.circuit_corrected, simulatorcc, coupling_map=None, basis_gates=['mcx'], optimization_level=0)
            pass
        corrected_result = simulatorcc.run(corrected_circuit, shots=1024, memory=True).result()
        
        #print("cregs:", self.circuit_corrected.cregs)
        #print("num classical bits:", self.circuit_corrected.num_clbits)

        #raw_memory = corrected_result.get_memory()
        #print("first 5 raw memory strings:", raw_memory[:5])
        #print("after split:", [r.split(" ") for r in raw_memory[:5]])
        # Results are formatted in this way : 0000 0000011111, the first part being the synd 
        # classical lines which can now be discarded.
        #print(corrected_result.get_memory())


        corrected_mem = [result.split(" ")[1] for result in corrected_result.get_memory()]
        corrected_counts = {}
        for mem in corrected_mem:
            value_measured = []
            
            n_groups = len(mem) // self.physical
            for k in range(n_groups):
                zero_vs_one = {0: 0, 1: 0}
                for t in range(self.physical):
                    # reverse indexing: rightmost bits = lowest classical index
                    bit_pos = len(mem) - 1 - (self.physical * k + t)
                    zero_vs_one[int(mem[bit_pos])] += 1

                value_measured.append('0' if zero_vs_one[0] >= zero_vs_one[1] else '1')

            value_measured = ''.join(reversed(value_measured))

            if value_measured in corrected_counts:
                corrected_counts[value_measured] += 1
            else:
                corrected_counts[value_measured] = 1

        return {"uncorrected" : counts, "corrected" : corrected_counts}


    