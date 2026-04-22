from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)
from qiskit.circuit.library import IGate
from enum import Enum

class CorrectionMethod(Enum):
    BITFLIP_3LINES = 1
    FULL_LMPZ_5LINES = 2
    PHASEFLIP_3LINES = 3
    STEANE_7LINES = 4


class ErrorModel():
    """
    A class in charge of simulating noise and implementing the functions that will try to correct it.
    An instance of this class needs to be supplied to the CorrectedCircuit class upon creation. 
    The `CorrectedCircuit` instance will then automatically use this instance's methods when necessary.
    """

    def __init__(self, 
                 max_size_of_mcx,
                 p_bitflip : float = 0.001, 
                 p_bitflip_measure : float = 0.0,
                 p_phaseflip : float = 0.001,
                 p_phaseflip_measure : float = 0.0,
                 error_correction_period : int = 80,
                 correction_method : CorrectionMethod = CorrectionMethod.BITFLIP_3LINES):
        """
        Initiates an ErrorModel instance which can later be used to get NoiseModel instances.

        :param max_size_of_mcx: The maximum size of an mcx gate in the circuit, counting the control qubits and the output qubit. (Minimum size is 3).
        :param p_bitflip: Probability of getting a bitflip error on a quantum line when applying a gate.
        :param p_bitflip_measure: Probability of getting a bitflip error when measuring a line.
        :param p_phaseflip: Same, but for phaseflip.
        :param p_phaseflip_measure: Same, but for phaseflip.
        :param error_correction_period: This is the T parameter that sets the number of successive 
        quantum gates that triggers a correction. 
        :param correction_method: The actual correction to use. The functions are implemented in this
          class and are called by the CorrectedCircuit class when they need to be triggered.
        """
        self.p_bitflip = p_bitflip
        self.p_bitflip_measure = p_bitflip_measure
        self.p_phaseflip = p_phaseflip
        self.p_phaseflip_measure = p_phaseflip_measure
        self.T = error_correction_period
        self.correction_method = correction_method
        self.max_size_of_mcx = max_size_of_mcx

    def mcx_error(self, n):
        # Single-qubit combined Pauli channel
        # I, X (bit-flip), Z (phase-flip), Y (both)
        p_identity = 1 - self.p_bitflip - self.p_phaseflip

        single_qubit_error = pauli_error([
            ('I', 1 - self.p_bitflip - self.p_phaseflip),
            ('X', self.p_bitflip),
            ('Z', self.p_phaseflip),
            ('Y', 0.0)  # optional: set if you want combined events explicitly
        ])

        # Tensor product across n qubits
        error = single_qubit_error
        for _ in range(n - 1):
            error = error.tensor(single_qubit_error)

        return error

    def get_noise_model_instance(self) -> NoiseModel:
        """
        Returns a `NoiseModel` objects that follows the error simulation parameters supplied upon the `ErrorModel` instance creation.
        """
        bit_flip = (self.p_bitflip != 0.0)
        phase_flip = (self.p_phaseflip != 0.0)

        single_gate_bitflip = pauli_error([("X", self.p_bitflip), ("I", 1-self.p_bitflip)]) # Bit flip sur des portes a 1 qubit
        bitflip_measure = pauli_error([("X", self.p_bitflip_measure), ("I", 1-self.p_bitflip_measure)])
        flip_cx = single_gate_bitflip.tensor(single_gate_bitflip)
        
        singlegate_phaseflip = pauli_error([('Z', self.p_phaseflip), ('I', 1 - self.p_phaseflip)])
        phaseflip_measure = pauli_error([('Z', self.p_phaseflip_measure), ('I', 1 - self.p_phaseflip_measure)])
        phaseflip_cx = singlegate_phaseflip.tensor(singlegate_phaseflip)

        singlegate_bitphaseflip = single_gate_bitflip.compose(singlegate_phaseflip)
        bitphaseflip_measure = bitflip_measure.compose(phaseflip_measure)
        bitphaseflip_cx = flip_cx.compose(phaseflip_cx)

        noise = NoiseModel()

        for n in range(3, self.max_size_of_mcx):
            error = self.mcx_error(n)
            noise.add_all_qubit_quantum_error(error, ['mcx'], num_qubits=n)

        if bit_flip and phase_flip:
            noise.add_all_qubit_quantum_error(singlegate_bitphaseflip, ["rz", "sx", "x"])
            noise.add_all_qubit_quantum_error(bitphaseflip_measure, "measure")
            noise.add_all_qubit_quantum_error(bitphaseflip_cx, ["cx"])
        elif bit_flip:
            noise.add_all_qubit_quantum_error(single_gate_bitflip, ["rz", "sx", "x"])
            noise.add_all_qubit_quantum_error(bitflip_measure, "measure")
            noise.add_all_qubit_quantum_error(flip_cx, ["cx"])
        elif phase_flip:
            noise.add_all_qubit_quantum_error(singlegate_phaseflip, ["rz", "sx", "x"])
            noise.add_all_qubit_quantum_error(phaseflip_measure, "measure")
            noise.add_all_qubit_quantum_error(phaseflip_cx, ["cx"])


        # rz : rotation axe Z
        # sx : racine de x
        # x : not gate
        # Implémenter l'erreur sur ces 3 portes est équivalent à l'implémenter sur toutes les portes a un qubit.
        return noise
    


    def get_num_physical_lines(self) -> int:
        """
        Returns the number of physical lines that needs to be used to encode one logical qubit.
        This number is not always the same depending on the correction method used. 
        """
        if self.correction_method == CorrectionMethod.BITFLIP_3LINES or self.correction_method == CorrectionMethod.PHASEFLIP_3LINES:
            return 3
        elif self.correction_method == CorrectionMethod.FULL_LMPZ_5LINES:
            return 5
        elif self.correction_method == CorrectionMethod.STEANE_7LINES:
            return 7
        
        raise Exception(f"The correction method {self.correction_method} was not fully implemented in the ErrorModel methods's function get_num_physical_lines.")

    def get_num_ancillas_needed(self) -> int:
        """
        Returns the number of additional ancillas (working lines) that are used when
        correcting one logical lines. Again, this number depends on the choosen correction method.
        """
        if self.correction_method == CorrectionMethod.BITFLIP_3LINES or self.correction_method == CorrectionMethod.PHASEFLIP_3LINES:
            return 2
        elif self.correction_method == CorrectionMethod.FULL_LMPZ_5LINES:
            return 4
        elif self.correction_method == CorrectionMethod.STEANE_7LINES:
            return 6   # 3 X stabilizers + 3 Z stabilizers

        
        raise Exception(f"The correction method {self.correction_method} was not fully implemented in the ErrorModel methods's function get_num_ancillas_needed.")

    def get_num_classical_needed(self) -> int:
        """
        Each correction method uses a certain number of classical lines to store stuff.
        """
        if self.correction_method == CorrectionMethod.BITFLIP_3LINES or self.correction_method == CorrectionMethod.PHASEFLIP_3LINES :
            return 1
        elif self.correction_method == CorrectionMethod.FULL_LMPZ_5LINES:
            return 4
        elif self.correction_method == CorrectionMethod.STEANE_7LINES:
            return 6
        
        raise Exception(f"The correction method {self.correction_method} was not fully implemented in the ErrorModel methods's function get_num_classical_needed.")


    def correct_line(self, i : int, qcc):
        """
        Correct the `i`th quantum line in the circuit associated with the QuantumCircuitCorrected object.

        :param i: Qubit line to correct
        :param qcc: QuantumCircuitCorrected instance that stores the circuit we want to correct.
        """

        block = IGate().to_mutable()
        block.label = "Correction"
        qcc.circuit.append(block, [i])

        if self.correction_method == CorrectionMethod.BITFLIP_3LINES:
            self._bitflip_correct_qubit(i, qcc)
        elif self.correction_method == CorrectionMethod.FULL_LMPZ_5LINES:
            self._full_lmpz_5lines(i, qcc)
        elif self.correction_method == CorrectionMethod.PHASEFLIP_3LINES :
            self._phaseflip_3lines(i, qcc)
        elif self.correction_method == CorrectionMethod.STEANE_7LINES:
            self._steane_correct(i, qcc)
        else:
            raise Exception(f"The specified error correction method ({self.correction_method}) is not fully implemented (in correct_line method)")

    def _bitflip_correct_qubit(self, i : int, qcc, show_barriers = True):
        """
        :param i: Id of the qubit line to correct (not all qubit line 
        gets corrected : only those who have had enough gates applied)
        :param qcc: QuantumCircuitCorrected instance.

        Version plus compacte utilisant la logique:
            - Corriger q1 si syndrome = 10
            - Corriger q2 si syndrome = 01
            - Corriger q0 si syndrome = 11
        """

        # Here we know that 3*i 3*i + 1 and 3*i + 2 exists because the ErrorModel defined the right number of physical and measure qubits.
        q0, q1, q2 = 3 * i, 3*i + 1, 3*i + 2
        a0, a1 = 3 * qcc.n, 3 * qcc.n + 1
        qc = qcc.circuit_corrected
        # comparer q1 et q0
        if show_barriers:
            qc.barrier()
        qc.cx(q0, a0)
        qc.cx(q1, a0)
        
        # comparer q2 et q0
        qc.cx(q0, a1)
        qc.cx(q2, a1)
        
        # Corrections avec portes Toffoli et X
        # Pour q1: a0=1, a1=0
        qc.x(a1)
        qc.ccx(a0, a1, q1)
        qc.x(a1)
        
        # Pour q2: a0=0, a1=1
        qc.x(a0)
        qc.ccx(a0, a1, q2)
        qc.x(a0)
        
        # Pour q0: a0=1, a1=1
        qc.ccx(a0, a1, q0)

        synd = qcc.synd
        
        # Réinitialiser les ancillas (pour réutilisation)
        # Pour réinitialiser, on fait une mesure vers un qubit de mesure
        # puis on applique un conditional X
        qc.measure(a0, synd[0])
        with qc.if_test((synd[0], 1)):
            qc.x(a0)
        qc.measure(a1, synd[0])
        with qc.if_test((synd[0], 1)):
            qc.x(a1)

        if show_barriers:
            qc.barrier()

    
    # Code de correction à 5 qubits physiques et 4 ancillas
    def _full_lmpz_5lines(self, i : int, qcc):
        """
        Implémente la correction de bitflip et phaseflip en meme temps.
        
        :param i: Id of the qubit line to correct (not all qubit line gets corrected : only those who have had enough gates applied)
        :param qcc: QuantumCircuitCorrected instance.
        """
        qc = qcc.circuit_corrected
        q0, q1, q2 , q3, q4= 5 * i, 5*i + 1, 5*i + 2, 5*i+3, 5*i+4
        a0, a1 , a2, a3= 5 * qcc.n, 5 * qcc.n + 1, 5 * qcc.n + 2, 5 * qcc.n + 3
        
        qc.h(a0)
        qc.cx(q0, a0)
        qc.cz(q1, a0)
        qc.cz(q2, a0)
        qc.cx(q3, a0)
        # Identité pour q4
        qc.h(a0)
        
        qc.h(a1)
        # Identité pour q0
        qc.cx(q1, a1)
        qc.cz(q2, a1)
        qc.cz(q3, a1)
        qc.cx(q4, a1)
        qc.h(a1)
        
        qc.h(a2)
        qc.cx(q0, a2)
        # Identité pour q1
        qc.cx(q2, a2)
        qc.cz(q3, a2)
        qc.cz(q4, a2)
        qc.h(a2)
        
        qc.h(a3)
        qc.cx(q0, a3)
        qc.cz(q1, a3)
        # Identité pour q2
        qc.cz(q3, a3)
        qc.cx(q4, a3)
        qc.h(a3)
        
        synd = qcc.synd
        
        # Mesure des syndromes
        qc.measure(a0, synd[0])
        qc.measure(a1, synd[1])
        qc.measure(a2, synd[2])
        qc.measure(a3, synd[3])
        
        with qc.if_test((synd, 1)): qc.x(q0)   # 0001
        with qc.if_test((synd, 2)): qc.z(q2)   # 0010
        with qc.if_test((synd, 3)): qc.x(q4)   # 0011
        with qc.if_test((synd, 4)): qc.z(q4)   # 0100
        with qc.if_test((synd, 5)): qc.z(q1)   # 0101
        with qc.if_test((synd, 6)): qc.x(q3)   # 0110
        with qc.if_test((synd, 7)): qc.y(q4)   # 0111
        with qc.if_test((synd, 8)): qc.x(q1)   # 1000
        with qc.if_test((synd, 9)): qc.z(q3)   # 1001
        with qc.if_test((synd, 10)): qc.z(q0)  # 1010
        with qc.if_test((synd, 11)): qc.y(q0)  # 1011
        with qc.if_test((synd, 12)): qc.x(q2)  # 1100
        with qc.if_test((synd, 13)): qc.y(q1)  # 1101
        with qc.if_test((synd, 14)): qc.y(q2)  # 1110
        with qc.if_test((synd, 15)): qc.y(q3)  # 1111
        
        
        # Réinitialiser les ancillas (pour réutilisation)
        
        with qc.if_test((synd[0], 1)): qc.x(a0)
        with qc.if_test((synd[1], 1)): qc.x(a1)
        with qc.if_test((synd[2], 1)): qc.x(a2)
        with qc.if_test((synd[3], 1)): qc.x(a3)


    # Code de correction de phaseflip à 3 qubits physiques et 2 ancillas
    def _phaseflip_3lines(self, i, qcc):
        """
        Implémente la correction de phaseflip en utilisant la même logique que pour le bitflip
        Applique des portes H avant et après pour transformer les phase flips en bit flips.
        """
       
        qc = qcc.circuit_corrected
        q0, q1, q2 = 3 * i, 3*i + 1, 3*i + 2
        qc.barrier()
        qc.h(q0)
        qc.h(q1)
        qc.h(q2)
        
        self._bitflip_correct_qubit(i, qcc, show_barriers=False)
        
        qc.h(q0)
        qc.h(q1)
        qc.h(q2)
        qc.barrier()

    def _steane_correct(self, i, qcc):
        qc = qcc.circuit_corrected

        # Qubits physiques (7)
        q = [7 * i + k for k in range(7)]

        # Ancillas (6)
        a = [7 * qcc.n + k for k in range(6)]

        synd = qcc.synd  # supposé être un registre classique de taille 6

        qc.barrier()

        # =========================
        # 1) MESURE DES STABILIZERS Z (détection X errors)
        # =========================

        # S0: Z Z Z Z I I I
        qc.cx(q[0], a[0])
        qc.cx(q[1], a[0])
        qc.cx(q[2], a[0])
        qc.cx(q[3], a[0])

        # S1: Z Z I I Z Z I
        qc.cx(q[0], a[1])
        qc.cx(q[1], a[1])
        qc.cx(q[4], a[1])
        qc.cx(q[5], a[1])

        # S2: Z I Z I Z I Z
        qc.cx(q[0], a[2])
        qc.cx(q[2], a[2])
        qc.cx(q[4], a[2])
        qc.cx(q[6], a[2])

        # =========================
        # 2) MESURE DES STABILIZERS X (détection Z errors)
        # =========================

        # Préparation en base X
        for k in range(3, 6):
            qc.h(a[k])

        # S3: X X X X I I I
        qc.cz(q[0], a[3])
        qc.cz(q[1], a[3])
        qc.cz(q[2], a[3])
        qc.cz(q[3], a[3])

        # S4: X X I I X X I
        qc.cz(q[0], a[4])
        qc.cz(q[1], a[4])
        qc.cz(q[4], a[4])
        qc.cz(q[5], a[4])

        # S5: X I X I X I X
        qc.cz(q[0], a[5])
        qc.cz(q[2], a[5])
        qc.cz(q[4], a[5])
        qc.cz(q[6], a[5])

        # Retour base Z
        for k in range(3, 6):
            qc.h(a[k])

        # =========================
        # 3) MESURE DES SYNDROMES
        # =========================

        for k in range(6):
            qc.measure(a[k], synd[k])

        qc.barrier()

        # =========================
        # 4) CORRECTION DES ERREURS
        # =========================

        # --- BIT FLIP (X errors) ---
        # syndrome = (a0,a1,a2)

        with qc.if_test((synd[0], 1)):
            with qc.if_test((synd[1], 0)):
                with qc.if_test((synd[2], 0)):
                    qc.x(q[3])   # 100

        with qc.if_test((synd[0], 0)):
            with qc.if_test((synd[1], 1)):
                with qc.if_test((synd[2], 0)):
                    qc.x(q[5])   # 010

        with qc.if_test((synd[0], 0)):
            with qc.if_test((synd[1], 0)):
                with qc.if_test((synd[2], 1)):
                    qc.x(q[6])   # 001

        with qc.if_test((synd[0], 1)):
            with qc.if_test((synd[1], 1)):
                with qc.if_test((synd[2], 0)):
                    qc.x(q[1])   # 110

        with qc.if_test((synd[0], 1)):
            with qc.if_test((synd[1], 0)):
                with qc.if_test((synd[2], 1)):
                    qc.x(q[2])   # 101

        with qc.if_test((synd[0], 0)):
            with qc.if_test((synd[1], 1)):
                with qc.if_test((synd[2], 1)):
                    qc.x(q[4])   # 011

        with qc.if_test((synd[0], 1)):
            with qc.if_test((synd[1], 1)):
                with qc.if_test((synd[2], 1)):
                    qc.x(q[0])   # 111

        # --- PHASE FLIP (Z errors) ---
        # syndrome = (a3,a4,a5)

        with qc.if_test((synd[3], 1)):
            with qc.if_test((synd[4], 0)):
                with qc.if_test((synd[5], 0)):
                    qc.z(q[3])

        with qc.if_test((synd[3], 0)):
            with qc.if_test((synd[4], 1)):
                with qc.if_test((synd[5], 0)):
                    qc.z(q[5])

        with qc.if_test((synd[3], 0)):
            with qc.if_test((synd[4], 0)):
                with qc.if_test((synd[5], 1)):
                    qc.z(q[6])

        with qc.if_test((synd[3], 1)):
            with qc.if_test((synd[4], 1)):
                with qc.if_test((synd[5], 0)):
                    qc.z(q[1])

        with qc.if_test((synd[3], 1)):
            with qc.if_test((synd[4], 0)):
                with qc.if_test((synd[5], 1)):
                    qc.z(q[2])

        with qc.if_test((synd[3], 0)):
            with qc.if_test((synd[4], 1)):
                with qc.if_test((synd[5], 1)):
                    qc.z(q[4])

        with qc.if_test((synd[3], 1)):
            with qc.if_test((synd[4], 1)):
                with qc.if_test((synd[5], 1)):
                    qc.z(q[0])

        qc.barrier()

        # =========================
        # 5) RESET ANCILLAS
        # =========================

        for k in range(6):
            with qc.if_test((synd[k], 1)):
                qc.x(a[k])

        qc.barrier()