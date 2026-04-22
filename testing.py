from error_model import ErrorModel, CorrectionMethod
from corrected_circuit import QuantumCircuitCorrected
from qiskit import QuantumCircuit


def section_5_3_1():
    periods = []
    proportion_correct = []
    for T in range(10, 1000, 10):
        print(T)
        error_model = ErrorModel(max_size_of_mcx=3, p_bitflip=0.001, p_phaseflip=0, error_correction_period=T, 
                                correction_method=CorrectionMethod.BITFLIP_3LINES)
        qcc = QuantumCircuitCorrected(1, 1, name="section 5.3.1", error_model=error_model)
    
        for _ in range(1000):
            qcc.x(0)
        qcc.measure(0, 0)

        res = qcc.run_circuit(transpiling=False)
        proportion_correct.append(res["corrected"]["0"] / (res["corrected"]["0"] + res["corrected"]["1"]))
        periods.append(T)

    return periods, proportion_correct # Plot those values using matplotlib.

def section_5_3_1_multiple_lines():
    error_model = ErrorModel(max_size_of_mcx=3, p_bitflip=0.001, p_phaseflip=0, error_correction_period=20, 
                                correction_method=CorrectionMethod.BITFLIP_3LINES)
    qcc = QuantumCircuitCorrected(2, 2, name="section 5.3.1", error_model=error_model)
    for i in range(10):
        qcc.x(0)
    for i in range(11):
        qcc.x(1)
    qcc.cx(0, 1)
    for i in range(6):
        qcc.x(1)
    qcc.measure(0, 0)
    qcc.measure(1, 1)
    qcc.draw(corrected=False)
    print(qcc.run_circuit(transpiling=False))

def test():
    error_model = ErrorModel(max_size_of_mcx=3, p_bitflip=0.001, p_phaseflip=0, error_correction_period=200, 
                                correction_method=CorrectionMethod.BITFLIP_3LINES)
    qcc = QuantumCircuitCorrected(2, 2, name="section 5.3.1", error_model=error_model)
    for i in range(8):
        qcc.x(0)
    for i in range(3):
        qcc.x(1)
    qcc.correct_on_line(1)

    qcc.cx(0, 1)
    qcc.x(1)
    qcc.correct_on_line(1)
    for i in range(7):
        qcc.x(1)
    qcc.measure(0, 0)
    qcc.measure(1, 1)
    qcc.draw(corrected=False)
    print(qcc.run_circuit(transpiling=False))
    

    
    
print(test())