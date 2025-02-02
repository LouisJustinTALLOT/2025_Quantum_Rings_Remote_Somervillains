from QuantumRingsLib import QuantumCircuit, QuantumRegister, ClassicalRegister, job_monitor
import numpy as np
import math
import QR_secrets
import semiprimes
import time
import json
import matplotlib.pyplot as plt
from QuantumRingsLib import QuantumRegister, AncillaRegister, ClassicalRegister, QuantumCircuit


# https://github.com/tiagomsleao/ShorAlgQiskit/blob/master/Shor_Normal_QFT.py

from fractions import Fraction
# from helper import getAngle, cMULTmodN, get_factors
from helper_norm import getAngles, cMULTmodN, get_factors, create_inverse_QFT


shots = 256


def modular_exponentiation(base, exp, mod):
    """Computes (base^exp) % mod using iterative modular exponentiation."""
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp //= 2
    return result

def iqft_cct(qc, b, n):
    for i in range (n):
        for j in range (1, i+1):
            qc.cu1(-math.pi/2**(i-j+1), b[j-1], b[i])
        qc.h(b[i])
    qc.barrier()
    return


# https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/algorithms/shor_algorithm.ipynb


def extract_period_from_counts(counts, a, N):
    """
    Extracts the period r from measurement counts in Shor's Algorithm.

    Parameters:
    - counts: Dictionary of measurement results from the quantum circuit.
    - N: The integer to be factored.

    Returns:
    - r: The period found from phase estimation.
    """
    measured_values = list(counts.keys())  # Get binary strings
    probabilities = list(counts.values())  # Get occurrence counts

    print(measured_values, probabilities)

    measured_values = [int(meas, 2) for meas in counts.keys()]
    
    # Sort values by frequency (most common first)
    measured_values.sort(key=lambda x: counts[format(x, 'b')], reverse=True)
    
    # Number of qubits used for measurement (estimate from log2(N))
    num_qubits = int(np.ceil(np.log2(N)))  

    # Best candidate for the phase estimation
    best_candidate = measured_values[0] / (2 ** num_qubits)

    # Convert phase estimate to a fraction using continued fractions
    fraction = Fraction(best_candidate).limit_denominator(N)

    # The denominator of the fraction is the estimated period r
    r = fraction.denominator

    # Verify if r is correct: a^r ≡ 1 (mod N)
    a = np.random.randint(2, N)  # Same 'a' used in Shor’s algorithm
    if pow(a, r, N) == 1:
        return r
    else:
        return None


def shors_algorithm(N):
    if N % 2 == 0:
        return 2
    
    a = np.random.randint(2, N)

    # a = 4
    # N = 15

    if math.gcd(a, N) != 1:
        print("GCD is not 1, returning GCD")
        return math.gcd(a, N)
        
    n = int(np.ceil(np.log2(N)))


    # """auxilliary quantum register used in addition and multiplication"""
    # aux = QuantumRegister(n+2)
    # """single qubit where the sequential QFT is performed"""
    # up_reg = QuantumRegister(1)
    # """quantum register where the multiplications are made"""
    # down_reg = QuantumRegister(n)
    # """classical register where the measured values of the sequential QFT are stored"""
    # up_classic = ClassicalRegister(2*n)
    # """classical bit used to reset the state of the top qubit to 0 if the previous measurement was 1"""
    # c_aux = ClassicalRegister(1)
    # print(f'Looking for N={N} with a={a}, n={n} (total qubits: {2*n})')

    # circuit = QuantumCircuit(down_reg , up_reg , aux, up_classic, c_aux)

    # """ Initialize down register to 1"""
    # circuit.x(down_reg[0])

    # """ Cycle to create the Sequential QFT, measuring qubits and applying the right gates according to measurements """
    # for i in range(0, 2*n):
    #     """reset the top qubit to 0 if the previous measurement was 1"""
    #     circuit.x(up_reg).c_if(c_aux, 1)
    #     circuit.h(up_reg)
    #     cMULTmodN(circuit, up_reg[0], down_reg, aux, a**(2**(2*n-1-i)), N, n)
    #     """cycle through all possible values of the classical register and apply the corresponding conditional phase shift"""
    #     for j in range(0, 2**i):
    #         """the phase shift is applied if the value of the classical register matches j exactly"""
    #         circuit.u1(getAngle(j, i), up_reg[0]).c_if(up_classic, j)
    #     circuit.h(up_reg)
    #     circuit.measure(up_reg[0], up_classic[i])
    #     circuit.measure(up_reg[0], c_aux[0])


    """auxilliary quantum register used in addition and multiplication"""
    aux = QuantumRegister(n+2)
    """quantum register where the sequential QFT is performed"""
    up_reg = QuantumRegister(2*n)
    """quantum register where the multiplications are made"""
    down_reg = QuantumRegister(n)
    """classical register where the measured values of the QFT are stored"""
    up_classic = ClassicalRegister(2*n)

    """ Create Quantum Circuit """
    circuit = QuantumCircuit(down_reg , up_reg , aux, up_classic)

    """ Initialize down register to 1 and create maximal superposition in top register """
    circuit.h(up_reg)
    circuit.x(down_reg[0])

    """ Apply the multiplication gates as showed in the report in order to create the exponentiation """
    for i in range(0, 2*n):
        cMULTmodN(circuit, up_reg[i], down_reg, aux, int(pow(a, pow(2, i))), N, n)

    """ Apply inverse QFT """
    create_inverse_QFT(circuit, up_reg, 2*n ,1)

    """ Measure the top qubits, to get x value"""
    circuit.measure(up_reg,up_classic)

    # capture stdout to save qr circuit when calling qr.draw
    if False:
        import sys

        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        circuit.draw(fold=-1)

        # save to file
        with open(f'shor_circuit_{N}.txt', 'w') as f:
            f.write(sys.stdout.getvalue())

        sys.stdout = old_stdout

    # format depth and width as as string with explanations
    print(f"circuit depth: {circuit.depth()}, circuit width: {circuit.width()}")

    # exit()


    qc = circuit

    backend = QR_secrets.backend
    job = backend.run(qc, shots=shots)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts()

    measured_values = list(counts.keys())  # Get binary strings
    probabilities = list(counts.values())  # Get occurrence counts

    print(measured_values, probabilities)
    
    # r = extract_period_from_counts(counts, a, N)
    # exit()

    """ Initialize this variable """
    prob_success=0
    sim_result = result
    counts_result = list(counts.values())
    number_shots = shots
    """ For each simulation result, print proper info to user and try to calculate the factors of N"""
    i=0
    while i < len(counts_result):

        """ Get the x_value from the final state qubits """
        all_registers_output = list(sim_result.get_counts().keys())[i]
        output_desired = all_registers_output#.split(" ")[1]
        x_value = int(output_desired, 2)
        prob_this_result = 100 * ( int( list(sim_result.get_counts().values())[i] ) ) / (number_shots)

        print("------> Analysing result {0}. This result happened in {1:.4f} % of all cases\n".format(output_desired,prob_this_result))

        """ Print the final x_value to user """
        print('In decimal, x_final value for this result is: {0}\n'.format(x_value))

        """ Get the factors using the x value obtained """   
        success, f1, f2 = get_factors(int(x_value),int(2*n),int(N),int(a))

        if success==True:
            prob_success = prob_success + prob_this_result
            print("FOUND FACTORS: {0} and {1}\n".format(f1,f2))
            return f1, f2

        i=i+1

    # measured_value = int(list(counts.keys())[0], 2)

    # r = 1
    # while (a**r)%N != 1:
    #     r += 1
    #     # exit condition to prevent infinite loop
    #     if r > 10000:
    #         print("Warning: Exiting to prevent infinite loop")
    #         print(a, N)
    #         break

    # if r % 2 == 0 and (a**(r//2)+1)%N != 0:
    #     factor1 = math.gcd(a**(r//2)-1, N)
    #     factor2 = math.gcd(a**(r//2)+1, N)
    #     return factor1, factor2

    return None

if __name__ == "__main__":
    semiprimes_list = semiprimes.semiprimes
    summary_results = []
    
    for N in list(semiprimes_list.values())[:4]:
        start_time = time.time()
        factor = None
        while factor is None:
            factor = shors_algorithm(N)
        
        if isinstance(factor, tuple):
            factor = list(factor)
        else:
            factor = [factor, N // factor]
        
        elapsed_time = time.time() - start_time
        
        result_entry = {
            "semiprime": N,
            "factors": factor,
            "time_taken": elapsed_time
        }
        summary_results.append(result_entry)
        
        print(f"Semiprime {N} is factored into {factor} in {elapsed_time:.2f} seconds")
        
        with open("shor_summary.json", "w") as f:
            json.dump(summary_results, f, indent=4)
