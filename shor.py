from QuantumRingsLib import QuantumCircuit, QuantumRegister, ClassicalRegister, job_monitor
import numpy as np
import math
import QR_secrets
import semiprimes
import time
import json
import matplotlib.pyplot as plt

from fractions import Fraction


shots = 1024


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

    if math.gcd(a, N) != 1:
        print("GCD is not 1, returning GCD")
        return math.gcd(a, N)
    
    n = int(np.ceil(np.log2(N)))
    qr = QuantumCircuit(2*n, 2*n)

    print(f'Looking for N={N} with a={a}, n={n} (total qubits: {2*n})')

    for qubit in range(n):
        qr.h(qubit)

    qr.barrier()

    for qubit in range(n):
        qr.cu1(2*np.pi*a**(2**qubit)/N, qubit, n) # TODO : Find a way to remove that a**(2**qubit)?

    
    # mod_exp = 1  # a^(2^0) mod N
    # for qubit in range(n):
    #     phase = (2 * np.pi * mod_exp) / N
    #     qr.p(phase, n)  # Replace cu1 with equivalent phase gate
    #     mod_exp = modular_exponentiation(a, 2**qubit, N)  # Compute a^(2^qubit) mod N

    qr.barrier()
    iqft_cct(qr, range(n), n)

    qr.barrier()

    qr.measure(range(n), range(n))

    # capture stdout to save qr circuit when calling qr.draw
    import sys

    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    qr.draw()

    # save to file
    with open(f'shor_circuit_{N}.txt', 'w') as f:
        f.write(sys.stdout.getvalue())

    sys.stdout = old_stdout


    qc = qr

    backend = QR_secrets.backend
    job = backend.run(qc, shots=shots)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts()
    
    r = extract_period_from_counts(counts, a, N)

    # measured_value = int(list(counts.keys())[0], 2)

    # r = 1
    # while (a**r)%N != 1:
    #     r += 1
    #     # exit condition to prevent infinite loop
    #     if r > 10000:
    #         print("Warning: Exiting to prevent infinite loop")
    #         print(a, N)
    #         break

    if r % 2 == 0 and (a**(r//2)+1)%N != 0:
        factor1 = math.gcd(a**(r//2)-1, N)
        factor2 = math.gcd(a**(r//2)+1, N)
        return factor1, factor2

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
