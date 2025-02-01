from QuantumRingsLib import QuantumCircuit, QuantumRegister, ClassicalRegister, job_monitor
import numpy as np
import math
import QR_secrets
import semiprimes

shots = 1024

def iqft_cct(qc, b, n):
    for i in range (n):
        for j in range (1, i+1):
            qc.cu1(-math.pi/2**(i-j+1), b[j-1], b[i])
        qc.h(b[i])
    qc.barrier()
    return

def shors_algorithm(N):
    if N % 2 == 0:
        return 2
    
    a = np.random.randint(2, N)

    if math.gcd(a, N) != 1:
        return math.gcd(a, N)
    
    n = int(np.ceil(np.log2(N)))
    qr = QuantumCircuit(2*n, 2*n)

    for qubit in range(n):
        qr.h(qubit)

    for qubit in range(n):
        qr.cu1(2*np.pi*a**(2**qubit)/N, qubit, n) # TODO : Find a way to remove that a**(2**qubit)?

    iqft_cct(qr, range(n), n)

    qr.measure(range(n), range(n))

    qc = qr

    backend = QR_secrets.backend
    job = backend.run(qc, shots=shots)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts()

    measured_value = int(list(counts.keys())[0], 2)

    r = 1
    while (a**r)%N != 1:
        r += 1

    if r % 2 == 0 and (a**(r//2)+1)%N != 0:
        factor1 = math.gcd(a**(r//2)-1, N)
        factor2 = math.gcd(a**(r//2)+1, N)
        return factor1, factor2

    return None

if __name__=="__main__":
    semiprimes=semiprimes.semiprimes
    for N in list(semiprimes.values())[:2]:
        print(f"Semiprime {N} is factored into {shors_algorithm(N)}")