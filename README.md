# Quantum Rings Remote Challenge - iQuHACK 2025

## Team Members
- Florent Pollet (flo1.raymond@gmail.com)
- Pierre Sibut-Bourde (pierre.sibut.bourde@gmail.com)
- Louis-Justin Tallot (lj.tallot@gmail.com)

## Introduction
This repository contains the code and the report for the Quantum Rings Remote Challenge of the iQuHACK 2025. The goal of this challenge is to factorize a number using Shor's algorithm. We implemented the algorithm using QuantumRingsLib, which was run on the pierre.sibut.bourde@gmail.com account.

As not to disclose personal information regarding the accound, we used a gitignored file called 'QR_secrets.py' to store the token and the email of the account. The file is structured as follows:
```python
from QuantumRingsLib import QuantumRingsProvider

provider = QuantumRingsProvider(
    token='XXX',
    name='pierre.sibut.bourde@gmail.com'
)
backend = provider.get_backend("scarlet_quantum_rings")
```

## Algorithm used
We used Shor's algorithm to factorize the number. The algorithm is implemented in the `shor.py` file. The algorithm is as follows:
1. Choose a random number `a` between 2 and `N-1`.
2. Compute the greatest common divisor of `a` and `N`. If it is not 1, then we have found a factor of `N`.
3. Otherwise, compute the period `r` of the function `f(x) = a^x mod N`.
4. If `r` is odd, go back to step 1.
5. If `r` is even, compute `p = gcd(a^(r/2) - 1, N)` and `q = gcd(a^(r/2) + 1, N)`. If `p` and `q` are not 1, then we have found the factors of `N`.

## Circuit used
The circuit used is the one provided in the `shor.py` file. It is a simple implementation of the algorithm using QuantumRingsLib.

We used an implementation following :
- https://arxiv.org/abs/quant-ph/0205095
- https://docs.quantum.ibm.com/api/qiskit/0.25/qiskit.algorithms.Shor

The circuit is built with :
- An auxiliary register of `n+2` qubits, initialized to the state `|1>`.
- An up register of size `2n` where quantum Fourier transform is applied.
- A down register of size `n` where the function `f(x) = a^x mod N` is applied.
- A classical register of size `2n` where the measured values are stored.

Then we do : 
- Apply Hadamard gates to the up register to have a full superposition.
- Initialize the down register to the state `|1>`
- Apply the multiplication gates using the function `cMULTmodN` (defined in the `helper_norm.py` file)
- Apply the inverse quantum Fourier transform to the up register.
- Measure the top qubits.

## Results
We ran the algorithm by iterating on the `semiprimes.py` file. We found the factors of the numbers in the file. The results are stored in the `shor_summary.json` file and are as follows.

| Number | Factors | Qubits used | Gate operations | Execution time |
|--------|---------|-------------|-----------------|----------------|
|  6     | 2,3     | N/A         | N/A             | 0s             |
|  15    | 3,5     | 26          | 2400            | 1.12s          |
|  21    | 3,7     | 32          | 4140            | 1.18s          |
|  143   | 11,13   | 50          | 13632           | 96.5s          |
|  899   | XX,XX   | 62          | 24480           | Xs             |

## Conclusion
We successfully implemented Shor's algorithm using QuantumRingsLib. We were able to factorize the numbers in the `semiprimes.py` file. The algorithm was able to factorize numbers up to 143 in a reasonable amount of time. However, the algorithm was not able to factorize the number 899 in a reasonable amount of time, as the algorithm is not efficient enough to factorize it. 

We can definitely improve our implementation using circuit optimization, as well as finding a better way to implement Quantum Fourier Transform.

## References used during the Hackathon
### ArXiV papers
- https://arxiv.org/pdf/2306.09122
- https://arxiv.org/pdf/quant-ph/0408006
- https://arxiv.org/pdf/1411.6758
- https://arxiv.org/abs/2405.17021v2
- https://arxiv.org/pdf/2306.09122
- https://arxiv.org/pdf/quant-ph/0205095

### GitHub repositories
- https://github.com/mett29/Shor-s-Algorithm/blob/master/Shor.ipynb
- https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/shor.ipynb
- https://github.com/tiagomsleao/ShorAlgQiskit/blob/master/Shor_Normal_QFT.py
- https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/shor.ipynb
- https://github.com/tiagomsleao/ShorAlgQiskit

### Other
- https://www.science.org/doi/10.1126/science.aad9480
- https://docs.quantum.ibm.com/api/qiskit/0.25/qiskit.algorithms.Shor
