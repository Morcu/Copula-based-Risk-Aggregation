import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from qiskit import QuantumCircuit, execute
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit_algorithms.optimizers import SPSA
from qiskit.visualization import plot_histogram

# Function to create a parameterized quantum circuit for 'n' variables with 'm' qubits each
def create_quantum_circuit(n, m, layer):
    """
    Creates a parameterized quantum circuit.

    Parameters:
    - n (int): Number of variables (assets).
    - m (int): Number of qubits per variable.
    - layer (int): Number of layers of parameterized gates.

    Returns:
    - QuantumCircuit: The parameterized quantum circuit.
    - ParameterVector: The parameters used in the circuit.
    """
    num_qubits = n * m # Total number of qubits
    qc = QuantumCircuit(num_qubits, num_qubits) # Create quantum circuit with required number of qubits
    qc.h(range(num_qubits))  # Add a layer of Hadamard gates to create superposition
    
    # Interlace qubits corresponding between assets
    for asset1 in range(n):
        for asset2 in range(asset1 + 1, n): # Ensure no pair repeats and no asset compares with itself
            for qubit in range(m):
                qubit1 = asset1 * m + qubit
                qubit2 = asset2 * m + qubit
                qc.cx(qubit1, qubit2) # Apply CNOT with Hadamard qubit as control and corresponding in other asset as target

    # Add parameterized qubit rotations            
    params = ParameterVector('θ', length=n * (3 * m - 1) * layer)
    param_index = 0
    for _ in range(layer):
        for qubit in range(num_qubits):
            qc.rz(params[param_index], qubit)
            param_index += 1
            qc.rx(params[param_index], qubit)
            param_index += 1
        for asset in range(n): # Iterate over each asset for RZZ between consecutive qubits
            start_index = asset * m
            for qubit_index in range(start_index, start_index + m - 1):
                qc.rzz(params[param_index], qubit_index, qubit_index + 1)
                param_index += 1
    
    qc.measure(range(num_qubits), range(num_qubits)) # Add measurements at the end of the circuit
    return qc, params

# Function to digitize real values into binary strings based on the number of qubits 'm'
def real_to_binary(samples, m):
    """
    Converts real samples into binary strings based on 'm' qubits.

    Parameters:
    - samples (array): 2D array where each row is a pseudo-sample and each column is a variable.
    - m (int): Number of qubits per variable.

    Returns:
    - list: List of binary strings representing the samples.
    """
    binary_samples = []
    for sample in samples:
        binary_sample = ''
        for value in sample:
            scaled_value = int(value * (2**m)) # Scale value to [0, 2^m)
            binary_sample += format(scaled_value, '0{}b'.format(m))  # Convert to binary string of length 'm'
        binary_samples.append(binary_sample)
    return binary_samples

# Function to calculate the KL divergence
def kl_divergence(p, q, epsilon=1e-8):
    """
    Calculates the KL divergence between two distributions.

    Parameters:
    - p (array): Target distribution.
    - q (array): Measured distribution.
    - epsilon (float): Small number to avoid division by zero.

    Returns:
    - float: The KL divergence.
    """
    q = np.clip(q, epsilon, 1 - epsilon) # Ensure 'q' has no zeros
    # Avoid computation where p is zero
    mask = p > 0
    kl_div = np.sum(p[mask] * np.log(p[mask] / q[mask]))    
    return kl_div

# Cost function for optimization
def cost_function(params, circuit, target_distribution, m, n, backend):
    """
    Cost function for optimization, calculates KL divergence between target and measured distributions.

    Parameters:
    - params (array): Parameters for the quantum circuit.
    - circuit (QuantumCircuit): The quantum circuit.
    - target_distribution (array): Target probability distribution.
    - m (int): Number of qubits per variable.
    - n (int): Number of variables (assets).
    - backend (Backend): Simulation backend.

    Returns:
    - float: The cost (KL divergence).
    """
    parameterized_circuit = circuit.bind_parameters(params) # Assign parameters to the circuit
    job = execute(parameterized_circuit, backend, shots=1024)  # Execute the circuit
    result = job.result()
    counts = result.get_counts() # Get probability distribution of measured states
    measured_distribution = np.zeros(2**(m*n))
    for state, count in counts.items():
        measured_distribution[int(state, 2)] = count
    measured_distribution /= measured_distribution.sum()
    kl_div = kl_divergence(target_distribution, measured_distribution) # Calculate KL divergence
    return kl_div

def train_model(circuit, parameters, backend, target_distribution, m, n):
    """
    Executes the optimization process.

    Parameters:
    - circuit (QuantumCircuit): The quantum circuit.
    - parameters (ParameterVector): Parameters of the circuit.
    - backend (Backend): Simulation backend.
    - target_distribution (array): Target probability distribution.
    - m (int): Number of qubits per variable.
    - n (int): Number of variables (assets).

    Returns:
    - array: Optimal parameters.
    - float: Final cost.
    - list: Cost values over iterations.
    """
    cost_values = []
    def objective_function(params):
        cost = cost_function(params, circuit, target_distribution, m, n, backend)
        cost_values.append(cost)
        return cost
    initial_params = np.random.rand(parameters._size) * 2 * np.pi
    optimizer = SPSA(maxiter=100)
    result = optimizer.minimize(objective_function, initial_params)
    return result.x, result.fun, cost_values

def main(n, m, layer, data_file):
    data_samples = np.loadtxt(data_file, delimiter=",")
    binary_data_samples = real_to_binary(data_samples, m)
    target_distribution = np.zeros(2**(m*n))
    for binary_sample in binary_data_samples:
        target_distribution[int(binary_sample, 2)] += 1
    target_distribution /= target_distribution.sum()

    backend = AerSimulator()
    qc, parameters = create_quantum_circuit(n, m, layer)
    optimal_params, final_cost, cost_values = train_model(qc, parameters, backend, target_distribution, m, n)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.plot(cost_values)
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function Value')
    plt.title('Cost Function over Iterations')
    plt.savefig(f'cost_function_{timestamp}.png')
    plt.close()

    qc_optimized = qc.bind_parameters(optimal_params)
    job = execute(qc_optimized, backend, shots=1024)
    result = job.result()
    counts = result.get_counts(qc_optimized)
    plot_histogram(counts)
    plt.savefig(f'sample_distribution_{timestamp}.png')

    # Print optimization results and qc
    print(qc.draw())
    print("Optimal parameters found:", optimal_params)
    print("Final cost (KL Divergence):", final_cost)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum Optimization Script')
    parser.add_argument('--n', type=int, required=True, help='Number of variables')
    parser.add_argument('--m', type=int, required=True, help='Number of qubits per variable')
    parser.add_argument('--layers', type=int, required=True, help='Number of layers')
    parser.add_argument('--file', type=str, required=True, help='Data file path')

    args = parser.parse_args()

    main(args.n, args.m, args.layers, args.file)
