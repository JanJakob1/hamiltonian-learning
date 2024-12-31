import itertools
import pickle

from tqdm import tqdm
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt

from src.utils import integer_to_binary
from src.dataset import Dataset, ExactSimulationDatasetGenerator
from src.models import ExactModel, ODEModel, NeuralODEModel
from src.hamiltonian import HeteroHeisenbergHamiltonian
from src.loop import NLLLoop, NLLWeightDecayLoop


from jax import config
config.update("jax_enable_x64", True)


def run_robustness_experiment():
    n, num_states, num_times, num_paulis, shots = 6, 5, 5, 200, 100
    H = HeteroHeisenbergHamiltonian(n)

    print(f'Hamiltonian: {H.__class__.__name__}, Parameters {H.num_parameters}, Pauli ops {H.num_observables}')

    #for trial in range(50):
    trial = 1
    print(f'Running trial {trial}...')
    
    model = NeuralODEModel(H, key=jax.random.key(trial + 100000))

    # for this simple example just use a fixed learning rate instead of the curriculum learning scheme
    schedule = 0.01

    optimizer = optax.adam(learning_rate=schedule)

    loop = NLLWeightDecayLoop(model, optimizer, l2=1e-3)
    loop.load_model('./runs/node_HeteroHeisenbergHamiltonian_6_049.eqx')
    loop.load_metrics('./runs/node_HeteroHeisenbergHamiltonian_6_049.pkl')
    #print(loop.metrics)
    loop.plot_loss()
    loop.plot_params()
    print(loop.l1_error())
      
if __name__ == '__main__':
    run_robustness_experiment()

