# Copyright 2019 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# THIS FILE HAS BEEN MODIFIED FROM ITS ORIGINAL (mutual-information-feature-selection)

# Mostly from D-Wave (I added some libraries)------------------------------
import itertools
import os
import matplotlib
matplotlib.use("agg")    # must select backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from mlxtend.plotting import checkerboard_plot # pip install mlxtend 
import seaborn # pip install seaborn
import math

# D-Wave Ocean tools
import dimod
from dwave.embedding.chimera import find_clique_embedding
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dimod.reference.samplers import ExactSolver
import neal
# ----------------------------------------------------------------------

def run_demo():
    # Declare a binary quadratic model
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    
    # Mostly me -----------------------------------------------------------
    n = 6 # Size of grid is n x n
    qubits = list(range(1,n**2+1))
    qubits = [str(qubit) for qubit in qubits] # Name the qubits
    
    # Create "matrices" for rows, columns, and diagonals
    rows = list(range(1,n+1))
    rows = np.repeat(rows, n)

    columns = list(range(1,n+1))
    columns = np.tile(columns, n)

    fw_diags = list(range(1,n**2+1))
    for i in list(range(0,n**2)):
        fw_diags[i] = list(range(1,n+1))[(i)%n] + math.floor(i/n) 

    bw_diags = list(range(1,n**2+1))
    for i in list(reversed(range(0,n**2))):
        bw_diags[i] = list(range(1,n+1))[i%n] + (n - math.floor(i/n) -1)

    # Define penalties and rewards
    a_i = -2
    row_penalty = 2
    diag_penalty = 1
    other_penalty = 0
    
    # Whenever a qubit is activated, add this reward
    for qubit in qubits:
        bqm.add_variable(qubit, a_i)

    # If two qubits are in the same row, column, or diagonal
    # Add the appropriate penalty
    for q1, q2 in itertools.combinations(qubits, 2):
        q1_num = int(q1)
        q2_num = int(q2)
        row1 = rows[q1_num-1]
        row2 = rows[q2_num-1]
        col1 = columns[q1_num-1]
        col2 = columns[q2_num-1]
        fwd1 = fw_diags[q1_num-1]
        fwd2 = fw_diags[q2_num-1]
        bwd1 = bw_diags[q1_num-1]
        bwd2 = bw_diags[q2_num-1]

        if row1==row2:
            bqm.add_interaction(q1, q2, row_penalty)
            bqm.add_interaction(q2, q1, row_penalty)
        elif col1==col2:
            bqm.add_interaction(q1, q2, row_penalty)
            bqm.add_interaction(q2, q1, row_penalty)
        elif fwd1==fwd2:
            bqm.add_interaction(q1, q2, diag_penalty)
            bqm.add_interaction(q2, q1, diag_penalty)
        elif bwd1==bwd2:
            bqm.add_interaction(q1, q2, diag_penalty)
            bqm.add_interaction(q2, q1, diag_penalty)
        else:
            bqm.add_interaction(q1, q2, other_penalty)
            bqm.add_interaction(q2, q1, other_penalty)

    # ---------------------------------------------------------------
            
    # Mostly from D-Wave and modified --------------------------------------------
    # These are mostly settings that shouldn't really be tampered with
    bqm.normalize()  # Normalize biases & interactions to the range -1, 1

    # Sets up a CPU solver that goes through all 2^(n^2) configurations
    # Can become time consuming!
    #mys = ExactSolver()
    #ans = mys.sample(bqm)


    # Set up a QPU sampler with a fully-connected graph of all the variables
    qpu_sampler = DWaveSampler(solver={'qpu': True})
    embedding = find_clique_embedding(bqm.variables,
                                      16, 16, 4,  # size of the chimera lattice
                                      target_edges=qpu_sampler.edgelist)
    sampler = FixedEmbeddingComposite(qpu_sampler, embedding)
    sampleset = sampler.sample(bqm, num_reads=1000)


    # Sets up a CPU sampler (using simulated annealing)
    #mys = neal.SimulatedAnnealingSampler()
    #ans = mys.sample(bqm, num_reads=1000)
    # ------------------------------------------------------------------
    
    # A combination of my code / from D-Wave example -------------------
    # I made the checkerboard, using a heatmap and an array
    # Display results
    print(sampleset.first.sample)

    ary = np.ones((1,n**2))
    for qubit in qubits:
        ary[0][int(qubit)-1] = ary[0][int(qubit)-1] - sampleset.first.sample[qubit]
    ary = np.reshape(ary, (n,n))

    brd = seaborn.heatmap(ary, cbar=0, square=1, linewidths=1, 
                          xticklabels=[str(num) for num in list(range(1,n+1))],
                          yticklabels=[str(num) for num in list(range(1,n+1))])
    plt.show()
    plt.savefig('NQueensSolution.png')


if __name__ == "__main__":
    run_demo()