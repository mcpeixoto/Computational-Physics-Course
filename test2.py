import numpy as np
from numpy.random import rand
from copy import copy

def maiornadia(a,b):
    aa=None
    bb=None
    t=copy(a.T)

    for i in range(len(a)):
        x=np.max(np.abs(t[i][i:]).tolist())
        y=t[i].tolist().index(x)

        # Eliminar os elementos 
        for idx_line in range(len(t)):
            t[idx_line][y] = 0

        if aa is None:
            aa = a[y].reshape(1,-1)
            bb = b[y].reshape(1,-1)
        else:
            aa=np.append(aa,a[y].reshape(1,-1),axis=0)
            bb=np.append(bb,b[y].reshape(1,-1),axis=0)
    
    return (aa,bb)


A=np.array([
[1,6,3,5],
[3,5,4,5],
[4,5,6,6],
[1,2,3,4]
],dtype=np.float64)

B=np.array([[2],[6],[3],[9]],dtype=np.float64).reshape(-1)




def biggest_number_on_diagonal(matrix, results):
    original_matrix = copy(matrix)
    new_matrix = np.ones(matrix.shape)
    new_results = np.ones(results.shape)

    # Lines still available on new_matrix
    lines_avail = [i for i in range(matrix.shape[0])]
    # Lines idx of the original matrix already used 
    idx_used = []

    while lines_avail != []:
        # Compute the indexes of the biggest numbers for each column
        # of the original matrix
        biggest_numbers_idxs = [list(np.argwhere(line == np.amax(line)).reshape(-1)) for line in abs(matrix.T)]

        # For each computed idxs of each column
        for i, idxs in enumerate(biggest_numbers_idxs):
            # Idxs of the biggest numbers of the column if they
            # weren't still used
            idx = [idx for idx in idxs if idx not in idx_used]
            if idx != [] and i in lines_avail:
                # Let's select the first one
                idx = idx[0]


                new_matrix[i] = matrix[idx]
                new_results[i] = results[idx]

                idx_used.append(idx)
                lines_avail.remove(i)

                # Reset line so that
                # it dosen't get picked 
                # again in the future
                matrix[idx] *= 0
    
    print("Original Matrix:\n", original_matrix, results)
    print("New Matrix:\n", new_matrix, new_results)

    return new_matrix, new_results

from itertools import permutations
def biggest_number_on_diagonal(matrix, results):
    every_permutation = permutations([i for i in range(matrix.shape[0])])

    best_trace = 0
    best_permutation = None

    for permutation in every_permutation:
        new_matrix = np.ones(matrix.shape)

        for i,idx in enumerate(permutation):
            new_matrix[i] = matrix[idx]

        trace = np.trace(np.abs(new_matrix))
        if trace > best_trace:
            best_trace = trace
            best_permutation = permutation

    # Now that we have the best permutation possible 
    new_matrix = np.ones(matrix.shape)
    new_results = np.ones(results.shape)
    for i,idx in enumerate(best_permutation):
        new_matrix[i] = matrix[idx]
        new_results[i] = results[idx]

    print("Original Matrix:\n", matrix, results)
    print("New Matrix:\n", new_matrix, new_results)

    return new_matrix, new_results


dim = 6

print(biggest_number_on_diagonal(A, B))