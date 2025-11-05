'''

Entropy Check... yuh
'''

#%%
import random
import quimb.tensor as qtn
import numpy as np
from tqdm import tqdm
from qiskit.quantum_info import random_clifford

def make_sim(dim, simmean, simwidth):
    
    RR = np.random.normal(simmean, simwidth, (dim,dim))
    SM = RR + 1j * np.random.normal(simmean, simwidth, (dim,dim))
    return SM

def make_unitary(dim, simmean, simwidth):
    
    sim = make_sim(dim, simmean, simwidth)
    Q, R = np.linalg.qr(sim)
    Etta = np.zeros((dim,dim))
    for j in range(dim):
        Etta[j,j] = R[j,j]/np.linalg.norm(R[j,j])
    U = np.matmul(Q, Etta)
    return U

def dephase(unitary):
    
    glob = np.linalg.det(unitary)
    theta = np.arctan(np.imag(glob) / np.real(glob)) / 2
    unitary = unitary * np.exp(-1j*theta)
    if np.round(np.linalg.det(unitary)) < 0:
        unitary = unitary * 1j
    return unitary

def PPgate():

    u1 = make_unitary(2, 0, 1)
    u2 = make_unitary(2, 0, 1)
    u1 = dephase(u1)
    u2 = dephase(u2)
    G_AB = np.array([[u1[0,0], 0, 0, u1[0,1]],
                      [0, u2[0,0], u2[0,1], 0],
                      [0, u2[1,0], u2[1,1], 0],
                      [u1[1, 0], 0, 0, u1[1,1]]])
    return G_AB


def gate_funct(gt):
    
    if gt== 'Haar':
        gate = make_unitary(4, 0, 1)
    if gt == 'Match':
        gate = PPgate()
    if gt == 'Clifford':
        gate = random_clifford(2).to_matrix() 
    return gate

#%%
gat = 'Match'
N = 12

its = 100
entsM = [0]*20
for g in tqdm(range(its)):
    mps = qtn.MPS_computational_state('0'*N)
    for i in range(N):
        mps.gate(make_unitary(2,0,1), i, inplace=True,contract=True)
    for l in range(N**2):
        for p in np.arange(0,N-1,2):
            gate = gate_funct(gat)
            mps.gate_split(gate,(p, p+1),inplace=True)
        for p in np.arange(1,N-1,2):
            gate = gate_funct(gat)
            mps.gate_split(gate,(p, p+1),inplace=True)
    
    for k in range(20):
        entsM[k] += (sum( np.array(mps.schmidt_values(N//2)**k) ))
            

gat = 'Haar'
entsH = [0]*20
for g in tqdm(range(its)):
    mps = qtn.MPS_computational_state('0'*N)
    for i in range(N):
        mps.gate(make_unitary(2,0,1), i, inplace=True,contract=True)
    for l in range(N**2):
        for p in np.arange(0,N-1,2):
            gate = gate_funct(gat)
            mps.gate_split(gate,(p, p+1),inplace=True)
        for p in np.arange(1,N-1,2):
            gate = gate_funct(gat)
            mps.gate_split(gate,(p, p+1),inplace=True)
    
    for k in range(20):
        entsH[k] += (sum( np.array(mps.schmidt_values(N//2)**k) ))
            
gat = 'Clifford'
entsC = [0]*20
for g in tqdm(range(its)):
    mps = qtn.MPS_computational_state('0'*N)
    for l in range(N**2):
        for p in np.arange(0,N-1,2):
            gate = gate_funct(gat)
            mps.gate_split(gate,(p, p+1),inplace=True)
        for p in np.arange(1,N-1,2):
            gate = gate_funct(gat)
            mps.gate_split(gate,(p, p+1),inplace=True)
    
    for k in range(20):
        entsC[k] += (sum( np.array(mps.schmidt_values(N//2)**k) ))


entsHy = [0]*20
for g in tqdm(range(its)):
    mps = qtn.MPS_computational_state('0'*N)
    for i in range(N):
        mps.gate(make_unitary(2,0,1), i, inplace=True,contract=True)
    for l in range(N**2):
        for p in np.arange(0,N-1,2):
            gate = gate_funct('Match')
            mps.gate_split(gate,(p, p+1),inplace=True)
        for p in np.arange(1,N-1,2):
            gate = gate_funct('Match')
            mps.gate_split(gate,(p, p+1),inplace=True)
    for l in range(N**2):
        for p in np.arange(0,N-1,2):
            gate = gate_funct('Clifford')
            mps.gate_split(gate,(p, p+1),inplace=True)
        for p in np.arange(1,N-1,2):
            gate = gate_funct('Clifford')
            mps.gate_split(gate,(p, p+1),inplace=True)
    
    for k in range(20):
        entsHy[k] += (sum( np.array(mps.schmidt_values(N//2)**k) ))


for k in range(20):
    entsM[k] = entsM[k] / its
    entsH[k] = entsH[k] / its
    entsC[k] = entsC[k] / its
    entsHy[k] = entsHy[k] / its
    
#%%
import matplotlib.pyplot as plt
plt.plot(np.arange(0,20,1), entsM, label='Matchgate')
plt.plot(np.arange(0,20,1), entsC, label='Clifford')
plt.plot(np.arange(0,20,1), entsH, '--', label='Haar')
plt.plot(np.arange(0,20,1), entsHy, alpha=0.7, label='Matchgate+Clifford')
plt.yscale('log')
plt.legend()
plt.title('kth order sum(p_i^k)')
plt.show()