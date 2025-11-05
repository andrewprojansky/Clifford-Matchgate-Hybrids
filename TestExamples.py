#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:31:28 2024

@author: andrewprojansky

TextExamples.py: Code that contains test examples for four cases: Matchgates,
Cliffords following MGs, MG+, and Cliffords following MG+. In each of them, 
1000 random Pauli expectation values are tested, and are shown to match 
expectation values calculated from dense statevector simulation. 
"""

from constant_import import *
from CliffordMGFinal import *

#%%
'''
Expectation values for MG circuits. To get expectation value for specific 
Pauli, or single random pauli, uncomment/comment out lines as needed
'''

N = 10
pe, po = Pair_maker(N)
R = np.identity(2*N)
psi = qtn.MPS_computational_state('0'*N)
for j in range(10):
    psi, R = ExEvo(N,R,pe, ext=False, psi=psi)
    psi, R = ExEvo(N,R,po, ext=False, psi=psi)

cov = make_cov_0(N)
Cov_mat = R @ cov @ R.T
psi = psi.to_dense()

l = JW_Ms(N)

'''
#P = Pauli('ZIIIIIIIII') 
P = random_pauli(N); gindsP = op_indices(N,P,l)
exp_val = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)
'''

for j in range(1000):
    P = random_pauli(N); gindsP = op_indices(N,P,l)
    v1 = (np.conj(psi).T @ P.to_matrix() @ psi)
    v2 = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)
    
    if np.round(v1,5) != np.round(v2,5):
        print(P)
        print(v1)
        print(v2)
        
#%%
'''
Expectation values for MG circuits, same as above but with no dense check
to verify testing works, and showing that we can go to 100+ qubits
'''
from tqdm import tqdm

bdict = {}
bdictc = {}

for k in range(25):

    N = 50
    pe, po = Pair_maker(N)
    R = np.identity(2*N)
    for j in tqdm(range(N)):
        R = ExEvo(N,R,pe, ext=False,psi=None)
        R = ExEvo(N,R,po, ext=False,psi=None)
    
    cov = make_cov_0(N)
    Cov_mat = R @ cov @ R.T
    
    l = JW_Ms(N)
    
    '''
    #P = Pauli('ZIIIIIIIII') 
    P = random_pauli(N); gindsP = op_indices(N,P,l)
    exp_val = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)
    '''
    
    nops = 10000
    for j in tqdm(range(nops)):
        ginds = np.arange(0,100,1)
        ginds = np.random.permutation(ginds)
        gindsP = ginds[0:np.random.randint(1,50)*2]
        v2 = (get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N))
        lops = len(gindsP) 
        if lops in bdict.keys():
            bdict[lops] = bdict[lops] + (v2)
            bdictc[lops] = bdictc[lops] + 1
        else:
            if lops%2 == 0:
                bdict[lops] = (v2)
                bdictc[lops] = 1
            
for ks in bdict.keys():
    bdict[ks] = bdict[ks]/bdictc[ks]
    
import matplotlib.pyplot as plt
#%%
bdictcp = bdict
xvals = np.arange(2,100,2)
yvals = [bdict[x] for x in xvals]
plt.plot(xvals, yvals)
#plt.yscale('log')
plt.title('Expectation value of operators based on majorana weight')
plt.xlabel('Majorana weight')
plt.ylabel('(expectation value no absolute)')
plt.show()


'''
nops = 20000
bdict = {}
bdictc = {}
for j in tqdm(range(nops)):
    P = random_pauli(N); gindsP = op_indices(N,P,l)
    v2 = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)
    lops = len(gindsP) 
    if lops in bdict.keys():
        bdict[lops] = bdict[lops] + np.abs(v2)
        bdictc[lops] = bdictc[lops] + 1
    else:
        if lops%2 == 0:
            bdict[lops] = np.abs(v2)
            bdictc[lops] = 1
            
for ks in bdict.keys():
    bdict[ks] = bdict[ks]/bdictc[ks]
'''
#%%
'''
Expectation values for MG circuits, followed by Cliffords. To get expectation 
value for specific Pauli, or single random pauli, uncomment/comment out lines 
as needed. Clifford is defined from Qiskit, and is random. This shows
how to translate from desired Pauli on conjugated encoding back to what the 
Pauli would be in JW 
'''

N = 10
pe, po = Pair_maker(N)
R = np.identity(2*N)
psi = qtn.MPS_computational_state('0'*N)
for j in range(10):
    psi, R = ExEvo(N,R,pe, ext=False, psi=psi)
    psi, R = ExEvo(N,R,po, ext=False, psi=psi)

cov = make_cov_0(N)
Cov_mat = R @ cov @ R.T

C = random_clifford(N)
Ct = C.adjoint()

psi = psi.to_dense()
psi = C.to_matrix() @ psi

l = JW_Ms(N)

'''
#P = Pauli('ZIIIIIIIII') 
P = random_pauli(N)
Pnew = P.evolve(C)
if Pnew.to_label()[0] == '-':
    pphase=-1
    Pnew = Pnew*-1
else:
    pphase=1  
gindsP = op_indices(N,Pnew,l)
exp_val = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)
'''

for j in range(1000):
    P = random_pauli(N)
    Pnew = P.evolve(C)
    if Pnew.to_label()[0] == '-':
        pphase=-1
        Pnew = Pnew*-1
    else:
        pphase=1  
    gindsP = op_indices(N,Pnew,l)
        
    v1 = (np.conj(psi).T @ P.to_matrix() @ psi)
    v2 = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)*pphase
    
    if np.round(v1,0) != np.round(v2,0):
        print(P)
        print(v1)
        print(v2)
#%%
'''
Expectation values for MG circuits. To get expectation value for specific 
Pauli, or single random pauli, uncomment/comment out lines as needed

Slightly odd that this works honestly, because this relies on Wick's theorem, 
which I thought we'd loose when we extend to the larger space. Need to review 
a bit of theory here... isn't too  surprising because this is eqiuvalent to
subalgebra over larger space, but still need to review
'''

N = 10
pe, po = Pair_maker(N)
R = np.identity(2*N+1)
psi = qtn.MPS_computational_state('0'*N)

psi, R = RProdfromCov(N, R, psi)

for j in range(10):
    psi, R = ExEvo(N,R,pe, ext=True, psi=psi)
    psi, R = ExEvo(N,R,po, ext=True, psi=psi)
    
cov = make_cov_0(N, Ext=True)
Cov_mat = R @ cov @ R.T
psi = psi.to_dense()
l = JW_Ms(N)

'''
#P = Pauli('ZIIIIIIIII') 
P = random_pauli(N); gindsP = op_indices(N,P,l)
exp_val = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)
'''

for j in range(1000):
    P= random_pauli(N)
    gindsP = op_indices(N,P,l,Ext=True)
    v1 = (np.conj(psi).T @ P.to_matrix() @ psi)
    v2 = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N, Ext=True)
    
    if np.round(v1,5) != np.round(v2,5):
        print(P)
        print(gindsP)
        print(Multiply_forPhase2(gindsP, N, Ext=True))
        print(v1)
        print(v2)

#%%
'''
Expectation values for MG+ circuits, followed by Cliffords. To get expectation 
value for specific Pauli, or single random pauli, uncomment/comment out lines 
as needed. Clifford is defined from Qiskit, and is random. This shows
how to translate from desired Pauli on conjugated encoding back to what the 
Pauli would be in JW 
'''

N = 10
pe, po = Pair_maker(N)
R = np.identity(2*N+1)
psi = qtn.MPS_computational_state('0'*N)

psi, R = RProdfromCov(N, R, psi)

for j in range(10):
    psi, R = ExEvo(N,R,pe, ext=True, psi=psi)
    psi, R = ExEvo(N,R,po, ext=True, psi=psi)
    
cov = make_cov_0(N, Ext=True)
Cov_mat = R @ cov @ R.T
C = random_clifford(N)
Ct = C.adjoint()
psi = psi.to_dense()
psi = C.to_matrix() @ psi
l = JW_Ms(N)

'''
#P = Pauli('ZIIIIIIIII') 
P = random_pauli(N)
Pnew = P.evolve(C)
if Pnew.to_label()[0] == '-':
    pphase=-1
    Pnew = Pnew*-1
else:
    pphase=1  
gindsP = op_indices(N,Pnew,l)
exp_val = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)
'''

for j in range(1000):
    P = random_pauli(N)
    Pnew = P.evolve(C)
    if Pnew.to_label()[0] == '-':
        pphase=-1
        Pnew = Pnew*-1
    else:
        pphase=1  
    gindsP = op_indices(N,Pnew,l,Ext=True)
    v1 = (np.conj(psi).T @ P.to_matrix() @ psi)
    v2 = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N, Ext=True)*pphase
    
    if np.round(v1,0) != np.round(v2,0):
        print(P)
        print(gindsP)
        print(Multiply_forPhase2(gindsP, N, Ext=True))
        print(v1)
        print(v2)
    if np.abs(np.real(v1)) < 0.0000001:
        print(np.real(v1))
#%%
v = np.zeros(8)
v[0] = 1
vp = np.kron(H, np.kron(H,H)) @ v
vp = np.kron(CZ, np.identity(2)) @ np.kron(np.identity(2), CZ) @ vp
vp = np.kron(np.identity(2), GHs) @ np.kron(GHs, np.identity(2)) @ vp
vpf = np.kron(vp,vp)
vpf = np.kron(np.identity(4), np.kron(GH, np.identity(4))) @ vpf
vpf = np.kron(np.identity(8), np.kron(fSWAP, np.identity(2))) @ vpf
vpf = np.kron(np.identity(16), fSWAP) @ vpf
vpf = np.kron(np.identity(4), np.kron(fSWAP, np.identity(4))) @ vpf
vpf = np.kron(np.identity(8), np.kron(fSWAP, np.identity(2))) @ vpf
#%%
'''
Expectation values for MG circuits, same as above but with no dense check
to verify testing works, and showing that we can go to 100+ qubits
'''
from tqdm import tqdm


#wl = [2,12,22,32,42,52,62,72,82,92,102,112,122,132,142,152,162,172,182,192]
wl = [2,12,22,32,42,52,62,72,82,92]
tl = [[] for x in range(len(wl))]

for i in range(20):
    N = 50
    pe, po = Pair_maker(N)
    R = np.identity(2*N)
    for j in tqdm(range(N)):
        R = ExEvo(N,R,pe, ext=False,psi=None)
        R = ExEvo(N,R,po, ext=False,psi=None)
    
    cov = make_cov_0(N)
    Cov_mat = R @ cov @ R.T
    
    l = JW_Ms(N)
    
    '''
    #P = Pauli('ZIIIIIIIII') 
    P = random_pauli(N); gindsP = op_indices(N,P,l)
    exp_val = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)
    '''
    for wc in range(len(wl)):
        w = wl[wc]
        for k in range(50):
            ginds = np.arange(0,100,1)
            ginds = np.random.permutation(ginds)
            gindsP = ginds[0:w]
            #v2 = np.log(np.abs(get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)))
            v2 = ((get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)))
            tl[wc].append(v2)
for v in range(len(tl)):
    plt.hist(tl[v], bins=100)
    plt.title('weight = ' + str(wl[v]))
    plt.xlabel('log of expectation values binned')
    plt.ylabel('Counts')
    plt.show()