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

N = 100
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

for j in tqdm(range(100)):
    P = random_pauli(N); gindsP = op_indices(N,P,l)
    v2 = get_exp(Cov_mat, gindsP)*Multiply_forPhase2(gindsP, N)
    #print(P,v2) #these values are really small for randomly chosen Paulis...
    #worth noting. Really dominated by expectation values over small length
    #ginds
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
    
    if np.round(v1,5) != np.round(v2,5):
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
    
    if np.round(v1,5) != np.round(v2,5):
        print(P)
        print(gindsP)
        print(Multiply_forPhase2(gindsP, N, Ext=True))
        print(v1)
        print(v2)


