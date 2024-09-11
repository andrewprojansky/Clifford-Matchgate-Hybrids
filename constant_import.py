#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:27:15 2024

@author: andrewprojansky

constant_import.py: File for all pre-defined constant list of indices
and gates, as well as all nessesary inputs for file. Pfaffian call is 
seperate file, which can be found and downloaded from
https://github.com/python-adaptive/paper/blob/master/pfaffian.py
"""


'''
All inputs in one file, as well as nice/nessesary pre-defined matrices. 
'''

import quimb.tensor as qtn
import numpy as np
from quimb import *
import scipy.linalg as lng
from qiskit_qec.operators import Pauli
from qiskit.quantum_info import random_clifford
from qiskit.quantum_info import random_pauli
from pfaffian import *

ips = [[1,2],[0,3],[1,3],[0,2],[0,1],[2,3]]

H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])
P = np.array([[1,0],[0,1j]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
CNOTb = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
Id = np.identity(4)
sId = np.identity(2)
delta_list = [Id, CNOT, CNOTb, SWAP]
sl = [-1, 1]
T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
fSWAP = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
HTs = np.array([np.kron(X,X), np.kron(Y,Y), np.kron(X,Y), np.kron(Y,X), np.kron(Z,np.identity(2)), np.kron(np.identity(2),Z)])


