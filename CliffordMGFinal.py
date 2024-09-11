#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:28:39 2024

@author: andrewprojansky
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for testing Cliffords following Matchgates and the 
simulability of such. With this code, for Cliffords following Matchgate and 
Matchgate + Circuits we can calculate all Pauli expectation values
in polynomial time. Will be adding support for completely random MG+ circuits
as well as conjugated/sandwich MG circuits on computational basis states
"""

from constant_import import *

'''
All Functions for Code... will eventually segment more
'''

def Pair_maker(N):
    """
    makes pairs of indices for doing random brickwork

    Parameters
    ----------
    N : int
        number of qubits

    Returns
    -------
    even_l : list
        even pairs in brickwork
    odd_l : list
        odd pairs in brickwork

    """
    
    even_l = [[2*i,2*i+1] for i in range(N//2)]
    odd_l = [[2*i+1, 2*i+2] for i in range((N-1)//2)]
    return even_l, odd_l

def JW_Ms(N):
    """
    Generates X and Z in terms of majoranas, ordered 

    Parameters
    ----------
    N : int
        number of qubits

    Returns
    -------
    l : list
        list of how to write X and Z in terms of majoranas, ordered with Xs 
        first then Zs

    """
    
    l = []
    for j in range(N):
        lext = ([1]*(2*j+1))+([0]*(2*N-2*j-1))
        l.append(lext)
    for j in range(N):
        lext = [0]*(2*N)
        lext[2*j] = 1
        lext[2*j+1] = 1
        l.append(list(lext))
    return l

def make_cov_0(N, Ext=False):
    """
    makes base covariance matrix either for MG or MG+; [[0,1],[-1,0]] block 
    diagonal for MG, and the same but with an extra row/column at start all 
    zero. 

    Parameters
    ----------
    N : int
        number of qubits
    Ext : bool, optional
        If we're dealing with MG+ or not. The default is False.

    Returns
    -------
    Cov_mat : array
        block diagonal covariance matrix

    """
    
    if Ext == True:
        Cov_matb = np.kron(np.identity(N),np.array([[0,1],[-1,0]]))
        Cov_mat = np.zeros((2*N+1,2*N+1))
        for d1 in range(2*N):
            for d2 in range(2*N):
                Cov_mat[d1+1,d2+1] = Cov_matb[d1,d2]
    else:
        Cov_mat = np.kron(np.identity(N),np.array([[0,1],[-1,0]]))
        
    return Cov_mat

def ExEvo(N,R,pairs, psi = None, ext=False):
    """
    Facilitates evolution of random matchgate circuits, either in the original
    space of majoranas or the extended space. Returns test psi (which I am 
    going to make optional) and representation of unitary in SO(2n+1), R. 

    Parameters
    ----------
    N : int
        Number of qubits
    R : array
        representation of state/unitary in SO(2n+1)
    pairs : list
        list of pairs of indices which is gone through in order
    ext : bool, optional
        If we're dealing with MG+ or not. The default is False.
    psi : numpy.ndarray or None, optional
        matrix prodcut state from quimb if given, or None type

    Returns
    -------
    psi : numpy.ndarray
        updated matrix prodcut state from quimb 
    R : array
        updated representation of state/unitary in SO(2n+1)

    """
    if ext==False:
        dim = 2*N
    else:
        dim = 2*N+1
    for p in pairs:
        ind = p[0]
        #generates random coefficients for fermionic Hamiltonian
        coeffs = np.random.normal(0,1,6)
        H = np.zeros((dim,dim))
        for ps in range(len(ips)):
            H[ips[ps][0]+2*ind+ext,ips[ps][1]+2*ind+ext] = coeffs[ps]/2
            H[ips[ps][1]+2*ind+ext,ips[ps][0]+2*ind+ext] = -coeffs[ps]/2
        R = lng.expm(4*H) @ R
        #need minus signs in correct places to translate back to spin 
        #Hamiltonian for testing against statevector simulation
        if type(psi) != type(None):
            coeffs = np.multiply(coeffs, np.array([-1,1,-1,1,-1,-1]))
            HS = np.zeros((4,4),dtype='complex')
            for k in range(6):
                HS += coeffs[k]*HTs[k]
            mg = lng.expm(-1j*HS)
            psi.gate_split(mg, (p[0],p[1]), inplace=True)
    if type(psi) != type(None):
        return psi, R
    else:
        return R

def localRCov(N,R, st=None):
    """
    Generates random single qubit rotations on first qubit, assuming we're 
    using MG+. This is called in the making of (pseudo)-random product states 
    (pseudo here saying they aren't Haar random distributed)
     
     
     as a detail worth noting, the signs throughout this code need to be
     treated very carefully. If starting with spins then going back to 
     fermions, for X, Y, and Z, coefficients in the antisymmetric
     representation of our FF Hamiltonian must be flipped. 

    Parameters
    ----------
    N : int
        number of qubits
    st : numpy.ndarray
        matrix prodcut state from quimb 
    R : array
        representation of state/unitary in SO(2n+1)

    Returns
    -------
    st : numpy.ndarray
        updated matrix prodcut state from quimb 
    R : array
        updated representation of state/unitary in SO(2n+1)

    """
    
    Xcoef = np.random.normal()
    Zcoef = np.random.normal()
    Ycoef = np.random.normal()
    
    #Do Y rotations first
    H = np.zeros((2*N+1,2*N+1))
    H[0,2] = -Ycoef/2
    H[2,0] = Ycoef/2
    Hexp = np.real(lng.expm(4*H))
    R = Hexp @ R
    
    #Then X rotations 
    H = np.zeros((2*N+1,2*N+1))
    H[0,1] = -Xcoef/2
    H[1,0] = Xcoef/2
    Hexp = np.real(lng.expm(4*H))
    R = Hexp @ R
    
    #Then Z rotations
    H = np.zeros((2*N+1,2*N+1))
    H[1,2] = -Zcoef/2
    H[2,1] = Zcoef/2
    Hexp = np.real(lng.expm(4*H))
    R = Hexp @ R
    
    if type(st) != type(None):
        g1 = lng.expm(-1j*Ycoef*Y)
        st.gate(g1, 0, inplace=True,contract=True)
        g2 = lng.expm(-1j*Xcoef*X)
        st.gate(g2, 0, inplace=True,contract=True)
        g3 = lng.expm(-1j*Zcoef*Z)
        st.gate(g3, 0, inplace=True,contract=True)
        return st, R
    else:
        return R

def fSWAPs(N, R, k, st=None):
    """
    Performs fSWAP gate on nearest neighbors: called in making product states,
    to fSWAP local rotations down chain of qubits

    Parameters
    ----------
    N : int
        number of qubits
    st : numpy.ndarray
        matrix prodcut state from quimb 
    R : array
        representation of state/unitary in SO(2n+1)
    k : int
        indice of fSWAP to act on from k and k+1

    Returns
    -------
    st : numpy.ndarray
        updated matrix prodcut state from quimb 
    R : array
        updated representation of state/unitary in SO(2n+1)

    """

    #First, the (XX+YY) term
    H = np.zeros((2*N+1,2*N+1))
    H[2*k+2, 2*k+3] = np.pi/8
    H[2*k+3, 2*k+2] = -np.pi/8
    H[2*k+1, 2*k+4] = -np.pi/8
    H[2*k+4, 2*k+1] = np.pi/8
    Hexp = np.real(lng.expm(4*H))
    R = Hexp @ R
    #followed by S \otimes S^{\dagger}: needed to make fSWAP instead of 
    #generalized fSWAP
    H = np.zeros((2*N+1,2*N+1))
    H[2*k+1,2*k+2] = -np.pi/8
    H[2*k+2,2*k+1] = np.pi/8
    H[2*k+3,2*k+4] = -np.pi/8
    H[2*k+4,2*k+3] = np.pi/8
    Hexp = np.real(lng.expm(4*H))
    R = Hexp @ R
    
    if type(st) != type(None):
        st.gate_split(fSWAP, (k,k+1), inplace=True)
        return st, R
    else:
        return R

def RProdfromCov(N, R, st=None):
    """
    Makes product state, tracks evolution in SO(2n+1) rep of unitary

    Parameters
    ----------
    N : int
        number of qubits
    st : numpy.ndarray
        matrix prodcut state from quimb 
    R : array
        representation of state/unitary in SO(2n+1)

    Returns
    -------
    st : numpy.ndarray
        updated matrix prodcut state from quimb 
    R : array
        updated representation of state/unitary in SO(2n+1)

    """
    if type(st) != type(None):
        for j in range(N):
            for k in range(5):
                st, R = localRCov(N, R,st)
            for k in range(N-j-1):
                st, R = fSWAPs(N, R, k,st)
        return st, R
    else:
        for j in range(N):
            for k in range(5):
                R = localRCov(N, R)
            for k in range(N-j-1):
                R = fSWAPs(N, R, k)
        return R
            
def op_indices(N, p, JWL, Ext=False):
    """
    Gets the majoranas needed to write some Pauli in terms of the Jordan-
    Wigner encoding. Works either on MG or MG+, on MG+ extends indices/adjust 
    for extra mode added. 

    Parameters
    ----------
    N : int
        number of qubits
    p : table
        from Qiskit, description of Pauli
    JWL : list
        Jordan-Wigner majoranas, written in sympletic form
    Ext : bool, optional
        If we're dealing with MG+ or not. The default is False.

    Returns
    -------
    list
        list of indices which determins which majoranas are used in writing
        pauli P

    """
    
    oinds = np.zeros(2*N)
    for l in range(len(p.to_label())):
        if p.to_label()[l] == 'X':
            oinds = np.round(np.array((oinds+ JWL[l]))%2)
        elif p.to_label()[l] == 'Y':
            oinds = np.round(np.array((oinds+ JWL[l]))%2)
            oinds = np.round(np.array((oinds+ JWL[l+N]))%2)
        elif p.to_label()[l] == 'Z':
            oinds = np.round(np.array((oinds+ JWL[l+N]))%2)
    if Ext==False:
        return list(oinds.nonzero()[0])
    #if True, extend all current inds by 1, and if length is odd, then we know 
    #we need the parity operator to write in terms of even string of majoranas
    elif Ext == True:
        l = oinds.nonzero()[0]
        for i in range(len(l)):
            l[i] = l[i]+1
        if len(l)%2 == 0:
            return l
        else:
            return [0]+list(l)

def get_exp(Cov, inds):
    """
    Gets expectation value of operator by taking Pfaffian of covariance
    matrix: if there is only one ind, then we default to zero to avoid error
    in pfaffian()

    Parameters
    ----------
    Cov : array
        covariance matrix for our MG or MG+ evolved system
    inds : list
        indices for which we define submatrix to take Pfaffian over

    Returns
    -------
    float
        returns Pfaffian

    """
    
    if len(inds) == 1:
        return 0
    else:
        '''
        Pfaffian called from outside module. Needed because in pf(A)^2 = det(A)
        solving for det doesn't tell us if Pfaffian is minus or positive:
        however, we can use matrix identity that Pf(BAB^T) = det(B)pf(A), to 
        not have sign ambiguity
        '''
        return pfaffian(Cov[np.ix_(inds,inds)])

def TexEvo(N,R, pairs,st=None):
    """
    Placeholder for random MG+ evolution

    Parameters
    ----------
    N : int
        Number of qubits
    psi : numpy.ndarray
        matrix prodcut state from quimb 
    R : array
        representation of state/unitary in SO(2n+1)
    pairs : list
        list of pairs of indices which is gone through in order

    Returns
    -------
    psi : numpy.ndarray
        updated matrix prodcut state from quimb 
    R : array
        updated representation of state/unitary in SO(2n+1)

    """
    pass

def JWL(N):
    """
    Writes Jordan-Wigner majoranas in sympletic form

    Parameters
    ----------
    N : int
        number of qubits

    Returns
    -------
    jwl_ : list
        Jordan-Wigner majoranas in sympletic form

    """
    
    jwl_ = []
    
    for j in range(N):
        l = np.zeros(2*N)
        for k in range(j):
            l[k+N] = 1
        l[j] = 1
        jwl_.append(list(np.round(l)))
        brow = np.zeros(2*N)
        brow[j+N] = 1
        jwl_.append(list(np.round(np.array(l)+brow)))
    return jwl_

def read_tostr(maj, N):
    """
    reads symplletic form and writes corresponding Pauli string

    Parameters
    ----------
    maj : list
        sympletic representation of Pauli
    N : int
        number of qubits

    Returns
    -------
    wd : str
        string representing pauli

    """
    
    wd = ''
    for l in range(N):
        if maj[l] == 1:
            if maj[l+N] == 1:
                wd = wd + 'Y'
            else:
                wd = wd + 'X'
        elif maj[l+N] == 1:
            wd = wd + 'Z'
        else:
            wd = wd + 'I'
    return wd

def MultiplyP(w1, w2):
    """
    Multiplies two Pauli strings together, most importantly keeping track
    of signs of multiplied strings. Could in theory make this work over
    sympletic representation, but if it ain't broke... 

    Parameters
    ----------
    w1 : str
        Pauli string
    w2 : str
        Pauli string

    Returns
    -------
    wd : str
        multiplied Pauli string
    phase : int
        1,1j,-1j,-1, phase from multiplying together Paulis

    """
    
    wd = ''
    phase = 1
    for lett in range(len(w1)):
        l1 = w1[lett]
        l2 = w2[lett]
        if l1 == l2:
            wd = wd + 'I'
        elif l1 == 'I' or l2 == 'I':
            wd = wd + [l1,l2][([l1,l2].index('I')+1)%2]
        elif l1 == 'X':
            if l2 == 'Y':
                wd = wd + 'Z'
                phase = phase * (1j)
            elif l2 == 'Z':
                wd = wd + 'Y'
                phase = phase * (-1j)
        elif l1 == 'Y':
            if l2 == 'Z':
                wd = wd + 'X'
                phase = phase * (1j)
    return wd, phase

def Multiply_forPhase2(ginds, N, Ext=False):
    """
    Multiplies two Paulis together to get back phase information; 
    needed because it may be that for the Pauli we'd like to get the 
    expectation value of, the Pfaffian of our covariance matrix gives us 
    minus the expectation value

    Parameters
    ----------
    ginds : list
        majoranas used to make pauli
    N : int
        number of qubits
    Ext : bool, optional
        If we're dealing with MG+ or not. The default is False.

    Returns
    -------
    phase : int
        phase of plus or minus 1

    """
    
    Plist = []
    phase = (-1j)**(len(ginds)//2)
    jwll = JWL(N)
    if Ext==True:
        Parity = [0]*N + [1]*N
        jwll = [Parity]+(jwll)
        if ginds[0] == 0:
            ginds = ginds[1:]
            #extra phase information needed, from the translation from 
            #majoranas in extended space back to orginal space
            phase = phase * 1j
    for ind in ginds:
        Plist.append(read_tostr(jwll[ind], N))
    
    fp = Plist[0]
    for ind in np.arange(1,len(Plist),1):
        fp, ph = MultiplyP(fp, Plist[ind])
        phase = ph * phase
    return phase