#
#Copyright 2022 Kurt R. Brorsen
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import numpy as np
import sys
import numpy.linalg as LA
import time
import diis 
from diis import mc_cc_diis
cimport cython

#note that this only works for one quantum nucleus/positron. Would need to put in T2pp terms.

def spat_fock(e_mo,ntot,p_mo,ptot):
    #build elec and prot fock matrix in spinorb
    fe = np.zeros(ntot)
    fp = np.zeros(ptot)
    
    for i in range(ntot):
        fe[i] = e_mo[i]

    for i in range(ptot):
        fp[i] = p_mo[i]

    fe = np.diag(fe)
    fp = np.diag(fp)
    return fe, fp


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef c_amps( const double[:,:] fe, const double[:,:,:,:] eri, int nocc, int ntot, const double[:,:] fp, const double[:,:,:,:] eri_ep, int pocc, int ptot): 
    #get intial (mp2) amps and compute the (mc-)mp2 energy
    print(f'MP2 start: {time.perf_counter()}')
    cdef int i,j,a,b,I,A,nvir,pvir
    nvir = ntot - nocc
    pvir = ptot - pocc
    cdef double e_mp2, mc_emp2

    cdef double[:,:,:,:] t2ee = np.zeros((nocc,nocc,nvir,nvir),dtype=float)
    cdef double[:,:] t1e = np.zeros((nocc,nvir),dtype=float)
    cdef double[:,:,:,:] t2ep = np.zeros((nocc,pocc,nvir,pvir),dtype=float)
    cdef double[:,:] t1p = np.zeros((pocc,pvir),dtype=float)


    e_mp2 = 0
    mc_emp2 = 0
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc, ntot):
                for b in range(nocc, ntot):
                    t2ee[i,j,a-nocc,b-nocc] = ( ( eri[i,j,a,b] ) / ( fe[i,i] + fe[j,j] - fe[a,a] - fe[b,b] ) )
                    e_mp2 += (2*eri[i,j,a,b] - eri[i,j,b,a]) * t2ee[i,j,a-nocc,b-nocc]


    for i in range(nocc):
        for I in range(pocc):
            for a in range(nocc,ntot):
                for A in range(pocc,ptot):
                    t2ep[i,I,a-nocc,A-pocc] = -2* ( ( eri_ep[i,I,a,A] ) / ( fe[i,i] + fp[I,I] - fe[a,a] - fp[A,A] ) )
                    mc_emp2 += eri_ep[i,I,a,A]* t2ep[i,I,a-nocc,A-pocc]

    print(f'mp2 energy is {e_mp2}')
    print(f'mc-mp2 energy: {mc_emp2}')
    print(f'MP2 stop: {time.perf_counter()}')
    return t1e, t2ee, t1p, t2ep


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef eff_dubs_e(int nocc, int nvir, const double[:,:] t1e, const double[:,:,:,:] t2ee):

#    print(f'ee doubles start: {time.perf_counter()}')   
 
    cdef int i,j,a,b,ntot
    ntot = nocc + nvir
    cdef double[:,:,:,:] tau = np.zeros((nocc,nocc,nvir,nvir),dtype=float)
    cdef double[:,:,:,:] T = np.zeros((nocc,nocc,nvir,nvir),dtype=float)

    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc,ntot):
                for b in range(nocc,ntot):
                    tau[i,j,a-nocc,b-nocc] = t2ee[i,j,a-nocc,b-nocc] + t1e[i,a-nocc] * t1e[j,b-nocc]
                    T[i,j,a-nocc,b-nocc] = 0.5 * t2ee[i,j,a-nocc,b-nocc] + t1e[i,a-nocc] * t1e[j,b-nocc]

#    print(f'ee doubles stop: {time.perf_counter()}') 
    return tau, T


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef ee_intermediates(const double[:,:] fock, const double[:,:,:,:] eri, int nocc, int nvir, const double[:,:] t1, const double[:,:,:,:] t2, const double[:,:,:,:] tau, const double[:,:,:,:] T):
    print(f'ee intermediates start: {time.perf_counter()}')

    cdef int u,v,B,y,i,j,a,b,c,ntot
    cdef double[:,:] hui = np.zeros((nocc,nocc),dtype=float)
    cdef double[:,:] hab = np.zeros((nvir,nvir),dtype=float)
    cdef double[:,:] hai = np.zeros((nvir,nocc),dtype=float)
    cdef double[:,:] gui = np.zeros((nocc,nocc),dtype=float)
    cdef double[:,:] gab = np.zeros((nvir,nvir),dtype=float)
    cdef double[:,:,:,:] Ai = np.zeros((nocc,nocc,nocc,nocc),dtype=float)
    cdef double[:,:,:,:] Bi = np.zeros((nvir,nvir,nvir,nvir),dtype=float)
    cdef double[:,:,:,:] J = np.zeros((nocc,nvir,nvir,nocc),dtype=float)
    cdef double[:,:,:,:] K = np.zeros((nocc,nvir,nocc,nvir),dtype=float)
    ntot = nocc + nvir

    for a in range(nocc,ntot):
        for B in range(nocc,ntot):
            if a != B:
                hab[a-nocc,B-nocc] += fock[a,B]
                gab[a-nocc,B-nocc] += fock[a,B]
            for i in range(nocc):
                gab[a-nocc,B-nocc] -= fock[a,i] * t1[i,B-nocc]
                for b in range(nocc,ntot):
                    gab[a-nocc,B-nocc] += ( 2 * eri[B,i,a,b] - eri[B,i,b,a] ) * t1[i,b-nocc]
                for j in range(nocc):
                    for b in range(nocc,ntot):
                        hab[a-nocc,B-nocc] -= ( 2 * eri[i,j,a,b] - eri[j,i,a,b] ) * tau[i,j,B-nocc,b-nocc]
                        gab[a-nocc,B-nocc] -= ( 2 * eri[i,j,a,b] - eri[j,i,a,b] ) * tau[i,j,B-nocc,b-nocc]
    for i in range(nocc):
        for u in range(nocc):
            if i != u:
                hui[u,i] += fock[u,i]
                gui[u,i] += fock[u,i]
            for a in range(nocc,ntot):
                gui[u,i] += fock[i,a] * t1[u,a-nocc]
                for j in range(nocc):
                    gui[u,i] += ( 2 * eri[i,j,u,a] - eri[j,i,u,a] ) * t1[j,a-nocc]
                    for b in range(nocc,ntot):
                        hui[u,i] += ( 2 * eri[i,j,a,b] - eri[i,j,b,a] ) * tau[u,j,a-nocc,b-nocc]
                        gui[u,i] += ( 2 * eri[i,j,a,b] - eri[i,j,b,a] ) * tau[u,j,a-nocc,b-nocc]
        for a in range(nocc,ntot):
            hai[a-nocc,i] += fock[a,i]
            for j in range(nocc):
                for b in range(nocc,ntot):
                    hai[a-nocc,i] += ( 2 * eri[i,j,a,b] - eri[i,j,b,a] ) * t1[j,b-nocc]

    for i in range(nocc):
        for u in range(nocc):
            for j in range(nocc):
                for v in range(nocc):
                    Ai[u,v,i,j] += eri[i,j,u,v]
                    for a in range(nocc,ntot):
                        Ai[u,v,i,j] += ( eri[i,j,u,a] * t1[v,a-nocc] + eri[i,j,a,v] * t1[u,a-nocc] )
                        for b in range(nocc,ntot):
                            Ai[u,v,i,j] += eri[i,j,a,b] * tau[u,v,a-nocc,b-nocc]

            for B in range(nocc,ntot):
                for a in range(nocc,ntot):
                    J[u,a-nocc,B-nocc,i] += eri[B,i,u,a]
                    K[u,a-nocc,i,B-nocc] += eri[i,B,u,a]
                    for j in range(nocc):
                        J[u,a-nocc,B-nocc,i] -= eri[j,i,u,a] * t1[j,B-nocc]
                        K[u,a-nocc,i,B-nocc] -= eri[i,j,u,a] * t1[j,B-nocc]
                        for b in range(nocc,ntot):
                            J[u,a-nocc,B-nocc,i] -= eri[i,j,a,b] * T[u,j,b-nocc,B-nocc] 
                            J[u,a-nocc,B-nocc,i] += 0.5 * ( 2 * eri[i,j,a,b] - eri[i,j,b,a] ) * t2[u,j,B-nocc,b-nocc]  # may have screwed up the distributive pty here...
                            K[u,a-nocc,i,B-nocc] -= eri[i,j,b,a] * T[u,j,b-nocc,B-nocc]

                    for b in range(nocc,ntot):
                        J[u,a-nocc,B-nocc,i] += eri[B,i,b,a] * t1[u,b-nocc]
                        K[u,a-nocc,i,B-nocc] += eri[i,B,b,a] * t1[u,b-nocc]

    for a in range(nocc,ntot):
        for b in range(nocc,ntot):
            for B in range(nocc,ntot):
                for y in range(nocc,ntot):
                    Bi[a-nocc,b-nocc,B-nocc,y-nocc] += eri[B,y,a,b] 
                    for i in range(nocc):
                        Bi[a-nocc,b-nocc,B-nocc,y-nocc] -= eri[B,i,a,b] * t1[i,y-nocc] 
                        Bi[a-nocc,b-nocc,B-nocc,y-nocc] -= eri[i,y,a,b] * t1[i,B-nocc] 


    print(f'ee intermediates end: {time.perf_counter()}') 
    return hab, hui, hai, gui, gab, Ai, Bi, J, K 

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef T1e( const double[:,:] fock, const double[:,:,:,:] eri, int nocc, int nvir, const double[:,:] t1e, const double[:,:,:,:] t2ee, const double[:,:] hab, const double[:,:] hui, const double[:,:] hai, const double[:,:,:,:] eri_ep, const double[:,:] fockp, int pocc, int pvir, const double[:,:] t1p, const double[:,:,:,:] t2ep, const double[:,:,:,:] tau):
    # compute t1e amps. electronic parts see stanton jcp 1990; ep and p parts see pavosevic 2019
    print(f'T1e start: {time.perf_counter()}')
    
    cdef int i,a,m,e,f,n,K,C,c,k,u,B,j,b,ntot,ptot
    ptot = pocc + pvir
    ntot = nocc + nvir
    cdef double D1e
    cdef double[:,:] T1e = np.zeros((nocc,nvir),dtype=float)

    for u in range(nocc):
        for B in range(nocc,ntot):
            T1e[u,B-nocc] += fock[u,B]
#            D1e = fock[u,u] - fock[B,B]
#            
            for i in range(nocc):
                T1e[u,B-nocc] -= t1e[i,B-nocc] * hui[u,i]

                for a in range(nocc,ntot):
                    T1e[u,B-nocc] -= 2 * fock[i,a] * t1e[i,B-nocc] * t1e[u,a-nocc]
                    T1e[u,B-nocc] += hai[a-nocc,i] * ( 2 * t2ee[i,u,a-nocc,B-nocc] - t2ee[u,i,a-nocc,B-nocc] + t1e[u,a-nocc] * t1e[i,B-nocc] )
                    T1e[u,B-nocc] += ( 2 * eri[i,B,a,u] - eri[i,B,u,a] ) * t1e[i,a-nocc]
                    
                    for b in range(nocc,ntot):
                        T1e[u,B-nocc] += ( 2 * eri[i,B,a,b] - eri[i,B,b,a] ) * tau[i,u,a-nocc,b-nocc]

                    for j in range(nocc):
                        T1e[u,B-nocc] -= ( 2 * eri[i,j,a,u] - eri[j,i,a,u] ) * tau[i,j,a-nocc,B-nocc]

            for a in range(nocc,ntot):
                T1e[u,B-nocc] += t1e[u,a-nocc] * hab[a-nocc,B-nocc]
#
##
##    #now ep terms
    for i in range(nocc):
        for a in range(nocc,ntot):
            D1e = fock[i,i] - fock[a,a]
            for K in range(pocc):
                for C in range(pocc,ptot):
                   T1e[i,a-nocc] -= 2*0.5*eri_ep[a,K,i,C]*t1p[K,C-pocc] #check
#
                   T1e[i,a-nocc] += 0.5*fockp[C,K] * t2ep[i,K,a-nocc,C-pocc] #check (switched fock matrix indices, not that it matters for canon. orbs)
#                   
                   for c in range(nocc,ntot):
                       T1e[i,a-nocc] -= 0.5*eri_ep[a,K,c,C] * t2ep[i,K,c-nocc,C-pocc] #check
                       T1e[i,a-nocc] -= 2*0.5*eri_ep[a,K,c,C]*t1e[i,c-nocc]*t1p[K,C-pocc] #check 
#
                       for k in range(nocc):
                           T1e[i,a-nocc] -= 0.5*2*eri_ep[k,K,c,C]*t1e[k,c-nocc]*t2ep[i,K,a-nocc,C-pocc] #check
                           T1e[i,a-nocc] += 0.5*eri_ep[k,K,c,C]*t1e[i,c-nocc]*t2ep[k,K,a-nocc,C-pocc] #check
                           T1e[i,a-nocc] -= 2*0.5*eri_ep[k,K,c,C]*t1p[K,C-pocc]*(t2ee[i,k,a-nocc,c-nocc]-t2ee[i,k,c-nocc,a-nocc])
                           T1e[i,a-nocc] -= 2*0.5*eri_ep[k,K,c,C]*t1p[K,C-pocc]*t2ee[i,k,a-nocc,c-nocc] #check
                           T1e[i,a-nocc] += 0.5*eri_ep[k,K,c,C]*t2ep[i,K,c-nocc,C-pocc]*t1e[k,a-nocc] #check
                           T1e[i,a-nocc] += 2*0.5*eri_ep[k,K,c,C]*t1e[i,c-nocc]*t1p[K,C-pocc]*t1e[k,a-nocc] #check
#
                   for k in range(nocc):
                       T1e[i,a-nocc] += 0.5*eri_ep[k,K,i,C]* t2ep[k,K,a-nocc,C-pocc] #check 
                       T1e[i,a-nocc] += 2*0.5*eri_ep[k,K,i,C]*t1p[K,C-pocc]*t1e[k,a-nocc] #check
#
            T1e[i,a-nocc] /= D1e
##            T1e[u,B-nocc] /= D1e
            D1e = 0

    print(f'T1e stop: {time.perf_counter()}')         
#    T1e = build_t1(T1e)
    return T1e


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef T1p( const double[:,:] fe, const double[:,:,:,:] tei_mo, int nocc, int nvir, const double[:,:] t1e, const double[:,:,:,:] t2ee, const double[:,:,:,:] eri_ep, const double[:,:] fockp, int pocc, int pvir, const double[:,:] t1p, const double[:,:,:,:] t2ep):
    #compute t1p amps. protonic part is just normal cc w/no two particle integrals. ep and e parts see pavosevic 2019.
    print(f'T1p start: {time.perf_counter()}')
    cdef double D1p
    cdef int i,a,e,m,k,c,K,C,d,l,I,A,ptot,ntot
    ntot = nocc + nvir
    ptot = pocc + pvir
    cdef double[:,:] T1p = np.zeros((pocc,pvir),dtype=float)

    
    # trying pp terms w/o using intermediates.
    for I in range(pocc):
        for A in range(pocc,ptot):
            T1p[I,A-pocc] += fockp[I,A]
            D1p = fockp[I,I] - fockp[A,A]
            for C in range(pocc,ptot):
                if C != A:
                    T1p[I,A-pocc] += fockp[C,A]*t1p[I,C-pocc]
                for K in range(pocc):
                    T1p[I,A-pocc] -= fockp[C,K]*t1p[I,C-pocc]*t1p[K,A-pocc]
            for K in range(pocc):
                if K != I:
                    T1p[I,A-pocc] -= fockp[I,K]*t1p[K,A-pocc]



  #now ep terms. note here that lowecase i and a are actually proton indices. change later? changed.

            for k in range(nocc):
                for c in range(nocc,ntot):
                    T1p[I,A-pocc] -= 2*eri_ep[k,A,c,I]*t1e[k,c-nocc] #check

                    T1p[I,A-pocc] += fe[c,k] * t2ep[k,I,c-nocc,A-pocc] #check. switched indices.
#                   
                    for C in range(pocc,ptot):
                        T1p[I,A-pocc] -= eri_ep[k,A,c,C] * t2ep[k,I,c-nocc,C-pocc] #check
                        T1p[I,A-pocc] -= 2*eri_ep[k,A,c,C]*t1e[k,c-nocc]*t1p[I,C-pocc] #check
#
                        for K in range(pocc):
                            T1p[I,A-pocc] -= eri_ep[k,K,c,C]*t1p[K,C-pocc]*t2ep[k,I,c-nocc,A-pocc] #check
                            T1p[I,A-pocc] += eri_ep[k,K,c,C]*t1p[I,C-pocc]*t2ep[k,K,c-nocc,A-pocc] #check
                            T1p[I,A-pocc] += eri_ep[k,K,c,C]*t2ep[k,I,c-nocc,C-pocc]*t1p[K,A-pocc] #check
                            T1p[I,A-pocc] += 2*eri_ep[k,K,c,C]*t1e[k,c-nocc]*t1p[I,C-pocc]*t1p[K,A-pocc] #check
#
                    for K in range(pocc):
                        T1p[I,A-pocc] += eri_ep[k,K,c,I]* t2ep[k,K,c-nocc,A-pocc] #check
                        T1p[I,A-pocc] += 2*eri_ep[k,K,c,I]*t1p[K,A-pocc]*t1e[k,c-nocc] #check
#
                    for d in range(nocc,ntot):
                        for l in range(nocc):
                            T1p[I,A-pocc] += (tei_mo[k,l,c,d]-tei_mo[l,k,c,d])*t2ep[k,I,c-nocc,A-pocc]*t1e[l,d-nocc] #check
                            T1p[I,A-pocc] += tei_mo[k,l,c,d]*t2ep[k,I,c-nocc,A-pocc]*t1e[l,d-nocc]
            T1p[I,A-pocc] /= D1p                               
            D1p = 0 

    print(f'T1p end: {time.perf_counter()}')          
    return T1p


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef T2ee( const double[:,:,:,:] eri, int nocc, int nvir, const double[:,:] t1e, const double[:,:,:,:] t2ee, const double[:,:] gui, const double[:,:] gab, const double[:,:,:,:] Ai, const double[:,:,:,:] Bi, const double[:,:,:,:] J, const double[:,:,:,:] Kp, const double[:,:] fe, const double[:,:,:,:] eri_ep, const double[:,:] fockp, const double[:,:] t1p, const double[:,:,:,:] t2ep, int pocc, int pvir, const double[:,:,:,:] tau):   
    # compute t2ee amps. electronic part is stanton jcp 1990 (only computed for t(alpha-beta)(beta-alpha), then construct full array via scuseria jcp 1987 in build_t2). ep and p parts see pavosevic jctc 2019.
    print(f'T2ee start: {time.perf_counter()}') 
    cdef int i,j,a,b,e,m,f,n,c,k,C,K,ntot,ptot,B,u,y,v
    ntot = nocc + nvir
    ptot = pocc + pvir
    cdef double D2ee = 0
#    cdef double[:,:,:,:] tao2ee = eff_dubs_e(s_occ,nso,t1e,t2ee)[1]
    cdef double[:,:,:,:] T2ee = np.zeros((nocc,nocc,nvir,nvir),dtype=float)

    cdef double[:,:,:,:] BB = np.zeros((nocc,nocc,nvir,nvir),dtype=float)
    cdef double[:,:,:,:] H = np.zeros((nocc,nvir,nvir,nvir),dtype=float)
    cdef double[:,:,:,:] JJ = np.zeros((nocc,nocc,nocc,nvir),dtype=float)
    cdef double[:,:] D = np.zeros((nvir,nvir),dtype=float)
    cdef double[:,:] G = np.zeros((nocc,nocc),dtype=float)
    cdef double[:,:] KK = np.zeros((nocc,nvir),dtype=float)

    for i in range(nocc):
        for a in range(nocc,ntot):
            for b in range(nocc,ntot):
                for c in range(nocc,ntot):
                    for K in range(pocc):
                        for C in range(pocc,ptot):
                            H[i,c-nocc,a-nocc,b-nocc] += eri_ep[b,K,a,C]*t2ep[i,K,c-nocc,C-pocc]
        for k in range(nocc):
            for j in range(nocc):
                for a in range(nocc,ntot):
                    for K in range(pocc):
                        for C in range(pocc,ptot):
                            JJ[i,j,k,a-nocc] += eri_ep[k,K,j,C]*t2ep[i,K,a-nocc,C-pocc]
            for a in range(nocc,ntot):
                for c in range(nocc,ntot):
                    for K in range(pocc):
                        for C in range(pocc,ptot):
                            BB[i,k,a-nocc,c-nocc] += eri_ep[k,K,c,C]*t2ep[i,K,a-nocc,C-pocc]

    for K in range(pocc):
        for C in range(pocc,ptot):
            for a in range(nocc,ntot):
                for b in range(nocc,ntot):
                    D[a-nocc,b-nocc] += t1p[K,C-pocc]*eri_ep[b,K,a,C]
            for i in range(nocc):
                for j in range(nocc):
                    G[i,j] += t1p[K,C-pocc]*eri_ep[j,K,i,C]
            for k in range(nocc):
                for c in range(nocc,ntot):
                    KK[k,c-nocc] += t1p[K,C-pocc]*eri_ep[k,K,c,C]


    for u in range(nocc):
        for v in range(nocc):
            for B in range(nocc,ntot):
                for y in range(nocc,ntot):
                    T2ee[u,v,B-nocc,y-nocc] += 2*eri[B,y,u,v]

                    for i in range(nocc):
                        T2ee[u,v,B-nocc,y-nocc] -= 2*t2ee[i,v,B-nocc,y-nocc] * gui[u,i]
                        T2ee[u,v,B-nocc,y-nocc] -= 2*t2ee[i,u,y-nocc,B-nocc] * gui[v,i]

                        T2ee[u,v,B-nocc,y-nocc] -= 2*eri[B,i,u,v]*t1e[i,y-nocc]
                        T2ee[u,v,B-nocc,y-nocc] -= 2*eri[y,i,v,u]*t1e[i,B-nocc]

                        for a in range(nocc,ntot):
                            T2ee[u,v,B-nocc,y-nocc] -= 2*eri[i,y,u,a]*t1e[i,B-nocc]*t1e[v,a-nocc]
                            T2ee[u,v,B-nocc,y-nocc] -= 2*eri[i,B,v,a]*t1e[i,y-nocc]*t1e[u,a-nocc]
                            T2ee[u,v,B-nocc,y-nocc] -= 2*eri[B,i,u,a]*t1e[v,a-nocc]*t1e[i,y-nocc]
                            T2ee[u,v,B-nocc,y-nocc] -= 2*eri[y,i,v,a]*t1e[u,a-nocc]*t1e[i,B-nocc]

                            T2ee[u,v,B-nocc,y-nocc] += 2*0.5 * ( ( 2. * J[u,a-nocc,B-nocc,i] - Kp[u,a-nocc,i,B-nocc] ) * ( 2. * t2ee[i,v,a-nocc,y-nocc] - t2ee[i,v,y-nocc,a-nocc] ) )
                            T2ee[u,v,B-nocc,y-nocc] += 2*0.5 * ( ( 2. * J[v,a-nocc,y-nocc,i] - Kp[v,a-nocc,i,y-nocc] ) * ( 2. * t2ee[i,u,a-nocc,B-nocc] - t2ee[i,u,B-nocc,a-nocc] ) )

                            T2ee[u,v,B-nocc,y-nocc] -= 2*0.5 * Kp[u,a-nocc,i,B-nocc] * t2ee[i,v,y-nocc,a-nocc]
                            T2ee[u,v,B-nocc,y-nocc] -= 2*0.5 * Kp[v,a-nocc,i,y-nocc] * t2ee[i,u,B-nocc,a-nocc]

                            T2ee[u,v,B-nocc,y-nocc] -= 2*Kp[u,a-nocc,i,y-nocc] * t2ee[i,v,B-nocc,a-nocc]
                            T2ee[u,v,B-nocc,y-nocc] -= 2*Kp[v,a-nocc,i,B-nocc] * t2ee[i,u,y-nocc,a-nocc]

                        for j in range(nocc):
                            T2ee[u,v,B-nocc,y-nocc] += 2*Ai[u,v,i,j] * tau[i,j,B-nocc,y-nocc]

                    for a in range(nocc,ntot):
                        T2ee[u,v,B-nocc,y-nocc] += 2*t2ee[u,v,a-nocc,y-nocc] * gab[a-nocc,B-nocc]
                        T2ee[u,v,B-nocc,y-nocc] += 2*t2ee[v,u,a-nocc,B-nocc] * gab[a-nocc,y-nocc]

                        T2ee[u,v,B-nocc,y-nocc] += 2*eri[B,y,u,a]*t1e[v,a-nocc]
                        T2ee[u,v,B-nocc,y-nocc] += 2*eri[y,B,v,a]*t1e[u,a-nocc]

                        for b in range(nocc,ntot):
                            T2ee[u,v,B-nocc,y-nocc] += 2*Bi[a-nocc,b-nocc,B-nocc,y-nocc] * tau[u,v,a-nocc,b-nocc] 

## now ep terms
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nocc,ntot):
                for b in range(nocc,ntot):
                    D2ee = 2*(fe[i,i] + fe[j,j] - fe[a,a] - fe[b,b])

                    for k in range(nocc):
                        T2ee[i,j,a-nocc,b-nocc] += 2*t2ee[i,k,a-nocc,b-nocc]*G[j,k]
                        T2ee[i,j,a-nocc,b-nocc] += 2*t2ee[j,k,b-nocc,a-nocc]*G[i,k]

                        T2ee[i,j,a-nocc,b-nocc] += t1e[k,b-nocc]*JJ[i,j,k,a-nocc]
                        T2ee[i,j,a-nocc,b-nocc] += t1e[k,a-nocc]*JJ[j,i,k,b-nocc]
                        for c in range(nocc,ntot):
                            T2ee[i,j,a-nocc,b-nocc] += t2ee[j,k,b-nocc,a-nocc]*BB[i,k,c-nocc,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] += t2ee[i,k,a-nocc,b-nocc]*BB[j,k,c-nocc,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] += t2ee[i,j,c-nocc,b-nocc]*BB[k,k,a-nocc,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] -= (t2ee[j,k,b-nocc,c-nocc]-t2ee[j,k,c-nocc,b-nocc])*BB[i,k,a-nocc,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] -= t2ee[j,k,b-nocc,c-nocc]*BB[i,k,a-nocc,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] += t2ee[i,j,a-nocc,c-nocc]*BB[k,k,b-nocc,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] -= (t2ee[i,k,a-nocc,c-nocc]-t2ee[i,k,c-nocc,a-nocc])*BB[j,k,b-nocc,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] -= t2ee[i,k,a-nocc,c-nocc]*BB[j,k,b-nocc,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] += t1e[j,c-nocc]*t1e[k,b-nocc]*BB[i,k,a-nocc,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] += t1e[i,c-nocc]*t1e[k,a-nocc]*BB[j,k,b-nocc,c-nocc]
                            
                            T2ee[i,j,a-nocc,b-nocc] += 2*t1e[k,b-nocc]*t2ee[i,j,a-nocc,c-nocc]*KK[k,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] += 2*t1e[j,c-nocc]*t2ee[i,k,a-nocc,b-nocc]*KK[k,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] += 2*t1e[i,c-nocc]*t2ee[j,k,b-nocc,a-nocc]*KK[k,c-nocc]
                            T2ee[i,j,a-nocc,b-nocc] += 2*t1e[k,a-nocc]*t2ee[i,j,c-nocc,b-nocc]*KK[k,c-nocc]

                    for c in range(nocc,ntot):
                        T2ee[i,j,a-nocc,b-nocc] -= 2*t2ee[i,j,a-nocc,c-nocc]*D[b-nocc,c-nocc]
                        T2ee[i,j,a-nocc,b-nocc] -= 2*t2ee[i,j,c-nocc,b-nocc]*D[a-nocc,c-nocc]

                        T2ee[i,j,a-nocc,b-nocc] -= t1e[j,c-nocc]*H[i,a-nocc,b-nocc,c-nocc]
                        T2ee[i,j,a-nocc,b-nocc] -= t1e[i,c-nocc]*H[j,b-nocc,a-nocc,c-nocc]

                    for K in range(pocc):
                        for C in range(pocc, ptot):
                            T2ee[i,j,a-nocc,b-nocc] -= eri_ep[b,K,j,C] * t2ep[i,K,a-nocc,C-pocc]
                            T2ee[i,j,a-nocc,b-nocc] -= eri_ep[a,K,i,C] * t2ep[j,K,b-nocc,C-pocc]

                    T2ee[i,j,a-nocc,b-nocc] /= D2ee
                    D2ee = 0

    print(f'T2ee end: {time.perf_counter()}')
    return T2ee


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef T2ep( const double[:,:,:,:] eri, int nocc, int nvir, const double[:,:] t1e, const double[:,:,:,:] t2ee, const double[:,:,:,:] eri_ep, const double[:,:] fockp, const double[:,:] t1p, const double[:,:,:,:] t2ep, const double[:,:] fe, int pocc, int pvir):
    #compute ep amps. pavosevic jctc 2019. only done for electonic alpha-alpha, then build_ep constructs beta-beta.
    print(f'T2ep start: {time.perf_counter()}')
    cdef int i,I,a,A,c,C,K,k,d,l,ntot,ptot
    ntot = nocc + nvir
    ptot = pocc + pvir
    cdef double D2ep
    cdef double[:,:,:,:] T2ep = np.zeros((nocc,pocc,nvir,pvir),dtype=float)

    cdef double[:,:,:,:] V = np.zeros((nocc,nvir,pocc,pvir),dtype=float)
    cdef double[:,:,:,:] P = np.zeros((nocc,nocc,pocc,pocc),dtype=float)
    cdef double[:,:,:,:] Z = np.zeros((nocc,nocc,pocc,pocc),dtype=float)
    cdef double[:,:] X = np.zeros((nocc,nvir),dtype=float)
    cdef double[:,:] Y = np.zeros((pocc,pvir),dtype=float)
    cdef double[:,:,:,:] N = np.zeros((nocc,nocc,pvir,pvir),dtype=float)
    cdef double[:,:,:,:] U1 = np.zeros((nvir,nvir,pocc,pocc),dtype=float)
    cdef double[:,:,:,:] M = np.zeros((nocc,nocc,nvir,nvir),dtype=float)
    cdef double[:,:,:,:] O = np.zeros((pocc,pocc,pvir,pvir),dtype=float)

    cdef double[:,:,:] J = np.zeros((nocc,nvir,nvir),dtype=float)
    cdef double[:,:,:] G = np.zeros((nocc,nocc,nvir),dtype=float)
    cdef double[:,:,:,:] H = np.zeros((nocc,nocc,nvir,nvir),dtype=float)

    cdef double[:,:,:,:] Q = np.zeros((nocc,nvir,pocc,pvir),dtype=float)
    cdef double[:,:,:,:] S = np.zeros((nocc,nvir,pocc,pocc),dtype=float)
    cdef double[:,:,:,:] T = np.zeros((nocc,nocc,pocc,pvir),dtype=float)
    cdef double[:,:,:,:] L = np.zeros((nocc,nocc,pocc,pocc),dtype=float)
    cdef double[:,:,:,:] N1 = np.zeros((nocc,nvir,pvir,pvir),dtype=float)
    cdef double[:,:,:,:] N2 = np.zeros((nocc,nocc,pocc,pvir),dtype=float)
    cdef double[:,:,:,:] U = np.zeros((nvir,nvir,pocc,pvir),dtype=float)
    cdef double[:,:,:,:] U2 = np.zeros((nocc,nvir,pocc,pocc),dtype=float)
    cdef double[:,:,:,:] V1 = np.zeros((nocc,nvir,pocc,pocc),dtype=float)
    cdef double[:,:] X1 = np.zeros((nocc,nocc),dtype=float)
    cdef double[:,:] X2 = np.zeros((nvir,nvir),dtype=float)
    cdef double[:,:] Y1 = np.zeros((pocc,pocc),dtype=float)
    cdef double[:,:] Y2 = np.zeros((pvir,pvir),dtype=float)


    for i in range(nocc):
        for c in range(nocc,ntot):
            for A in range(pocc,ptot):
                for C in range(pocc,ptot):
                    for a in range(nocc,ntot):
                        for k in range(nocc):
                            N1[i,a-nocc,A-pocc,C-pocc] += 2*t1e[i,c-nocc]*t1e[k,a-nocc]*eri_ep[k,A,c,C]
                        for K in range(pocc):
                            N1[i,a-nocc,A-pocc,C-pocc] += eri_ep[a,K,c,C]*t2ep[i,K,c-nocc,A-pocc]
    for I in range(pocc):
        for i in range(nocc):
            for c in range(nocc,ntot):
                for A in range(pocc,ptot):
                    for K in range(pocc):
                        for k in range(nocc):
                            N2[i,k,I,A-pocc] += eri_ep[k,K,c,I]*t2ep[i,K,c-nocc,A-pocc]
        for A in range(pocc,ptot):
            for a in range(nocc,ntot):
                for c in range(nocc,ntot):
                    for k in range(nocc):
                        for C in range(pocc,ptot):
                            U[a-nocc,c-nocc,I,A-pocc] += eri_ep[k,A,c,C]*t2ep[k,I,a-nocc,C-pocc]
            for k in range(nocc):
                for i in range(nocc):
                    for c in range(nocc,ntot):
                        for C in range(pocc,ptot):
                            T[i,k,I,A-pocc] += eri_ep[k,A,c,C]*t2ep[i,I,c-nocc,C-pocc]
                for c in range(nocc,ntot):
                    for C in range(pocc,ptot):
                        Q[k,c-nocc,I,A-pocc] -= t1p[I,C-pocc]*eri_ep[k,A,c,C]
                    for K in range(pocc):
                        Q[k,c-nocc,I,A-pocc] += t1p[K,A-pocc]*eri_ep[k,K,c,I]
        for K in range(pocc):
            for i in range(nocc):
                for k in range(nocc):
                    for c in range(nocc,ntot):
                        L[i,k,I,K] += t1e[i,c-nocc]*eri_ep[k,K,c,I]
                    for C in range(pocc,ptot):
                        L[i,k,I,K] += t1p[I,C-pocc]*eri_ep[k,K,i,C]
                for a in range(nocc,ntot):
                    for k in range(nocc):
                        for c in range(nocc,ntot):
                            V1[i,a-nocc,I,K] += t1e[i,c-nocc]*t1e[k,a-nocc]*eri_ep[k,K,c,I]
                        for C in range(pocc,ptot):
                            U2[i,a-nocc,I,K] += eri_ep[k,K,i,C]*(t2ep[k,I,a-nocc,C-pocc]+2*t1e[k,a-nocc]*t1p[I,C-pocc])
                    for c in range(nocc,ntot):
                        for C in range(pocc,ptot):
                            S[i,a-nocc,I,K] += eri_ep[a,K,c,C]*(t2ep[i,I,c-nocc,C-pocc]+2*t1e[i,c-nocc]*t1p[I,C-pocc])



    for k in range(nocc):
        for c in range(nocc,ntot):
            for I in range(pocc):
                for K in range(pocc):
                    Y1[I,K] += t1e[k,c-nocc]*eri_ep[k,K,c,I]
            for A in range(pocc,ptot):
                for C in range(pocc,ptot):
                    Y2[A-pocc,C-pocc] += t1e[k,c-nocc]*eri_ep[k,A,c,C]
            for a in range(nocc,ntot):
                for d in range(nocc,ntot):
                    J[k,a-nocc,c-nocc] += (2*eri[a,k,c,d]-eri[k,a,c,d])*t1e[k,d-nocc]
                    for l in range(nocc):
                        J[k,a-nocc,c-nocc] += -0.5*(2*eri[k,l,c,d]-eri[l,k,c,d])*t2ee[k,l,a-nocc,d-nocc] -0.5*(2*eri[l,k,c,d]-eri[k,l,c,d])*t2ee[k,l,d-nocc,a-nocc] - (2*eri[k,l,c,d]-eri[l,k,c,d])*t1e[l,d-nocc]*t1e[k,a-nocc]
                        for i in range(nocc):
                            H[i,l,a-nocc,d-nocc] += (4*eri[k,l,c,d]-2*eri[l,k,c,d])*t2ee[i,k,a-nocc,c-nocc] + (eri[l,k,c,d]-2*eri[k,l,c,d])*t2ee[i,k,c-nocc,a-nocc]
            for i in range(nocc):
                for a in range(nocc,ntot):
                    for d in range(nocc,ntot):
                        H[i,k,a-nocc,c-nocc] += (2*eri[k,a,c,d]-eri[a,k,c,d])*t1e[i,d-nocc]
                        for l in range(nocc):
                            H[i,k,a-nocc,c-nocc] -= (2*eri[k,l,c,d]-eri[l,k,c,d])*t1e[i,d-nocc]*t1e[l,a-nocc]
                    for l in range(nocc):
                        H[i,k,a-nocc,c-nocc] -= (2*eri[l,k,i,c]-eri[k,l,i,c])*t1e[l,a-nocc]
                for l in range(nocc):
                    G[i,k,c-nocc] += -t1e[l,c-nocc]*(2*eri[k,l,i,c]-eri[l,k,i,c])
                    for d in range(nocc,ntot):
                        G[i,k,c-nocc] += -0.5*(2*eri[k,l,c,d]-eri[l,k,c,d])*t2ee[i,l,c-nocc,d-nocc] -0.5*(2*eri[l,k,c,d]-eri[k,l,c,d])*t2ee[i,l,d-nocc,c-nocc] - (2*eri[k,l,c,d]-eri[l,k,c,d])*t1e[i,c-nocc]*t1e[l,d-nocc]


    for K in range(pocc):
        for C in range(pocc,ptot):
            for I in range(pocc):
                for A in range(pocc,ptot):
                    for k in range(nocc):
                        for c in range(nocc,ntot):
                            O[I,K,A-pocc,C-pocc] += eri_ep[k,K,c,C]*t2ep[k,K,c-nocc,A-pocc]
            for i in range(nocc):
                for a in range(nocc,ntot):
                    for k in range(nocc):
                        for c in range(nocc,ntot):
                            M[i,k,a-nocc,c-nocc] += eri_ep[k,K,c,C]*t2ep[i,K,a-nocc,C-pocc]
            for a in range(nocc,ntot):
                for c in range(nocc,ntot):
                    X2[a-nocc,c-nocc] += t1p[K,C-pocc]*eri_ep[a,K,c,C]
            for k in range(nocc):
                for i in range(nocc):
                    X1[i,k] += t1p[K,C-pocc]*eri_ep[k,K,i,C]
                for c in range(nocc,ntot):
                    X[k,c-nocc] += eri_ep[k,K,c,C]*t1p[K,C-pocc]
                    Y[K,C-pocc] += t1e[k,c-nocc]*eri_ep[k,K,c,C]
                    for I in range(pocc):
                        for i in range(nocc):
                            P[i,k,I,K] += eri_ep[k,K,c,C]*t2ep[i,I,c-nocc,C-pocc]
                            Z[i,k,I,K] += eri_ep[k,K,c,C]*t1e[i,c-nocc]*t1p[I,C-pocc]
                        for a in range(nocc,ntot):
                            U1[a-nocc,c-nocc,I,K] += eri_ep[k,I,c,C]*t2ep[k,I,a-nocc,C-pocc]
                        for A in range(pocc,ptot):
                            V[k,c-nocc,I,A-pocc] += t1p[I,C-pocc]*t1p[K,A-pocc]*eri_ep[k,K,c,C]
                    for i in range(nocc):
                        for A in range(pocc,ptot):
                            N[i,k,A-pocc,C-pocc] += eri_ep[k,K,c,C]*t2ep[i,K,c-nocc,A-pocc]


    for i in range(nocc):
        for I in range(pocc):
            for a in range(nocc,ntot):
                for A in range(pocc,ptot):
                    T2ep[i,I,a-nocc,A-pocc] -= 2*eri_ep[a,A,i,I] #check
                    D2ep = fe[i,i] + fockp[I,I] - fe[a,a] - fockp[A,A]
                    for c in range(nocc,ntot):
                        T2ep[i,I,a-nocc,A-pocc] -= 2*eri_ep[a,A,c,I]*t1e[i,c-nocc] #check

                        T2ep[i,I,a-nocc,A-pocc] -= t2ep[i,I,c-nocc,A-pocc]*X2[a-nocc,c-nocc] # ep int
                        T2ep[i,I,a-nocc,A-pocc] += t1e[i,c-nocc]*U[a-nocc,c-nocc,I,A-pocc] # ep int
#
                        if c != a:
                            T2ep[i,I,a-nocc,A-pocc] += fe[a,c]*t2ep[i,I,c-nocc,A-pocc] #check
#                   
                        for k in range(nocc):
                            #put these two temrs in at the end...
                            T2ep[i,I,a-nocc,A-pocc] += (eri[a,k,i,c]-eri[k,a,i,c])*t2ep[k,I,c-nocc,A-pocc] #check
                            T2ep[i,I,a-nocc,A-pocc] += eri[a,k,i,c] * t2ep[k,I,c-nocc,A-pocc]
                            T2ep[i,I,a-nocc,A-pocc] -= 2*eri_ep[k,A,c,I]*(t2ee[i,k,a-nocc,c-nocc]-t2ee[i,k,c-nocc,a-nocc]) #check
                            T2ep[i,I,a-nocc,A-pocc] -= 2*eri_ep[k,A,c,I]*t2ee[i,k,a-nocc,c-nocc]
#
                            T2ep[i,I,a-nocc,A-pocc] += 2*eri_ep[k,A,c,I]*t1e[i,c-nocc]*t1e[k,a-nocc] #check
#
                            T2ep[i,I,a-nocc,A-pocc] -= fe[c,k]*t1e[k,a-nocc]*t2ep[i,I,c-nocc,A-pocc]
                            T2ep[i,I,a-nocc,A-pocc] -= fe[c,k]*t1e[i,c-nocc]*t2ep[k,I,a-nocc,A-pocc]

                            T2ep[i,I,a-nocc,A-pocc] += t1e[i,c-nocc]*t2ep[k,I,a-nocc,A-pocc]*X[k,c-nocc] # ep int
                            T2ep[i,I,a-nocc,A-pocc] += t1e[k,a-nocc]*t2ep[i,I,c-nocc,A-pocc]*X[k,c-nocc] # ep int

                            T2ep[i,I,a-nocc,A-pocc] += 2*t2ee[i,k,a-nocc,c-nocc]*V[k,c-nocc,I,A-pocc] # ep int
                            T2ep[i,I,a-nocc,A-pocc] += 2*(t2ee[i,k,a-nocc,c-nocc]-t2ee[i,k,c-nocc,a-nocc])*V[k,c-nocc,I,A-pocc]
                            T2ep[i,I,a-nocc,A-pocc] += 0.5*t2ep[i,I,c-nocc,A-pocc]*M[k,k,a-nocc,c-nocc] # ep int
                            T2ep[i,I,a-nocc,A-pocc] -= t2ep[k,I,c-nocc,A-pocc]*M[i,k,a-nocc,c-nocc] # ep int

                            T2ep[i,I,a-nocc,A-pocc] += 2*(2*t2ee[i,k,a-nocc,c-nocc]-t2ee[i,k,c-nocc,a-nocc])*Q[k,c-nocc,I,A-pocc] # ep int

                            T2ep[i,I,a-nocc,A-pocc] += t2ep[i,I,c-nocc,A-pocc]*J[k,a-nocc,c-nocc] # ep (ee) int
                            T2ep[i,I,a-nocc,A-pocc] += t2ep[k,I,a-nocc,A-pocc]*G[i,k,c-nocc] # ep (ee) int
                            T2ep[i,I,a-nocc,A-pocc] += t2ep[k,I,c-nocc,A-pocc]*H[i,k,a-nocc,c-nocc] # ep (ee) int
#
#
                    for k in range(nocc):
                        T2ep[i,I,a-nocc,A-pocc] += 2*eri_ep[k,A,i,I]*t1e[k,a-nocc] #check

                        T2ep[i,I,a-nocc,A-pocc] += t1e[k,a-nocc]*T[i,k,I,A-pocc] # ep int
                        T2ep[i,I,a-nocc,A-pocc] -= t1e[k,a-nocc]*N2[i,k,I,A-pocc] # ep int
                        T2ep[i,I,a-nocc,A-pocc] += t2ep[k,I,a-nocc,A-pocc]*X1[i,k] # ep int

#
                        if k != i:
                            T2ep[i,I,a-nocc,A-pocc] -= fe[k,i]*t2ep[k,I,a-nocc,A-pocc] #check 
##
##ep terms follow...
                    for C in range(pocc,ptot):
                        T2ep[i,I,a-nocc,A-pocc] -= 2*eri_ep[a,A,i,C]*t1p[I,C-pocc]

                        T2ep[i,I,a-nocc,A-pocc] += t1p[I,C-pocc]*N1[i,a-nocc,A-pocc,C-pocc] # ep int
                        T2ep[i,I,a-nocc,A-pocc] -= 2*t2ep[i,I,a-nocc,C-pocc]*Y2[A-pocc,C-pocc] # ep int
#
                        if C != A:
                            T2ep[i,I,a-nocc,A-pocc] += fockp[A,C]*t2ep[i,I,a-nocc,C-pocc] 
                        for K in range(pocc):
                            T2ep[i,I,a-nocc,A-pocc] += 2*eri_ep[a,K,i,C]*t1p[I,C-pocc]*t1p[K,A-pocc]
#
                            T2ep[i,I,a-nocc,A-pocc] -= fockp[K,C]*t2ep[i,I,a-nocc,C-pocc]*t1p[K,A-pocc]
                            T2ep[i,I,a-nocc,A-pocc] -= fockp[K,C]*t2ep[i,K,a-nocc,A-pocc]*t1p[I,C-pocc]

                            T2ep[i,I,a-nocc,A-pocc] += 2*t1p[I,C-pocc]*t2ep[i,K,a-nocc,A-pocc]*Y[K,C-pocc] # ep int
                            T2ep[i,I,a-nocc,A-pocc] += 2*t1p[K,A-pocc]*t2ep[i,I,a-nocc,C-pocc]*Y[K,C-pocc] # ep int

                            T2ep[i,I,a-nocc,A-pocc] += t2ep[i,I,a-nocc,C-pocc]*O[I,K,A-pocc,C-pocc] # ep int
#
#
                        for c in range(nocc,ntot):
                            T2ep[i,I,a-nocc,A-pocc] -= eri_ep[a,A,c,C]*t2ep[i,I,c-nocc,C-pocc]
                            T2ep[i,I,a-nocc,A-pocc] -= 2*eri_ep[a,A,c,C]*t1e[i,c-nocc]*t1p[I,C-pocc]
#
#
                        for k in range(nocc):
                            T2ep[i,I,a-nocc,A-pocc] += eri_ep[k,A,i,C]*t2ep[k,I,a-nocc,C-pocc]
                            T2ep[i,I,a-nocc,A-pocc] += 2*eri_ep[k,A,i,C]*t1e[k,a-nocc]*t1p[I,C-pocc]

                            T2ep[i,I,a-nocc,A-pocc] -= t1p[I,C-pocc]*t1e[k,a-nocc]*N[i,k,A-pocc,C-pocc] # ep int
                            T2ep[i,I,a-nocc,A-pocc] -= 0.5*t2ep[k,I,a-nocc,C-pocc]*N[i,k,A-pocc,C-pocc] # ep int
#
                    for K in range(pocc):
                        T2ep[i,I,a-nocc,A-pocc] += 2*eri_ep[a,K,i,I]*t1p[K,A-pocc]

                        T2ep[i,I,a-nocc,A-pocc] += t1p[K,A-pocc]*S[i,a-nocc,I,K] # ep int
                        T2ep[i,I,a-nocc,A-pocc] -= t1p[K,A-pocc]*U2[i,a-nocc,I,K] # ep int
                        T2ep[i,I,a-nocc,A-pocc] += 2*t2ep[i,K,a-nocc,A-pocc]*Y1[I,K] # ep int
                        T2ep[i,I,a-nocc,A-pocc] -= 2*t1p[K,A-pocc]*V1[i,a-nocc,I,K] # ep int
#
                        if K != I:
                            T2ep[i,I,a-nocc,A-pocc] -= fockp[K,I]*t2ep[i,K,a-nocc,A-pocc]
#
                        for c in range(nocc,ntot):
                            T2ep[i,I,a-nocc,A-pocc] += eri_ep[a,K,c,I]*t2ep[i,K,c-nocc,A-pocc]
                            T2ep[i,I,a-nocc,A-pocc] += 2*eri_ep[a,K,c,I]*t1e[i,c-nocc]*t1p[K,A-pocc]

                            T2ep[i,I,a-nocc,A-pocc] -= t1e[i,c-nocc]*t1p[K,A-pocc]*U1[a-nocc,c-nocc,I,K] # ep int
#
#
                        for k in range(nocc):
                            T2ep[i,I,a-nocc,A-pocc] -= eri_ep[k,K,i,I]*t2ep[k,K,a-nocc,A-pocc]
                            T2ep[i,I,a-nocc,A-pocc] -= 2*eri_ep[k,K,i,I]*t1e[k,a-nocc]*t1p[K,A-pocc]

                            T2ep[i,I,a-nocc,A-pocc] -= 2*t1e[k,a-nocc]*t1p[K,A-pocc]*Z[i,k,I,K] # ep int
                            T2ep[i,I,a-nocc,A-pocc] -= t2ep[k,K,a-nocc,A-pocc]*Z[i,k,I,K] # ep int

                            T2ep[i,I,a-nocc,A-pocc] -= t1e[k,a-nocc]*t1p[K,A-pocc]*P[i,k,I,K] # ep int
                            T2ep[i,I,a-nocc,A-pocc] -= 0.5*t2ep[k,K,a-nocc,A-pocc]*P[i,k,I,K] # ep int

                            T2ep[i,I,a-nocc,A-pocc] += 0.5*t2ep[k,I,a-nocc,A-pocc]*P[i,k,K,K] # ep int
                            T2ep[i,I,a-nocc,A-pocc] += t2ep[i,K,a-nocc,A-pocc]*P[k,k,I,K] # ep int
                            T2ep[i,I,a-nocc,A-pocc] -= t2ep[k,K,a-nocc,A-pocc]*L[i,k,I,K] # ep int

#
                    T2ep[i,I,a-nocc,A-pocc] /= D2ep
                    D2ep = 0


    print(f'T2ep end: {time.perf_counter()}')
    return T2ep


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef mc_ecc(const double[:,:] fe, const double[:,:,:,:] eri, int nocc, int nvir, const double[:,:] t1e, const double[:,:,:,:] t2ee, const double[:,:] fockp, const double[:,:,:,:] eri_ep, const double[:,:,:,:] t2ep, const double[:,:] t1p, int pocc, int pvir):
    #compute mc-ecc energy. 
    print(f'MC ECC start: {time.perf_counter()}')
    cdef int C,K,c,k,d,l,ntot,ptot
    ntot = nocc + nvir
    ptot = pocc + pvir
    cdef double mc_ecc, elec_e, mc_e

    mc_ecc = 0
    elec_e = 0
    mc_e = 0
    for C in range(pocc,ptot):
        for K in range(pocc):
            mc_ecc += 2*fockp[K,C]*t1p[K,C-pocc]
            mc_e += 2*fockp[K,C]*t1p[K,C-pocc]

            for c in range(nocc,ntot):
                for k in range(nocc):
                    mc_ecc -= eri_ep[k,K,c,C]*t2ep[k,K,c-nocc,C-pocc] 
                    mc_ecc -= 2*eri_ep[k,K,c,C]*t1e[k,c-nocc]*t1p[K,C-pocc]
                    mc_e -= eri_ep[k,K,c,C] * ( t2ep[k,K,c-nocc,C-pocc] + 2*t1e[k,c-nocc]*t1p[K,C-pocc] )

    for c in range(nocc,ntot):
        for k in range(nocc):
            mc_ecc += 2*fe[k,c]*t1e[k,c-nocc]
            elec_e += 2*fe[k,c]*t1e[k,c-nocc]

            for d in range(nocc,ntot):
                for l in range(nocc):
                    mc_ecc += (2*eri[k,l,c,d]-eri[l,k,c,d])*t2ee[k,l,c-nocc,d-nocc]
                    mc_ecc += (2*eri[k,l,c,d]-eri[l,k,c,d])*t1e[k,c-nocc]*t1e[l,d-nocc]
                    elec_e += (2*eri[k,l,c,d]-eri[l,k,c,d])*(t2ee[k,l,c-nocc,d-nocc]+t1e[k,c-nocc]*t1e[l,d-nocc])
    print(f'MC ECC end: {time.perf_counter()}')
    print(f'Purely electronic energy: {elec_e}')
    print(f'Purely mc energy: {mc_e}')
    return mc_ecc


def t_rmsd( nso, t1, t1n, t2, t2n):
    #compute rmsd of cluster amps

    t1_rmsd = 0
    t2_rmsd = 0
    t1_msd = 0
    t2_msd = 0
    for i in range(nso):
        for j in range(nso):
            t1_msd += ( t1n[i,j] - t1[i,j] ) **2
            for k in range(nso):
                for l in range(nso):
                    t2_msd += ( t2n[i,j,k,l] - t2[i,j,k,l] ) **2
    t1_rmsd = np.sqrt(t1_msd)
    t2_rmsd = np.sqrt(t2_msd)
    return t1_rmsd, t2_rmsd


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef run_ccsd(const double[:,:] fe, double scf_e, const double[:,:] t1ei, const double[:,:] t1pi, const double[:,:,:,:] t2eei, const double[:,:,:,:] t2epi, const double[:,:,:,:] eri, const double[:,:,:,:] eri_ep, const double[:,:] pfock, int nocc, int nvir, int pocc, int pvir, conv=1e-11):
    #run mc-ccsd calculation.

    cdef int ntot,ptot
    ntot = nocc + nvir
    ptot = pocc + pvir
    cdef double current_ecc, decc
    cdef double[:,:] t1e = np.zeros((nocc,nvir), dtype=float)
    cdef double[:,:] t1p = np.zeros((pocc,pvir), dtype=float)
    cdef double[:,:,:,:] t2ee = np.zeros((nocc,nocc,nvir,nvir), dtype=float)
    cdef double[:,:,:,:] t2ep = np.zeros((nocc,pocc,nvir,pvir), dtype=float)

    cdef double[:,:] t1ne = np.zeros((nocc,nvir), dtype=float)
    cdef double[:,:] t1np = np.zeros((pocc,pvir), dtype=float)
    cdef double[:,:,:,:] t2nee = np.zeros((nocc,nocc,nvir,nvir), dtype=float)
    cdef double[:,:,:,:] t2nep = np.zeros((nocc,pocc,nvir,pvir), dtype=float)


    itr = 0 
    current_ecc = 0
    decc = 1.0
    dice = mc_cc_diis(ntot,ptot,nocc,pocc)
    t1ei, t2eei,t1pi,t2epi = dice.run_diis(t1ei,t2eei,t1pi,t2epi)
    if itr == 0:
        tau,T = eff_dubs_e(nocc,nvir,t1ei,t2eei)
        hab,hui,hai,gui,gab,Ai,Bi,J,K = ee_intermediates(fe,eri,nocc,nvir,t1ei,t2eei,tau,T) 
        t1e = T1e(fe,eri,nocc,nvir,t1ei,t2eei,hab,hui,hai,eri_ep,pfock,pocc,pvir,t1pi,t2epi,tau)
        t2ee = T2ee(eri,nocc,nvir,t1ei,t2eei,gui,gab,Ai,Bi,J,K,fe,eri_ep,pfock,t1pi,t2epi,pocc,pvir,tau)
        t1p = T1p(fe,eri,nocc,nvir,t1ei,t2eei,eri_ep,pfock,pocc,pvir,t1pi,t2epi)
        t2ep = T2ep(eri,nocc,nvir,t1ei,t2eei,eri_ep,pfock,t1pi,t2epi,fe,pocc,pvir)
        current_ecc = mc_ecc(fe,eri,nocc,nvir,t1e,t2ee,pfock,eri_ep,t2ep,t1p,pocc,pvir)
        print('Iter    Eccsd                    Decc')
        print(f'1      {current_ecc}')
        t1e,t2ee,t1p,t2ep = dice.run_diis(t1e,t2ee,t1p,t2ep)
        itr = 1
    while decc > conv:
        old_ecc = current_ecc
        tau,T = eff_dubs_e(nocc,nvir,t1e,t2ee)
        hab,hui,hai,gui,gab,Ai,Bi,J,K = ee_intermediates(fe,eri,nocc,nvir,t1e,t2ee,tau,T) 
        t1ne = T1e(fe,eri,nocc,nvir,t1e,t2ee,hab,hui,hai,eri_ep,pfock,pocc,pvir,t1p,t2ep,tau)
        t2nee = T2ee(eri,nocc,nvir,t1e,t2ee,gui,gab,Ai,Bi,J,K,fe,eri_ep,pfock,t1p,t2ep,pocc,pvir,tau)
        t1np = T1p(fe,eri,nocc,nvir,t1e,t2ee,eri_ep,pfock,pocc,pvir,t1p,t2ep)
        t2nep = T2ep(eri,nocc,nvir,t1e,t2ee,eri_ep,pfock,t1p,t2ep,fe,pocc,pvir)
        current_ecc = mc_ecc(fe,eri,nocc,nvir,t1ne,t2nee,pfock,eri_ep,t2nep,t1np,pocc,pvir)
        decc = abs(current_ecc - old_ecc)
        itr += 1
        print(f'{itr}      {current_ecc}      {decc}')
        t1e,t2ee,t1p,t2ep = dice.run_diis(t1ne,t2nee,t1np,t2nep)
#        t1e,t2ee,t1p,t2ep = t1ne,t2nee,t1np,t2nep

    print(f'CCSD energy converged with Eccsd = {current_ecc}')
    print(f'...And total energy = {current_ecc + scf_e}')

    return current_ecc, t1ne, t1np, t2nee, t2nep 
