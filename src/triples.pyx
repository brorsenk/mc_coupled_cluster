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
cimport cython

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef triples_correction(int e_nocc, int e_nvir, int p_nocc, int p_nvir, double[:,:,:,:] t2_ee, double[:,:,:,:] t2_ep, double[:,:] t1e, double[:,:] t1p, double[:,:,:,:] eri_ee_ovvv, double[:,:,:,:] eri_ee_ooov, double[:,:,:,:] eri_ee_ovov, double[:,:,:,:] eri_ep_ooOV, double[:,:,:,:] eri_ep_vvOV, double[:,:,:,:] eri_ep_ovOO, double[:,:,:,:] eri_ep_ovVV, double[:,:,:,:] eri_ep_ovOV, double[:,:] eia, double[:,:] ejb):
 cdef double bracket_t = 0.0
 cdef double parenth_t_1 = 0.0
 cdef double parenth_t_2 = 0.0
 cdef int i,j,a,b,I,A
 cdef double term_holder_1, term_holder_2, term_holder_3
 cdef double temp


 for i in range(e_nocc):
  for j in range(e_nocc):
   for a in range(e_nvir):
    for b in range(e_nvir):
     for I in range(p_nocc):
      for A in range(p_nvir):

       term_holder_1 = 0.0
       term_holder_2 = 0.0
       term_holder_3 = 0.0 

       #aaaa
       temp = 0.0
       for k in range(e_nocc):
         temp += -(1.0 * (t2_ee[i,k,a,b] - t2_ee[i,k,b,a] ) * -eri_ep_ooOV[k,j,I,A] )

         temp += (1.0 * (t2_ee[j,k,a,b] - t2_ee[j,k,b,a] ) * -eri_ep_ooOV[k,i,I,A] )

         temp += -(1.0 * eri_ee_ooov[k,i,j,b] * t2_ep[k,a,I,A] )
         temp += -(-1.0 * eri_ee_ooov[k,j,i,b] * t2_ep[k,a,I,A] )

         temp += (1.0 * eri_ee_ooov[k,i,j,a] * t2_ep[k,b,I,A] )
         temp += (-1.0 * eri_ee_ooov[k,j,i,a] * t2_ep[k,b,I,A] )

       for c in range(e_nvir):
         temp += (1.0 * (t2_ee[i,j,a,c] - t2_ee[i,j,c,a] ) * -eri_ep_vvOV[b,c,I,A]) 

         temp += -(1.0 * (t2_ee[i,j,b,c] - t2_ee[i,j,c,b] ) * -eri_ep_vvOV[a,c,I,A]) 

         temp += (1.0 * eri_ee_ovvv[j,b,c,a] * t2_ep[i,c,I,A]  ) 
         temp += (-1.0 * eri_ee_ovvv[j,a,c,b] * t2_ep[i,c,I,A]  ) 

         temp += -(1.0 * eri_ee_ovvv[i,b,c,a] * t2_ep[j,c,I,A]  ) 
         temp += -(-1.0 * eri_ee_ovvv[i,a,c,b] * t2_ep[j,c,I,A]  ) 

       for J in range(p_nocc):
         #1
         temp += -(1.0 * -eri_ep_ovOO[j,b,I,J] * t2_ep[i,a,J,A]  ) 
         #2
         temp += (1.0 * -eri_ep_ovOO[j,a,I,J] * t2_ep[i,b,J,A]  ) 
         #3
         temp += (1.0 * -eri_ep_ovOO[i,b,I,J] * t2_ep[j,a,J,A]  ) 
         #4
         temp += -(1.0 * -eri_ep_ovOO[i,a,I,J] * t2_ep[j,b,J,A]  ) 


       for B in range(p_nvir):
         #1
         temp += (1.0 * -eri_ep_ovVV[j,b,A,B] * t2_ep[i,a,I,B]  ) 
         #2
         temp += -(1.0 * -eri_ep_ovVV[i,b,A,B] * t2_ep[j,a,I,B]  ) 
         #3
         temp += -(1.0 * -eri_ep_ovVV[j,a,A,B] * t2_ep[i,b,I,B]  ) 
         #4
         temp += (1.0 * -eri_ep_ovVV[i,a,A,B] * t2_ep[j,b,I,B]  ) 

       term_holder_1 += temp**2
       term_holder_2 += t1e[i,a] * -eri_ep_ovOV[j,b,I,A] * temp
       term_holder_3 += t1p[I,A] * (eri_ee_ovov[i,a,j,b] - eri_ee_ovov[i,b,j,a]) * temp

       #abab
       temp = 0.0
       for k in range(e_nocc):
         temp += -(1.0 * t2_ee[i,k,a,b] * -eri_ep_ooOV[k,j,I,A] )

         temp += (-1.0 * t2_ee[j,k,b,a] * -eri_ep_ooOV[k,i,I,A] )

         temp += -(1.0 * eri_ee_ooov[k,i,j,b] * t2_ep[k,a,I,A] )

         temp += (-1.0 * eri_ee_ooov[k,j,i,a] * t2_ep[k,b,I,A] )

       for c in range(e_nvir):
         temp += (1.0 * t2_ee[i,j,a,c] * -eri_ep_vvOV[b,c,I,A] )

         temp += -(-1.0 * t2_ee[i,j,c,b] * -eri_ep_vvOV[a,c,I,A] )

         temp += (1.0 * eri_ee_ovvv[j,b,c,a] * t2_ep[i,c,I,A]  ) 

         temp += -(-1.0 * eri_ee_ovvv[i,a,c,b] * t2_ep[j,c,I,A]  ) 

       for J in range(p_nocc):
         #1
         temp += -(1.0 * -eri_ep_ovOO[j,b,I,J] * t2_ep[i,a,J,A]  ) 
         #2
         #3
         #4
         temp += -(1.0 * -eri_ep_ovOO[i,a,I,J] * t2_ep[j,b,J,A]  ) 

       for B in range(p_nvir):
         #1
         temp += (1.0 * -eri_ep_ovVV[j,b,A,B] * t2_ep[i,a,I,B]  ) 
         #2
         #3
         #4
         temp += (1.0 * -eri_ep_ovVV[i,a,A,B] * t2_ep[j,b,I,B]  ) 

       term_holder_1 += temp**2
       term_holder_2 += t1e[i,a] * -eri_ep_ovOV[j,b,I,A] * term_holder_2
       term_holder_3 += t1p[I,A] * (eri_ee_ovov[i,a,j,b] ) * temp

       #abba
       temp = 0.0
       for k in range(e_nocc):
         temp += -(-1.0 * t2_ee[i,k,b,a] * -eri_ep_ooOV[k,j,I,A] )

         temp += (1.0 * t2_ee[j,k,a,b] * -eri_ep_ooOV[k,i,I,A] )

         temp += -(-1.0 * eri_ee_ooov[k,j,i,b] * t2_ep[k,a,I,A] )

         temp += (1.0 * eri_ee_ooov[k,i,j,a] * t2_ep[k,b,I,A] )

       for c in range(e_nvir):
         temp += (-1.0 * t2_ee[i,j,c,a] * -eri_ep_vvOV[b,c,I,A] )

         temp += -(1.0 * t2_ee[i,j,b,c] * -eri_ep_vvOV[a,c,I,A] )

         temp += (-1.0 * eri_ee_ovvv[j,a,c,b] * t2_ep[i,c,I,A]  ) 

         temp += -(1.0 * eri_ee_ovvv[i,b,c,a] * t2_ep[j,c,I,A]  ) 

       for J in range(p_nocc):
         #1
         #2
         temp += (1.0 * -eri_ep_ovOO[j,a,I,J] * t2_ep[i,b,J,A]  ) 
         #3
         temp += (1.0 * -eri_ep_ovOO[i,b,I,J] * t2_ep[j,a,J,A]  ) 
         #4

       for B in range(p_nvir):
         #1
         #2
         temp += -(1.0 * -eri_ep_ovVV[i,b,A,B] * t2_ep[j,a,I,B]  ) 
         #3
         temp += -(1.0 * -eri_ep_ovVV[j,a,A,B] * t2_ep[i,b,I,B]  ) 
         #4

       term_holder_1 += temp**2
       term_holder_3 += t1p[I,A] * ( - eri_ee_ovov[i,b,j,a]) * temp

       #baab
       temp = 0.0
       for k in range(e_nocc):
         temp += -(-1.0 * t2_ee[i,k,b,a] * -eri_ep_ooOV[k,j,I,A] )

         temp += (1.0 * t2_ee[j,k,a,b] * -eri_ep_ooOV[k,i,I,A]) 

         temp += -(-1.0 * eri_ee_ooov[k,j,i,b] * t2_ep[k,a,I,A] )

         temp += (1.0 * eri_ee_ooov[k,i,j,a] * t2_ep[k,b,I,A] )

       for c in range(e_nvir):
         temp += (-1.0 * t2_ee[i,j,c,a] * -eri_ep_vvOV[b,c,I,A] )

         temp += -(1.0 * t2_ee[i,j,b,c] * -eri_ep_vvOV[a,c,I,A] )

         temp += (-1.0 * eri_ee_ovvv[j,a,c,b] * t2_ep[i,c,I,A]  ) 

         temp += -(1.0 * eri_ee_ovvv[i,b,c,a] * t2_ep[j,c,I,A]  ) 

       for J in range(p_nocc):
         #1
         #2
         temp += (1.0 * -eri_ep_ovOO[j,a,I,J] * t2_ep[i,b,J,A]  ) 
         #3
         temp += (1.0 * -eri_ep_ovOO[i,b,I,J] * t2_ep[j,a,J,A]  ) 
         #4

       for B in range(p_nvir):
         #1
         #2
         temp += -(1.0 * -eri_ep_ovVV[i,b,A,B] * t2_ep[j,a,I,B]  ) 
         #3
         temp += -(1.0 * -eri_ep_ovVV[j,a,A,B] * t2_ep[i,b,I,B]  ) 
         #4
 
       term_holder_1 += temp**2
       term_holder_3 += t1p[I,A] * (- eri_ee_ovov[i,b,j,a]) * temp

       #baba
       temp = 0.0
       for k in range(e_nocc):
         temp += -(1.0 * t2_ee[i,k,a,b] * -eri_ep_ooOV[k,j,I,A] )

         temp += (-1.0 * t2_ee[j,k,b,a] * -eri_ep_ooOV[k,i,I,A]) 

         temp += -(1.0 * eri_ee_ooov[k,i,j,b] * t2_ep[k,a,I,A] )
         
         temp += (-1.0 * eri_ee_ooov[k,j,i,a] * t2_ep[k,b,I,A] )

       for c in range(e_nvir):
         temp += (1.0 * t2_ee[i,j,a,c] * -eri_ep_vvOV[b,c,I,A] )

         temp += -(-1.0 * t2_ee[i,j,c,b] * -eri_ep_vvOV[a,c,I,A] )

         temp += (1.0 * eri_ee_ovvv[j,b,c,a] * t2_ep[i,c,I,A]  ) 

         temp += -(-1.0 * eri_ee_ovvv[i,a,c,b] * t2_ep[j,c,I,A]  ) 

       for J in range(p_nocc):
         #1
         temp += -(1.0 * -eri_ep_ovOO[j,b,I,J] * t2_ep[i,a,J,A]  ) 
         #2
         #3
         #4
         temp += -(1.0 * -eri_ep_ovOO[i,a,I,J] * t2_ep[j,b,J,A]  ) 


       for B in range(p_nvir):
         #1
         temp += (1.0 * -eri_ep_ovVV[j,b,A,B] * t2_ep[i,a,I,B]  ) 
         #2
         #3
         #4
         temp += (1.0 * -eri_ep_ovVV[i,a,A,B] * t2_ep[j,b,I,B]  ) 

       term_holder_1 += temp**2
       term_holder_2 += t1e[i,a] * -eri_ep_ovOV[j,b,I,A] * temp
       term_holder_3 += t1p[I,A] * (eri_ee_ovov[i,a,j,b] ) * temp

       #bbbb
       temp = 0.0
       for k in range(e_nocc):
         temp += -(1.0 * (t2_ee[i,k,a,b] - t2_ee[i,k,b,a] ) * -eri_ep_ooOV[k,j,I,A] )

         temp += (1.0 * (t2_ee[j,k,a,b] - t2_ee[j,k,b,a] ) * -eri_ep_ooOV[k,i,I,A]) 

         temp += -(1.0 * eri_ee_ooov[k,i,j,b] * t2_ep[k,a,I,A] )
         temp += -(-1.0 * eri_ee_ooov[k,j,i,b] * t2_ep[k,a,I,A] )

         temp += (1.0 * eri_ee_ooov[k,i,j,a] * t2_ep[k,b,I,A] )
         temp += (-1.0 * eri_ee_ooov[k,j,i,a] * t2_ep[k,b,I,A]) 

       for c in range(e_nvir):
         temp += (1.0 * (t2_ee[i,j,a,c] - t2_ee[i,j,c,a] ) * -eri_ep_vvOV[b,c,I,A] )

         temp += -(1.0 * (t2_ee[i,j,b,c] - t2_ee[i,j,c,b] ) * -eri_ep_vvOV[a,c,I,A] )

         temp += (1.0 * eri_ee_ovvv[j,b,c,a] * t2_ep[i,c,I,A]  ) 
         temp += (-1.0 * eri_ee_ovvv[j,a,c,b] * t2_ep[i,c,I,A]  ) 

         temp += -(1.0 * eri_ee_ovvv[i,b,c,a] * t2_ep[j,c,I,A]  ) 
         temp += -(-1.0 * eri_ee_ovvv[i,a,c,b] * t2_ep[j,c,I,A]  ) 

       for J in range(p_nocc):
         #1
         temp += -(1.0 * -eri_ep_ovOO[j,b,I,J] * t2_ep[i,a,J,A]  ) 
         #2
         temp += (1.0 * -eri_ep_ovOO[j,a,I,J] * t2_ep[i,b,J,A]  ) 
         #3
         temp += (1.0 * -eri_ep_ovOO[i,b,I,J] * t2_ep[j,a,J,A]  ) 
         #4
         temp += -(1.0 * -eri_ep_ovOO[i,a,I,J] * t2_ep[j,b,J,A]  ) 

       for B in range(p_nvir):
         #1
         temp += (1.0 * -eri_ep_ovVV[j,b,A,B] * t2_ep[i,a,I,B]  ) 
         #2
         temp += -(1.0 * -eri_ep_ovVV[i,b,A,B] * t2_ep[j,a,I,B]  ) 
         #3
         temp += -(1.0 * -eri_ep_ovVV[j,a,A,B] * t2_ep[i,b,I,B]  ) 
         #4
         temp += (1.0 * -eri_ep_ovVV[i,a,A,B] * t2_ep[j,b,I,B]  ) 

       term_holder_1 += temp**2
       term_holder_2 += t1e[i,a] * -eri_ep_ovOV[j,b,I,A] * temp
       term_holder_3 += t1p[I,A] * (eri_ee_ovov[i,a,j,b] - eri_ee_ovov[i,b,j,a]) * temp

       term_holder_1 = term_holder_1 / (eia[i,a] + eia[j,b] + ejb[I,A])
       bracket_t += term_holder_1

       parenth_t_1 += term_holder_2 / (eia[i,a] + eia[j,b] + ejb[I,A])
       parenth_t_2 += term_holder_3 / (eia[i,a] + eia[j,b] + ejb[I,A])

 bracket_t = bracket_t/4.0
 parenth_t_1 = parenth_t_1
 parenth_t_2 = parenth_t_2/4.0

 print('[T] correction = ', bracket_t)
 print('(T) term 1 = ',parenth_t_1)
 print('(T) term 2 = ',parenth_t_2)
 print('(T) correction = ',bracket_t+parenth_t_1+parenth_t_2)

