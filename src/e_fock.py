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

def construct_fock(eri_ee_full, eri_ep_full, mo_fock_1e, mo_fock_1p, e_tot, e_nocc, p_tot, p_nocc): 
  fock_ee = np.zeros((e_tot,e_tot),dtype = float)
  fock_pp = np.zeros((p_tot,p_tot),dtype = float)

  for i in range(e_tot):
   for k in range(e_tot):
     fock_ee[i,k] += mo_fock_1e[i,k]
     for j in range(e_nocc):
       fock_ee[i,k] += 2*eri_ee_full[i,k,j,j] - eri_ee_full[i,j,k,j]
     for j in range(p_nocc):
       fock_ee[i,k] -= eri_ep_full[i,k,j,j]

  for i in range(p_tot):
    for k in range(p_tot):
      fock_pp[i,k] += mo_fock_1p[i,k]
#      for j in range(p_nocc):
#        fock_pp[i,k] += eri_pp_full[i,k,j,j] - eri_pp_full[i,j,k,j]
      for j in range(e_nocc):
        fock_pp[i,k] -= 2.0 * eri_ep_full[j,j,i,k]

  return fock_ee, fock_pp

def construct_fock_ao(ee_ints, ep_ints, e1_ints, p1_ints, e_den, p_den, e_mo_coeff, p_mo_coeff, e_tot, p_tot):

  fock_ee_ao = np.zeros((e_tot,e_tot),dtype = float)
  fock_pp_ao = np.zeros((p_tot,p_tot),dtype = float)

  fock_ee = np.zeros((e_tot,e_tot),dtype = float)
  fock_pp = np.zeros((p_tot,p_tot),dtype = float)

# construct ao_fock matrices
  for i in range(e_tot):
    for j in range(e_tot):
      fock_ee_ao[i,j] += e1_ints[i,j]
      for k in range(e_tot):
        for l in range(e_tot):
          fock_ee_ao[i,j] += 0.5 * e_den[k,l] * (2*ee_ints[i,j,k,l] - ee_ints[i,k,j,l])
      for k in range(p_tot):
        for l in range(p_tot):
          fock_ee_ao[i,j] -= 1.0 * p_den[k,l] * ep_ints[i,j,e_tot+k,e_tot+l]  

  for i in range(p_tot):
    for j in range(p_tot):
      fock_pp_ao[i,j] += p1_ints[i,j]
      for k in range(e_tot):
        for l in range(e_tot):
          fock_pp_ao[i,j] -= 1.0 * e_den[k,l] * ep_ints[k,l,i+e_tot,j+e_tot]

# convert to mo basis

  fock_ee =  np.matmul(e_mo_coeff.T, np.matmul(fock_ee_ao, e_mo_coeff))
  fock_pp =  np.matmul(p_mo_coeff.T, np.matmul(fock_pp_ao, p_mo_coeff))

  return fock_ee, fock_pp
