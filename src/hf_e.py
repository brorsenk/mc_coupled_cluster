#!/usr/bin/env python3

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

def hf_energy(eri_ee_full, eri_ep_full, mo_fock_1e, mo_fock_1p, e_nocc, p_nocc): 

  e_hf = 0.0

  for i in range(e_nocc):
    e_hf += 2*mo_fock_1e[i,i]
    for j in range(e_nocc):
      e_hf += (2*eri_ee_full[i,i,j,j] - eri_ee_full[i,j,i,j])

# need to subtract this due to double counting by using ee and pp fock matrices
    for j in range(p_nocc):
      e_hf -= 2*eri_ep_full[i,i,j,j]

  for i in range(p_nocc):
    e_hf += mo_fock_1p[i,i]

  return e_hf
