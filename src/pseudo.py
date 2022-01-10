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

def pseudocanonicalize(e_mo_coeff, fock_ee, e_nocc, p_mo_coeff, fock_pp, p_nocc):

# get psuedo-canonical orbitals

# diagonalize elec occ-occ block and transform elec occ orbitals to pseudo-canonical form
  w, v = np.linalg.eigh(fock_ee[:e_nocc,:e_nocc])
  e_mo_coeff[:,:e_nocc] = np.matmul(e_mo_coeff[:,:e_nocc],v)
# diagonalize elec vir-vir block and transform elec vir orbitals to pseudo-canonical form
  w, v = np.linalg.eigh(fock_ee[e_nocc:,e_nocc:])
  e_mo_coeff[:,e_nocc:] = np.matmul(e_mo_coeff[:,e_nocc:],v)

# diagonalize prot occ-occ block and transform elec occ orbitals to pseudo-canonical form
#  w, v = np.linalg.eigh(fock_pp[:p_nocc,:p_nocc])
#  p_mo_coeff[:,:p_nocc] = np.matmul(p_mo_coeff[:,:p_nocc],v)
# diagonalize prot vir-vir block and transform elec vir orbitals to pseudo-canonical form
  w, v = np.linalg.eigh(fock_pp[p_nocc:,p_nocc:])
  p_mo_coeff[:,p_nocc:] = np.matmul(p_mo_coeff[:,p_nocc:],v)

  return e_mo_coeff, p_mo_coeff
