
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

import numpy
from pyscf import gto,scf,qmmm,lib,df
import numpy.linalg
from functools import reduce
import sys

### contruct a set of mol objects for electrons and proton
###
### Input: 
###       coords -
###       ebasis -
###       proton_num -
###       charge
###       nuc_mass
###
### Returns:
###          mol_elec
###          mol_nuc
###          mol_combined

def makemol(coords,basis_dict, proton_num,charge,nuc_mass,positron, cart_basis):

  basis_dict['ghost-H'] = gto.basis.load(basis_dict[coords[proton_num][0]],'H')

  #PsH edits (OJF, 1/27/21)
  coords1 = 'H 0,0,0'
  charge1 = charge-1


#  mol  = gto.M(atom=coords1,basis=basis_dict,charge=charge1,cart=cart_basis)

  if positron:
    mol  = gto.M(atom=coords,basis=basis_dict,charge=charge1,cart=cart_basis)
  else:
    mol = gto.M(atom=coords,basis=basis_dict,charge=charge,cart=cart_basis)

  #mol.unit = 'Angstrom'
  mf = scf.RHF(mol)
  print('\n Running standard Hartree-Fock')
  mf.kernel()

  print('\n Starting MC calculation \n')
  eldm = mf.make_rdm1(mf.mo_coeff,mf.mo_occ)

  if(positron):
#     mol_nuc = gto.M(atom=[['ghost',coords[proton_num][1]]],basis={'ghost':gto.basis.load(h_basis,'H')},cart=cart_basis)
     mol_nuc = gto.M(atom=[['ghost',coords[proton_num][1]]],basis={'ghost':gto.basis.parse('''
H   S
    8.9346 1.0000  
H   S
    1.5904 1.0000 
H   S
    0.51619 1.0000
H   S
    0.11092 1.0000
H   S
    0.052472 1.0000
H   S
    0.024562 1.0000
H   P
    0.47831 1.0000
H   P
    0.18726 1.0000
H   P
    0.069729 1.0000
H   D
    0.15439 1.0000
    ''')},cart=cart_basis)
  else:
    charge = charge-1
    coords[proton_num][0] = 'ghost-H'
    mol_nuc = gto.M(atom=[['ghost',coords[proton_num][1]]],basis={'ghost':gto.basis.parse('''
H   S
    5.973 1.0000
H   S
    10.645 1.00000
H    S
    17.943 1.00000
H   S
    28.950 1.0000
H   P
    7.604   1.0000
H   P
    14.701   1.0000
H   P
    23.308  1.0000
H   D
    9.011   1.0000
H   D
    19.787   1.0000
''')},cart=cart_basis)

  if positron:
    mol_elec = gto.M(atom=coords,basis=basis_dict,charge=charge1, cart=cart_basis)
  else:
    mol_elec = gto.M(atom=coords,basis=basis_dict,charge=charge,cart=cart_basis)

  print(f'num elec: {mol_elec.nelectron}')
  print(f'elec nao: {mol_elec.nao_nr()}')
  print(f'num pstrn: {mol_nuc.nelectron}')
  print(f'pstrn nao: {mol_nuc.nao_nr()}')
  mol_combined = mol_elec + mol_nuc

  nudm = get_nuc_guess(mol_nuc,nuc_mass,mol_elec,mol_combined,eldm)

  return mol_elec,mol_nuc,mol_combined,eldm,nudm

### contruct nuclear guess from 1e Hamiltonian
###
### Input:
###       mol_nuc
###       nuc_mass
###       mol_elec
###
### Returns:
###          dma 

def get_nuc_guess(mol_nuc,nuc_mass,mol_elec,mol_tot,eldm):

  nuc_core = get_nuc_core(mol_nuc,nuc_mass,mol_elec)
  s1n = scf.hf.get_ovlp(mol_nuc)

  if mol_nuc.cart:
      intor = 'int2e_cart'
  else:
      intor = 'int2e_sph'

  mixed_ints = mol_tot.intor(intor, shls_slice=(0, mol_elec.nbas, 0 ,mol_elec.nbas, mol_elec.nbas, mol_tot.nbas, mol_elec.nbas, mol_tot.nbas))
  mixed_ints = mixed_ints.reshape(mol_elec.nao_nr(),mol_elec.nao_nr(),mol_nuc.nao_nr(),mol_nuc.nao_nr())
  j_mixed_nuc = numpy.einsum('ijkl,ij->kl',mixed_ints,eldm)

  nuc_core = nuc_core-j_mixed_nuc

#  mo_energy, mo_coeff = scf.hf.eig(nuc_core,s1n)
  mo_energy, mo_coeff = eigmc(nuc_core,s1n)
  mo_occ = numpy.zeros(len(mo_energy))
  mo_occ[0]=2

  dm = scf.hf.make_rdm1(mo_coeff,mo_occ)
  dma = dm*.5

  return dma

### contruct nuclear 1e Hamiltonian
###
### Input:
###       mol_nuc
###       nuc_mass
###       mol_elec
###
### Returns:
###         nuc_core 

def get_nuc_core(mol_nuc,nuc_mass,mol_elec):

  charges = []
  for atm in range(len(mol_elec._atm)):
    charges.append(mol_elec.atom_charge(atm))
  coords = mol_elec.atom_coords()
  mf = qmmm.mm_charge(scf.UHF(mol_nuc),coords*lib.parameters.BOHR,charges)
  coords = numpy.array(coords,order='C')
  fakemol = gto.fakemol_for_charges(coords)

  if mol_nuc.cart:
      intor = 'int3c2e_cart'
  else:
      intor = 'int3c2e_sph'
  j3c = df.incore.aux_e2(mol_nuc, fakemol, intor=intor, aosym='s2ij')

  if mol_nuc.cart:
      intor = 'int1e_kin_cart'
  else:
      intor = 'int1e_kin_sph'
  
  nuc_repul = lib.unpack_tril(numpy.einsum('xk,k->x', j3c,charges))
  nuc_kin = mol_nuc.intor(intor)/nuc_mass
#  nuc_kin = mol_nuc.intor('int1e_kin_sph')/nuc_mass
  nuc_core = nuc_kin + nuc_repul

  return nuc_core

def get_vhf_mc(mol_elec,mol_nuc,mol_tot,dm_e,dm_p,dm_e_last=None,dm_p_last=None):
  j_el, k_el = scf.hf.get_jk(mol=mol_elec,dm=dm_e)
  j_nu, k_nu  = scf.hf.get_jk(mol=mol_nuc,dm=dm_p)
  if mol_nuc.cart:
      intor = 'int2e_cart'
  else:
      intor = 'int2e_sph'
  mixed_ints = mol_tot.intor(intor, shls_slice=(0, mol_elec.nbas, 0 ,mol_elec.nbas, mol_elec.nbas, mol_tot.nbas, mol_elec.nbas, mol_tot.nbas))
  mixed_ints = mixed_ints.reshape(mol_elec.nao_nr(),mol_elec.nao_nr(),mol_nuc.nao_nr(),mol_nuc.nao_nr())
  j_mixed_nuc = numpy.einsum('ijkl,ij->kl',mixed_ints,dm_e)
  j_mixed_elec=numpy.einsum('ijkl,kl->ij',mixed_ints,dm_p)
  vhfe = j_el - 0.5*k_el - j_mixed_elec
#  vhfp = j_nu - k_nu - j_mixed_nuc
  vhfp = - j_mixed_nuc
  vhfep = j_mixed_nuc
  return vhfe, vhfp, vhfep

def energy_tot(mf_e,mf_p,eldm,nudm,h1e,h1p,vhfe,vhfp,vhfep):
  e_tot = energy_elec(eldm,nudm,h1e,h1p,vhfe,vhfp,vhfep) + mf_e.energy_nuc()
  return e_tot.real

def energy_elec(eldm,nudm,h1e,h1p,vhfe,vhfp,vhfep):
  e1 = numpy.einsum('ji,ji', h1e.conj(), eldm).real
  e_coul = numpy.einsum('ji,ji', vhfe.conj(), eldm).real * .5
  p1 = numpy.einsum('ji,ji', h1p.conj(), nudm).real
  p_coul = numpy.einsum('ji,ji', vhfp.conj(), nudm).real * .5
  ep_coul = numpy.einsum('ji,ji', vhfep.conj(), nudm).real * .5
  return e1+e_coul+p1+p_coul

def eigmc(h, s):
    d, t = numpy.linalg.eigh(s)
    x = t[:,d>1e-5] / numpy.sqrt(d[d>1e-5])
    xhx = numpy.matmul(x.T,numpy.matmul(h, x)) 
    e, c = numpy.linalg.eigh(xhx)
    c = numpy.matmul(x, c)
    return e, c

'''
H   S 
    1.217  1.0000
H   S
   10.238  1.0000
H   S
   18.246  1.0000
H   S
   30.285  1.0000
H   P
   10.708  1.0000
H   P
   12.068  1.0000
H   P
   23.066  1.0000
H   D
   11.586  1.0000
H   D
   21.462  1.0000

H   S
    2.82843   1.0000
H   S
    4.00000   1.0000
H    S
    5.65685   1.0000
H   S
    8.00000   1.0000
H    S
    11.313708   1.0000
H   S
    16.0000000   1.0000
H    S
    22.6274   1.0000
H   S
    32.00000   1.0000
H   D
    2.82843   1.0000
H   D
    4.00000   1.0000
H   D
    5.65685   1.0000
H   D
    8.00000   1.0000
H   D
    11.313708   1.0000
H   D
    16.0000000   1.0000
H   D
    22.6274   1.0000
H   D
    32.00000   1.0000


H    S
    5.65685   1.0000
H   S
    8.00000   1.0000
H    S
    11.313708   1.0000
H   S
    16.0000000   1.0000
H    S
    22.6274   1.0000
H   S
    32.00000   1.0000
H   P
    5.65685   1.0000
H   P
    8.00000   1.0000
H   P
    11.313708   1.0000
H   P
    16.0000000   1.0000
H   P
    22.6274   1.0000
H   P
    32.00000   1.0000
H   D
    5.65685   1.0000
H   D
    8.00000   1.0000
H   D
    11.313708   1.0000
H   D
    16.0000000   1.0000
H   D
    22.6274   1.0000
H   D
    32.00000   1.0000
H   F
    5.65685   1.0000
H   F
    8.00000   1.0000
H   F
    11.313708   1.0000
H   F
    16.0000000   1.0000
H   F
    22.6274   1.0000
H   F 
    32.00000   1.0000

H   S
    5.973 1.0000
H   S
    10.645 1.00000
H    S
    17.943 1.00000
H   S
    28.950 1.0000
H   P
    7.604   1.0000
H   P
    14.701   1.0000
H   P
    23.308  1.0000
H   D
    9.011   1.0000
H   D
    19.787   1.0000
H   F
    10.914   1.0000
'''
