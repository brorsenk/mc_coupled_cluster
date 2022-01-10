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


import numpy as np
import rcc as rcc
from rcc import *
from makemol import *
from pyscf import gto, scf, lib
from pyscf.scf import diis
from pyscf.dft import numint
from pyscf import ao2mo, mp
from functools import reduce
import time
import sys
from e_fock import *
import hf_e
from triples import triples_correction 

### input parameters

coords = [['H', ( 0.,    0., 0.00000    )],
         ['N', (  0.,    0., 1.02995    )],
         ['N', (  0.,    0., 2.11818    )]]
charge = 1

calc = 'normal'
#calc = 'PsH'
restart = False

basis_dict = {'F': 'aug-cc-pvtz', 'O': 'aug-cc-pvtz', 'C': 'aug-cc-pvtz', 'N': 'aug-cc-pvtz',
              'H': 'aug-cc-pvtz', 'H1':'aug-cc-pvtz', 'H2':'aug-cc-pvtz', 'S': 'aug-cc-pvtz'}

proton_num = 0

if calc == 'PsH':
    nuc_mass = 1
else:
    nuc_mass = 1836.152673
cart_basis = False 
niter = 80

if(nuc_mass == 1):
  positron = True
else:
  positron = False

print('\ncoordinates of molecule are ')
print(coords,'\n')
print('electronic basis sets being used:')
print(basis_dict)

## set up individual mol objects, elec/nuc guess and mean-field objects
mol_elec, mol_nuc, mol_tot, dm0_e, dm0_p= makemol(coords, basis_dict, h_basis, proton_num, charge, nuc_mass, positron, cart_basis)

mf_e = scf.RHF(mol_elec)
mf_e.dm = dm0_e
mf_e = scf.addons.remove_linear_dep(mf_e)
mf_e_diis = diis.SCF_DIIS(mf_e, mf_e.diis_file)
mf_e_diis.space = mf_e.diis_space
mf_e_diis.rollback = mf_e.diis_space_rollback
mf_e.max_cycle=50

mf_p = scf.RHF(mol_nuc)
mf_p.dm = dm0_p

h1e  = scf.hf.get_hcore(mol=mol_elec)
h1p  = get_nuc_core(mol_nuc, nuc_mass, mol_elec)
s1e  = scf.hf.get_ovlp(mol_elec)
s1p  = scf.hf.get_ovlp(mol_nuc)
vhfe,vhfp,vhfep = get_vhf_mc(mol_elec, mol_nuc, mol_tot, mf_e.dm, mf_p.dm,dm_p_last=None,dm_e_last=None)

scf_conv = False
cycle = 0
tcycle = 0
mf_e.dm_last = mf_e.dm
mf_p.dm_last = mf_p.dm
norm_e_dm = np.linalg.norm(mf_e.dm-mf_e.dm_last)
norm_p_dm = np.linalg.norm(mf_p.dm-mf_p.dm_last)

# do MC-HF
while not scf_conv and cycle < max(1,mf_e.max_cycle):

    if(cycle%2==0):
      mf_e.dm_last = mf_e.dm
      focke = mf_e.get_fock(h1e, s1e, vhfe, mf_e.dm,cycle,mf_e_diis)
      mf_e.mo_energy, mf_e.mo_coeff = mf_e.eig(focke, s1e)
      mf_e.mo_occ = mf_e.get_occ(mf_e.mo_energy, mf_e.mo_coeff)
      mf_e.dm = mf_e.make_rdm1(mf_e.mo_coeff, mf_e.mo_occ)
      norm_e_dm = np.linalg.norm(mf_e.dm-mf_e.dm_last)

    else:
      mf_p.dm_last = mf_p.dm
      fockp = mf_p.get_fock(h1p, s1p, vhfp, mf_p.dm,cycle)
      mf_p.mo_energy, mf_p.mo_coeff = eigmc(fockp, s1p)
      mf_p.mo_occ = np.zeros(len(mf_p.mo_energy))
      mf_p.mo_occ[0]=2
      mf_p.dm = mf_p.make_rdm1(mf_p.mo_coeff, mf_p.mo_occ)*.5
      norm_p_dm = np.linalg.norm(mf_p.dm-mf_p.dm_last)

    focke = mf_e.get_fock(h1e, s1e, vhfe, mf_e.dm)
    fockp = mf_p.get_fock(h1p, s1p, vhfp, mf_p.dm)


    vhfe,vhfp,vhfep = get_vhf_mc(mol_elec, mol_nuc, mol_tot, mf_e.dm, mf_p.dm,dm_e_last=mf_e.dm_last,dm_p_last=mf_p.dm_last)

    energy = energy_tot(mf_e,mf_p,mf_e.dm,mf_p.dm,h1e,h1p,vhfe,vhfp,vhfep)

    if(cycle>=0):
      print('{:02d}'.format(cycle),'{0:.8f}'.format(energy),'{0:.9f}'.format(norm_e_dm),'{0:.9f}'.format(norm_p_dm))


    if (norm_p_dm<0.0000001 and norm_e_dm<0.0000001 and cycle>5):
        scf_conv = True
        print('\nMC-HF CONVERGED!!!\n')
        ehf = energy
    cycle = cycle + 1

### electronic mp2
if mol_nuc.cart:
    intor = 'int2e_cart'
else:
    intor = 'int2e_sph'

nocc = mol_elec.nelectron//2
nvir = len(mf_e.mo_occ) - nocc
ee_ints = mol_elec.intor(intor,aosym='s8')
#pp_ints = mol_nuc.intor(intor,aosym='s8')
co = mf_e.mo_coeff[:,:nocc]
cv = mf_e.mo_coeff[:,nocc:]

eri_ee = ao2mo.incore.general(ee_ints,(co,cv,co,cv),compact=False)
eia = mf_e.mo_energy[:nocc,None] - mf_e.mo_energy[None,nocc:]

emp2 = 0.0

for i in range(nocc):
   gi = np.asarray(eri_ee[i*nvir:(i+1)*nvir])
   gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
   t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
   theta = gi*2 - gi.transpose(0,2,1)
   emp2 += np.einsum('jab,jab', t2i, theta)

print ('electronic mp2 energy = ',emp2)

# mc-mp2
e_nocc = mol_elec.nelectron//2
e_nvir = len(mf_e.mo_occ) - e_nocc
e_tot = e_nocc + e_nvir
p_nocc = 1
p_nvir = len(mf_p.mo_occ) - p_nocc
p_tot = p_nocc + p_nvir
ep_tot = e_tot + p_tot

combined_mo = np.zeros((mol_tot.nao_nr(), ep_tot))
combined_mo[0:mol_elec.nao_nr(), 0:e_tot]= mf_e.mo_coeff
combined_mo[mol_elec.nao_nr():mol_tot.nao_nr(), e_tot:ep_tot]= mf_p.mo_coeff

co_e= combined_mo[:,:e_nocc]
cv_e= combined_mo[:,e_nocc:e_tot]
co_n=combined_mo[:,e_tot:e_tot+p_nocc]
cv_n=combined_mo[:,e_tot+p_nocc:]
c_e = combined_mo[:,:e_tot]
c_p = combined_mo[:,e_tot:]

mixed_ints = mol_tot.intor(intor,aosym='s8')
eri_ep = ao2mo.incore.general(mixed_ints,(co_e,cv_e,co_n,cv_n),compact=False)

emp2_mc = 0.0
eia = mf_e.mo_energy[:e_nocc,None] - mf_e.mo_energy[None,e_nocc:]
ejb = mf_p.mo_energy[:p_nocc,None] - mf_p.mo_energy[None,p_nocc:]

for i in range(e_nocc):
   gi = np.asarray(eri_ep[i*e_nvir:(i+1)*e_nvir])
   gi = gi.reshape(e_nvir,p_nocc,p_nvir)
   t2i = gi/lib.direct_sum('a+jb->ajb', eia[i], ejb)
   emp2_mc += np.einsum('ajb,ajb', t2i, gi)

eri_ep_full = (ao2mo.incore.general(mixed_ints, (c_e,c_e,c_p,c_p), compact=False)).reshape(e_tot,e_tot,p_tot,p_tot)
mc_emp2=0
for i in range(e_nocc):
    for I in range(p_nocc):
        for a in range(e_nocc,e_tot):
            for A in range(p_nocc,p_tot):
                mc_emp2 += (eri_ep_full[i,a,I,A] **2)/ (mf_e.mo_energy[i] +mf_p.mo_energy[I] - mf_e.mo_energy[a] - mf_p.mo_energy[A] )



print('MC-MP2 energy = ', 2*mc_emp2)
print ('mc-mp2 energy = ',2*emp2_mc)
print('total energy =', 2*emp2_mc+emp2+energy)

# now do mc-ccsd 

nocc = e_nocc
pocc = p_nocc
nvir = e_tot - nocc
pvir = p_tot - p_nocc
ntot = nocc + nvir
ptot = pocc + pvir

eri_ee_full = (ao2mo.incore.full(ee_ints, mf_e.mo_coeff, compact=False)).reshape(e_tot,e_tot,e_tot,e_tot)
eri_ep_full = (ao2mo.incore.general(mixed_ints, (c_e,c_e,c_p,c_p), compact=False)).reshape(e_tot,e_tot,p_tot,p_tot)

focke, fockp = rcc.spat_fock(mf_e.mo_energy,ntot,mf_p.mo_energy,ptot)

eri = eri_ee_full.transpose(0,2,1,3)
eri_ep = eri_ep_full.transpose(0,2,1,3)

if restart == False:
    t1e, t2ee, t1p, t2ep = rcc.c_amps(focke, eri, nocc, ntot, fockp, eri_ep, pocc, ptot)
elif restart == True:
    t1e = np.loadtxt('t1e')
    t1e = np.array(t1e)
    t1e = t1e.reshape(s_occ,nvir)
    
    t1p = np.loadtxt('t1p')
    t1p = np.array(t1p)
    t1p = t1p.reshape(p_nocc,pvir)
    
    t2ee = np.array(np.loadtxt('t2ee'))
    t2ee = t2ee.reshape(s_occ,s_occ,nvir,nvir)
    
    t2ep = np.array(np.loadtxt('t2ep'))
    t2ep = t2ep.reshape(s_occ,p_nocc,nvir,pvir)

mc_ecc_i = rcc.cc_ecc(focke,eri,nocc,nvir,t1e,t2ee,fockp,eri_ep,t2ep,t1p,pocc,pvir)

print('Init MC-CCSD energy: {mc_ecc_i}')

mc_ecc, t1e, t1p, t2ee, t2ep = rcc.run_ccsd(focke, energy,t1e,t1p,t2ee,t2ep,eri,eri_ep,fockp,nocc,nvir,pocc,pvir, conv=1e-9)

e_nocc = nocc
e_nvir = nvir

eri_ee_ovvv = eri_ee_full[:e_nocc, e_nocc:, e_nocc:, e_nocc:]
eri_ee_ooov = eri_ee_full[:e_nocc, :e_nocc, :e_nocc, e_nocc:]
eri_ee_ovov = eri_ee_full[:e_nocc, e_nocc:, :e_nocc, e_nocc:]

eri_ep_ooOV = eri_ep_full[:e_nocc, :e_nocc, :p_nocc, p_nocc:]
eri_ep_vvOV = eri_ep_full[e_nocc:, e_nocc:, :p_nocc, p_nocc:]
eri_ep_ovOO = eri_ep_full[:e_nocc, e_nocc:, :p_nocc, :p_nocc]
eri_ep_ovVV = eri_ep_full[:e_nocc, e_nocc:, p_nocc:, p_nocc:]
eri_ep_ovOV = eri_ep_full[:e_nocc, e_nocc:, :p_nocc, p_nocc:]

t2ee = np.asarray(t2ee)
t2ep = np.asarray(t2ep)
t1e  = np.asarray(t1e)
t1p  = np.asarray(t1p)

t2ep = t2ep.transpose(0,2,1,3)


start = time.time()
triples_correction(e_nocc, e_nvir, p_nocc, p_nvir, t2ee, t2ep, t1e, t1p, eri_ee_ovvv, eri_ee_ooov, eri_ee_ovov, eri_ep_ooOV, eri_ep_vvOV, eri_ep_ovOO, eri_ep_ovVV, eri_ep_ovOV, eia, ejb)
finish = time.time()
print('time for triples correction = ',finish-start)

