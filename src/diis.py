import numpy as np

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

class diis:

    def __init__(self, nao, s,x):

        self.nao = nao
        self.max_diis = 8
        self.iter = 0
        self.s = s
        self.e_m = []
        self.f = []
        self.x = x
                
    def run_diis(self, d,fock):
        nao = self.nao
        s = self.s
        e_m = self.e_m
        f = self.f
        x = self.x
        max_diis = self.max_diis
        F = np.zeros((nao,nao))
        E = np.zeros((nao,nao))

        if self.iter == 0:
            e0 = np.matmul(fock, np.matmul(d, s)) - np.matmul(s, np.matmul(d, fock))
            e0 = np.matmul(x.transpose(), np.matmul(e0, x))
            e_m.append(e0)
            f.append(fock)
            self.iter += 1
            F = fock

        else:
            ei = np.matmul(fock,np.matmul(d,s)) - np.matmul(s, np.matmul(d,fock))
            ei = np.matmul(x.transpose(), np.matmul(ei,x))
            e_m.append(ei)
            f.append(fock)
            if len(f) > max_diis:
                f.pop(0)
                e_m.pop(0)
                self.iter = 7
            n = self.iter + 1
            B = np.zeros((n+1,n+1))
            for i in range(n):
                for j in range(i+1):
                    B[i,j] = B[j,i] = np.dot(e_m[i].flatten(), e_m[j].flatten())
            B[n,:n] = -1
            B[:n,n] = -1
            B[n,n] = 0 
            b_vec = np.zeros((n+1))
            b_vec[n] = -1
            c = np.linalg.solve(B,b_vec)
            for i in range(len(f)):
                F += c[i]*f[i]
                E += c[i]*e_m[i]
            self.iter += 1
        return F


class cdiis:

    def __init__(self,nso):
        self.nso = nso
        self.e = []
        self.t = []
        self.iter = 0
        self.max_diis = 8
        self.ts = [0]

    def run_diis(self, T1,T2):
        nso = self.nso
        e = self.e
        t = self.t
        max_diis = self.max_diis
        ts = self.ts

        T1 = np.array(T1)
        T2 = np.array(T2)
        T1 = T1.flatten()
        T2 = T2.flatten()
        T = np.concatenate([T1,T2])

        if self.iter == 0:
            t.append(T)
            ts[0] = T
            self.iter += 1
            Tn = T
        else:
            ei = T - ts[0]
            t.append(T)
            ts[0] = T
            e.append(ei)
            if len(e) > max_diis:
                e.pop(0)
                t.pop(0)
                self.iter = 8
            n = self.iter + 1
            n = len(e) + 1
            B = np.zeros((n,n))
            b = np.zeros((n))
            for i in range(n-1):
                for j in range(i+1):
                    B[i,j] = B[j,i] = np.dot(e[i], e[j])
            
            B[n-1,:n-1] = -1
            B[:n-1,n-1] = -1
            B[:-1,:-1] /= np.abs(B[:-1,:-1]).max()
            b[n-1] = -1
            c = np.linalg.solve(B,b)
            Tn = np.zeros((len(T)))
            for i in range(len(e)):
                Tn += c[i] * t[i+1]
            ts[0] = Tn

        t1, t2 = Tn[:nso*nso], Tn[nso*nso:]
        t1 = t1.reshape(nso,nso)
        t2 = t2.reshape(nso,nso,nso,nso)
        return t1, t2


class mc_cc_diis:

    def __init__(self,nso,pso,s_occ,p_occ):
        self.nso = nso
        self.pso = pso
        self.socc = s_occ
        self.pocc = p_occ
        self.max_diis = 8
        socc = self.socc
        nvir = nso - socc
        pocc = self.pocc
        pvir = pso - pocc
        self.dim_diis = (socc*nvir) + (socc**2 * nvir**2) + (pocc*pvir) + ((socc*nvir) * (pocc*pvir))
        self.e = np.zeros((self.max_diis, (self.dim_diis)), dtype=float)
        self.t = np.zeros((self.max_diis+1, (self.dim_diis)), dtype=float)
        self.iter = 0
        self.ts = np.zeros((1, (self.dim_diis)),dtype=float) 

    def run_diis(self,T1e,T2ee,T1p,T2ep):
        nso = self.nso
        pso = self.pso
        socc = self.socc
        pocc = self.pocc
        nvir = nso - socc
        pvir = pso - pocc
        max_diis = self.max_diis

        T1e = np.array(T1e)
        T2ee = np.array(T2ee)
        T1p = np.array(T1p)
        T2ep = np.array(T2ep)
        T1e = T1e.flatten()
        T2ee = T2ee.flatten()
        T1p = T1p.flatten()
        T2ep = T2ep.flatten()
        T = np.concatenate([T1e,T2ee,T1p,T2ep])

        if self.iter == 0:
            self.t[self.iter+1] = T
            self.ts[0] = T
            self.iter += 1
            Tn = T
        elif self.iter < max_diis:
            ei = T - self.ts[0]
            self.t[self.iter+1] = T
            self.e[self.iter] = ei
            n = self.iter + 1
            B = np.zeros((n,n))
            b = np.zeros((n))
            for i in range(n-1):
                for j in range(i+1):
                    B[i,j] = B[j,i] = np.dot(self.e[i+1],self.e[j+1])

            B[n-1,:n-1] = -1
            B[:n-1,n-1] = -1
            B[:-1,:-1] /= np.abs(B[:-1,:-1]).max()
            b[n-1] = -1
            c = np.linalg.solve(B,b)
            Tn = np.zeros((len(T)))
            for i in range(n-1):
                Tn += c[i] *self.t[i+2]
            self.ts[0] = Tn
            self.iter += 1
        elif self.iter == max_diis:
            self.e = np.roll(self.e, -1, axis=0)
            self.t = np.roll(self.t, -1, axis=0)
            ei = T - self.ts[0]
            self.t[max_diis] = T
            self.e[max_diis-1] = ei
            n = max_diis + 1
            B = np.zeros((n,n))
            b = np.zeros((n))
            for i in range(n-1):
                for j in range(i+1):
                    B[i,j] = B[j,i] = np.dot(self.e[i],self.e[j])

            B[n-1,:n-1] = -1
            B[:n-1,n-1] = -1
            B[:-1,:-1] /= np.abs(B[:-1,:-1]).max()
            b[n-1] = -1
            c = np.linalg.solve(B,b)
            Tn = np.zeros((len(T)))
            for i in range(len(self.e)): # -1 bc t is longer than e
                Tn += c[i] * self.t[i+1]
            self.ts[0] = Tn
            self.iter += 1
        else:
            self.e = np.roll(self.e, -1, axis=0)
            self.t = np.roll(self.t, -1, axis=0)
            ei = T - self.ts[0]
            self.t[max_diis] = T
            self.e[max_diis-1] = ei
            n = max_diis + 1
            B = np.zeros((n,n))
            b = np.zeros((n))
            for i in range(n-1):
                for j in range(i+1):
                    B[i,j] = B[j,i] = np.dot(self.e[i],self.e[j])

            B[n-1,:n-1] = -1
            B[:n-1,n-1] = -1
            B[:-1,:-1] /= np.abs(B[:-1,:-1]).max()
            b[n-1] = -1
            c = np.linalg.solve(B,b)
            Tn = np.zeros((len(T)))
            for i in range(len(self.e)): # -1 bc t is longer than e
                Tn += c[i] * self.t[i+1]
            self.ts[0] = Tn
            self.iter +=1

        t1e, t2ee, t1p, t2ep = Tn[:socc*nvir], Tn[socc*nvir:(socc*nvir+(socc*socc*nvir*nvir))], Tn[(socc*nvir+(socc*socc*nvir*nvir)):(socc*nvir+(socc*socc*nvir*nvir))+(pocc*pvir)], Tn[(socc*nvir+(socc*socc*nvir*nvir))+(pocc*pvir):]
        t1e = t1e.reshape(socc,nvir)
        t2ee = t2ee.reshape(socc,socc,nvir,nvir)
        t1p = t1p.reshape(pocc,pvir)
        t2ep = t2ep.reshape(socc,pocc,nvir,pvir)

        return t1e, t2ee, t1p, t2ep
