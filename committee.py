import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import erf
from scipy.linalg import sqrtm
from mpi4py import MPI
from tqdm import tqdm
import sys



def norm2d(z, V_inv):
    N = 2*np.pi / np.sqrt(np.linalg.det(V_inv))
    return np.exp(-1/2*(V_inv[0,0]*z[0]**2 + V_inv[1,1]*z[1]**2 + 2*V_inv[0,1]*z[0]*z[1]))/N


def channel2(y, omega, V, mult = 10):
    V = np.linalg.inv(V + 1e-6*np.eye(3))
    
    def dV(V):
        dV12 = np.linalg.det(V[:2,:2])
        dV13 = np.linalg.det(np.delete(np.delete(V, 1, 0), 1, 1))
        dV123 = np.linalg.det(np.delete(np.delete(V, 2, 0), 1, 1))
        return np.array([np.array([dV12, dV123]), np.array([dV123, dV13])])
    def quad1(f, args, d1 = 1): #d1 is related to the domain of integration: 1 for (0, mult), -1 for (-mult, 0)
        in1, fin1 = np.sort([0, d1*mult])
        return quad(f, in1, fin1, args = args)[0]
    def quad2(f, args, d1 = 1, d2 = 1): #d1, d2 are related to the domains of integration: 1 for (0, mult), -1 for (-mult, 0)
        in1, fin1 = np.sort([0, d1*mult])
        in2, fin2 = np.sort([0, d2*mult])
        return dblquad(f, in2, fin2, in1, fin1, args = args)[0]
    def swap(v, i1, i2):
        M = v.copy()
        M[[i1,i2]] = M[[i2, i1]]
        if M.shape == (3,3):
            M[:,[i1,i2]] = M[:,[i2, i1]]
        return M
        
    def func(z1, z2, omega, V, s): #s = +-1
        Z1 = z1 - omega[1]
        Z2 = z2 - omega[2]
        DV = dV(V)/V[0,0]
        argument = (V[0,1]*Z1 + V[0,2]*Z2 - V[0,0]*omega[0]) / np.sqrt(2*V[0,0])
        return (1/2) * norm2d(np.array([Z1, Z2]), DV) * (1 - s*erf(argument))
    def dfunc2(z1, z2, omega, V, s): #dfunc/domega[2]
        Z1 = z1 - omega[1]
        Z2 = z2 - omega[2]
        DV = dV(V)/V[0,0]
        argument = (V[0,1]*Z1 + V[0,2]*Z2 - V[0,0]*omega[0]) / np.sqrt(2*V[0,0])

        dargument = - V[0,2] / np.sqrt(2*V[0,0])
        dexponent = Z2*DV[1,1] + Z1*DV[0,1]

        term1 = (1 - s*erf(argument)) * dexponent
        term2 = - s * (2 / np.sqrt(np.pi)) * np.exp(-argument**2) * dargument
        return (1/2) * norm2d(np.array([Z1, Z2]), dV(V)/V[0,0]) * (term1 + term2)
    def Z_out_fun(y, omega, V): 
        if np.abs(y) == 3:
            s = np.sign(y)
            return quad2(func, args = (omega, V, s), d1 = s, d2 = s)
        if np.abs(y) == 1:
            s = np.sign(y)
            I_mpp = quad2(func, args = (omega, V, -s), d1 = s, d2 = s)
            I_ppm = quad2(func, args = (omega, V, s), d1 = s, d2 = -s)
            I_pmp = quad2(func, args = (omega, V, s), d1 = -s, d2 = s)
            return I_mpp + I_ppm + I_pmp
    Z_out = Z_out_fun(y, omega, V)

    def g_out_num(y, omega, V): #Notation for dZ_out / domega[2] 
        if np.abs(y) == 3:
            s = np.sign(y)
            return s*quad1(func, args = (0, omega, V, s), d1 = s)
        if np.abs(y) == 1:
            s = np.sign(y)
            I_mpp = s * quad1(func, args = (0, omega, V, -s), d1 = s)
            I_ppm = -s * quad1(func, args = (0, omega, V, s), d1 = s)
            I_pmp = s * quad1(func, args = (0, omega, V, s), d1 = -s)
            return I_mpp + I_ppm + I_pmp
    omega02 = swap(omega, 0, 2)
    omega01 = swap(omega, 0, 1)
    omega12 = swap(omega, 1, 2)
    V02 = swap(V, 0, 2)
    V01 = swap(V, 0, 1)
    V12 = swap(V, 1, 2)
    g_out_2 = g_out_num(y, omega, V) / Z_out
    g_out_1 = g_out_num(y, omega12, V12) / Z_out
    g_out_0 = g_out_num(y, omega02, V02) / Z_out
    g_out = np.array([g_out_0, g_out_1, g_out_2])

    return g_out


def Q_func(Q_hat):
    return Q_hat @ np.linalg.inv(np.eye(3) + Q_hat)


def Q_hat_func_MCMC(alpha, Q, samples):
    Q_hat = np.zeros((3,3))
    for _ in tqdm(range(samples)):

        Z = np.random.normal(0,1, 3)
        U = np.random.normal(0,1, 3)

        sqrt_Q = sqrtm(Q)
        sqrt_one_minus_Q = sqrtm(np.eye(3) - Q)
        omega = sqrt_Q @ Z
        preact = sqrt_Q@Z + sqrt_one_minus_Q@U
        y = np.sign(preact[0]) + np.sign(preact[1]) + np.sign(preact[2])
        V = np.eye(3) - Q

        g_out_vec = channel2(y, omega, V)
        Q_hat += alpha * np.outer(g_out_vec, g_out_vec)

    return Q_hat / samples


def main(alpha, Q, samples, iter, damping=.7):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        Q_list = []

    for i in range(iter):
        Q_hat = Q_hat_func_MCMC(alpha, Q, samples)
        
        if rank != 0:
            comm.send(Q_hat, dest=0)
            Q = comm.recv(source=0)
        
        if rank == 0:
            Q_hat_all = np.zeros((size, 3, 3), dtype=np.float64)

            Q_hat_all[0] = Q_hat
            for j in range(1, size):
                Q_hat_all[j] = comm.recv(source=j)

            Q_hat = np.mean(Q_hat_all, axis=0)
            Q = damping*Q_func(Q_hat) + (1-damping)*Q
            Q_list.append(Q)
            np.save(f"data_multi_index_committee/Q_list_alpha_{alpha}_samples_{int(size*samples)}.npy", Q_list)

            Q = (Q + Q.T) / 2
            for j in range(1, size):
                comm.send(Q, dest=j)



        

if __name__=="__main__":
    alpha = float(sys.argv[1])
    iter = int(sys.argv[2])
    samples = int(sys.argv[3])
    Q = np.array([[.5,.0,.0],[.0,.5,.0],[.0,.0,.5]])


    main(alpha, Q, samples, iter)