import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.linalg import sqrtm
import sys
from mpi4py import MPI


def normal(x, v):
    return 1/np.sqrt(2*np.pi*v)*np.exp(- 0.5 * x**2 / v)


def func(z2, omega1, omega2, a, b, v11):
    return np.sign(z2) * normal(z2-omega2, v11) * erf((b*(z2-omega2) - a*omega1) / np.sqrt(2*a))


def g_out(y, omega, V, eps=1e-6, mult=10):
    V_inv = np.linalg.inv(V + eps*np.eye(2))
    omega1, omega2 = omega

    v00 = V[0,0] + eps
    v11 = V[1,1] + eps
    a = V_inv[0,0] 
    b = V_inv[0,1] + eps
    c = V_inv[1,1]

    I = quad(func, -mult*v11, mult*v11, args=(omega1, omega2, a, b, v11))[0]
    Z_int = (1 - y*I)/2
    
    return y/Z_int * np.array([normal(omega1, v00)*erf((b*omega1 + c*omega2)/np.sqrt(2*c)), normal(omega2, v11)*erf((b*omega2 + a*omega1)/np.sqrt(2*a))])


def Q_func(Q_hat):
    return Q_hat @ np.linalg.inv(np.eye(2) + Q_hat)


def Q_hat_func_MCMC(alpha, Q, samples):
    Q_hat = np.zeros((2,2))
    for _ in range(samples):

        Z = np.random.normal(0,1, 2)
        U = np.random.normal(0,1, 2)

        sqrt_Q = sqrtm(Q)
        sqrt_one_minus_Q = sqrtm(np.eye(2) - Q)
        omega = sqrt_Q @ Z
        y = np.sign( np.prod(sqrt_Q@Z + sqrt_one_minus_Q@U) ) 
        V = np.eye(2) - Q

        g_out_vec = g_out(y, omega, V)
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
            Q_hat_all = np.zeros((size, 2, 2), dtype=np.float64)

            Q_hat_all[0] = Q_hat
            for j in range(1, size):
                Q_hat_all[j] = comm.recv(source=j)

            Q_hat = np.mean(Q_hat_all, axis=0)
            Q = damping*Q_func(Q_hat) + (1-damping)*Q
            Q_list.append(Q)
            np.save(f"data_multi_index/Q_list_alpha_{alpha}_samples_{int(size*samples)}.npy", Q_list)

            for j in range(1, size):
                comm.send(Q, dest=j)



        

if __name__=="__main__":
    alpha = float(sys.argv[1])
    iter = int(sys.argv[2])
    samples = int(sys.argv[3])
    Q = np.array([[.01,.0],[.0,.01]])
    # Q = np.array([[.5,.0],[.0,.5]])

    main(alpha, Q, samples, iter)