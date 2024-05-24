import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import erf
import sys
from mpi4py import MPI

def sample_instance(size_x, rows_to_columns, var_noise=0): #x0 are the weights, F the data
    
    # Some pre-processing
    size_y = int(np.ceil(rows_to_columns * size_x))
    
    # Sample x from P_0(x)
    x01 = np.random.normal(0, 1, size_x)
    x02 = np.random.normal(0, 1, size_x)
    x03 = np.random.normal(0, 1, size_x)
    F = np.random.normal(0, 1 , (size_y, size_x)) / np.sqrt(size_x)
    z1 = F.dot(x01)
    z2 = F.dot(x02)
    z3 = F.dot(x03)
    y = np.sign(z1) + np.sign(z2) + np.sign(z3)
    return np.array([x01,x02,x03]), F, y

def sample_labels(x0, rows_to_columns, var_noise=0): #x0 must be K x d
    # Some pre-processing
    K, size_x = x0.shape
    size_y = int(np.ceil(rows_to_columns * size_x))
    
    # Generate F and y
    F = np.random.normal(0, 1 , (size_y, size_x)) / np.sqrt(size_x)
    y = np.sign(F.dot(x0[0,:])) + np.sign(F.dot(x0[1,:])) + np.sign(F.dot(x0[1,:]))
    return F, y

def get_estimate(xhat, W):
    '''
    xhat: d x K array!
    '''
    z1 = W.dot(xhat[:,0])
    z2 = W.dot(xhat[:,1])
    z3 = W.dot(xhat[:,2])
    y = np.sign(z1) + np.sign(z2) + np.sign(z3)
    return y
    
def get_error(y, W, xhat):
    yhat = get_estimate(xhat, W)
    return np.mean(((y-yhat))**2)/2

def prior(b, A): #Gaussian prior
    '''
    b: d x K dimensional tensor
    A: d x K x K tensor
    '''
    x_size, K = b.shape
    inv = np.linalg.inv(np.stack(x_size*[np.identity(K)], axis=0)+A)
    
    return np.einsum('ijk,ik->ij', inv, b), inv

def norm2d(z, V_inv):
    N = 2*np.pi / np.sqrt(np.linalg.det(V_inv))
    return np.exp(-1/2*(V_inv[0,0]*z[0]**2 + V_inv[1,1]*z[1]**2 + 2*V_inv[0,1]*z[0]*z[1]))/N

def channel2(y, omega, V, mult = 10):
    V = np.linalg.inv(V)
    
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
        
    def func(z1, z2, omega, V, s): #s = +-1; Python notation for z = (z[0], z[1], z[2]) -> z1 = z[1]
        Z1 = z1 - omega[1]
        Z2 = z2 - omega[2]
        DV = dV(V)/V[0,0]
        argument = (V[0,1]*Z1 + V[0,2]*Z2 - V[0,0]*omega[0]) / np.sqrt(2*V[0,0])
        return (1/2) * norm2d(np.array([Z1, Z2]), DV) * (1 - s*erf(argument))
    def d2func(z1, z2, omega, V, s): #dfunc/domega[2]
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

    def dZ_out_ij(y, omega, V): #notation for d(dZ_out / domega[2])/domega[1]
        if np.abs(y) == 3:
            s = np.sign(y)
            return func(0, 0, omega, V, s)
        if np.abs(y) == 1:
            s = np.sign(y)
            I_mpp = func(0, 0, omega, V, -s)
            I_ppm = -func(0, 0, omega, V, s)
            I_pmp = -func(0, 0, omega, V, s)
            return I_mpp + I_ppm + I_pmp
    def dZ_out_ii(y, omega, V): #notation for d(dZ_out / domega[2])/domega[2]
        if np.abs(y) == 3:
            s = np.sign(y)
            return s*quad1(d2func, args = (0, omega, V, s), d1 = s)
        if np.abs(y) == 1:
            s = np.sign(y)
            I_mpp = s * quad1(d2func, args = (0, omega, V, -s), d1 = s)
            I_ppm = -s * quad1(d2func, args = (0, omega, V, s), d1 = s)
            I_pmp = s * quad1(d2func, args = (0, omega, V, s), d1 = -s)
            return I_mpp + I_ppm + I_pmp
    
    dZ_out_21 = dZ_out_ij(y, omega, V)
    dZ_out_20 = dZ_out_ij(y, omega01, V01)
    dZ_out_01 = dZ_out_ij(y, omega02, V02)
    dZ_out_22 = dZ_out_ii(y, omega, V)
    dZ_out_11 = dZ_out_ii(y, omega12, V12)
    dZ_out_00 = dZ_out_ii(y, omega02, V02)

    dZ_out = np.zeros((3,3))
    dZ_out[0,0] = dZ_out_00
    dZ_out[0,1] = dZ_out_01
    dZ_out[0,2] = dZ_out_20
    dZ_out[1,0] = dZ_out_01
    dZ_out[1,1] = dZ_out_11
    dZ_out[1,2] = dZ_out_21
    dZ_out[2,0] = dZ_out_20
    dZ_out[2,1] = dZ_out_21
    dZ_out[2,2] = dZ_out_22
    dg_out = dZ_out / Z_out - np.outer(g_out, g_out)

    return g_out, dg_out


def channel(Y, W, V_, eps = 1e-3, mult = 10): #vectorized version, maybe could be improved
    s = len(Y)
    g = []
    dg = []
    for i in range(s):
        y = Y[i]
        w = W[i]
        # V = np.linalg.inv(V_[i] + np.eye(3)*eps)
        V = V_[i]
        v1, v2 = channel2(y,w,V, mult = mult)
        g.append(v1)
        dg.append(v2)
    return np.array(g), np.array(dg)

def check_increasing(mses, epochs=5):
    if epochs > len(mses):
        print('Number of epochs must be smaller than length of array!')
        return False
    else:
        return True if np.all(np.diff(mses[-epochs:]) > 0) else False

def damping(x_new, x_old, coeff=0.2):
    if coeff > 1:
        print('Coefficient must be between 0 and 1. Returning new value.')
        return x_new
    else:
        return (1-coeff) * x_new + coeff * x_old

def iterate_gamp(W, y, x0=None, max_iter=50, tol=1e-4, 
                 damp=0, early_stopping=False, verbose=1, test_err = False, plant = 0, mult = 10): #W are te data!, x0 are the weights
    '''
    Note: implementation assuming A commutes with C
    '''
    # Preprocessing
    y_size, x_size = W.shape #W sono i dati
    W2 = W * W   
    K = 3

    
    
    # Initialisation
    xhat = np.sqrt(plant)*x0.T + np.sqrt(1-plant)*np.random.randn(x_size, K)
    
    g = np.zeros((y_size, K))
    chat = np.stack(x_size*[np.identity(K)], axis=0)
    
    count = 0
    train_error = np.zeros(max_iter+1)
    test_error = np.zeros(max_iter+1)
    m = np.zeros((max_iter+1, K, K))
    
    m[0] =  np.tensordot(xhat, x0, axes=([0, 1])) / x_size #initial overlap
    #q0 = np.tensordot(xhat, xhat.T, axes=([0, 1])) / x_size #initial self-overlap, not stored
    print('initial truth ovelap -> ', m[0])
    #print('initial self overlap -> ', q0)
    
    train_error[0] = get_error(y, W, xhat)

    
    if test_err:
        alpha = y_size/x_size
        X_fresh, y_fresh = sample_labels(x0, alpha)
        test_error[0] = get_error(y_fresh, X_fresh, xhat)
    
    for t in range(max_iter):
        # First part
        V = np.tensordot(W2,chat, axes=([1,0]))
        tmp = np.tensordot(W2, chat, axes=([1,0]))
        omega = np.tensordot(W, xhat, axes=([1,0])) - np.einsum('ijk,ik->ij', tmp, g)
        
        g, dg = channel(y, omega, V, mult = mult)
        
        # Second part
        A = - np.tensordot(W2, dg, axes=([0, 0])) #Sigma^-1 in Aubin's paper
        b = np.einsum('ijk,ik->ij', A, xhat) + np.tensordot(W, g, axes=([0, 0]))
    
        xhat_old = xhat.copy() # Keep a copy of xhat to compute diff
        chat_old = chat.copy()
        xhat_, chat_ = prior(b,A)    

       
      

        xhat = damping(xhat_, xhat_old, damp)
        chat = damping(chat_, chat_old, damp)
        
        diff = np.linalg.norm(np.abs(xhat)-np.abs(xhat_old))/np.sqrt(x_size)
        train_error[t+1] =  get_error(y, W, xhat)

        m[t+1] = np.tensordot(xhat, x0, axes=([0, 1])) / x_size
        print(m[t+1], 'Truth OVERLAP')
        #print(np.tensordot(xhat, xhat.T, axes=([0, 1])) / x_size, 'Self OVERLAP')
        if (early_stopping) and (t>1) and (train_error[t]-train_error[t-1] > 0):
            count += 1
        else:
            count = 0
            
        # if count == 5:
        #     print('Early stopping')
        #     return mses[:t-4]
        if test_err:
           X_fresh, y_fresh = sample_labels(x0, alpha)
           test_error[t+1] = get_error(y_fresh, X_fresh, xhat) 
            
        if verbose:
            if test_err:
                print('t: {}, diff: {}, train_error: {}, test_err: {}'.format(t, diff, train_error[t+1], test_error[t+1]))
            else:
                print('t: {}, diff: {}, error: {}'.format(t, diff, train_error[t+1]))
        
        if (diff < tol) or (train_error[t+1] < tol):
            break
        

    return xhat, m[:t+2], train_error[:t+2],test_error[:t+2], t+2







def main(d, alpha, max_iter=200, plant=0.9, damp=0.2, mult=7):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    w0, X, y = sample_instance(d, alpha)

    _, m, mses, tests, _ = iterate_gamp(X, y, x0=w0, max_iter=max_iter, tol = 1e-3, test_err = True, mult = mult, damp = damp, plant = plant)

    if rank != 0:
        comm.send(m, dest=0)
        comm.send(tests, dest=0)


    if rank == 0:
        MS = []
        TESTS = []
        MS.append(m)
        TESTS.append(tests)
        for i in range(1, size):
            MS.append(comm.recv(source=i))
        
        for i in range(1, size):
            TESTS.append(comm.recv(source=i))

        np.save(f"data_AMP_committee/MS_{d}_{alpha}", MS)
        np.save(f"data_AMP_committee/TESTS_{d}_{alpha}", TESTS)


if __name__=="__main__":
    d = 100
    alpha = float(sys.argv[1])
    max_iter = 50
 
    main(d, alpha, max_iter=max_iter)

 