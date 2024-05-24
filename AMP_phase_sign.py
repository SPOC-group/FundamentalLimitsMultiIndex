import numpy as np
from scipy.integrate import quad
from scipy.special import erf
import sys
from mpi4py import MPI

def sample_instance(size_x, rows_to_columns, var_noise=0): #x0 are the weights, F the data
    
    # Some pre-processing
    size_y = int(np.ceil(rows_to_columns * size_x))
    
    # Sample x from P_0(x)
    w = np.random.randn(3, size_x)
    X = np.random.normal(0, 1 , (size_y, size_x)) / np.sqrt(size_x)
    z1 = X.dot(w[0])
    z2 = X.dot(w[1])
    z3 = X.dot(w[2])
    y = z1**2 + np.sign(z1*z2*z3)
    return w, X, y

def sample_labels(w, rows_to_columns, var_noise=0): 
    '''
    w: 3 x d array
    '''
    # Some pre-processing
    K, size_x = w.shape
    size_y = int(np.ceil(rows_to_columns * size_x))
    
    # Generate F and y
    X = np.random.normal(0, 1 , (size_y, size_x)) / np.sqrt(size_x)
    z1 = X.dot(w[0])
    z2 = X.dot(w[1])
    z3 = X.dot(w[2])
    y = z1**2 + np.sign(z1*z2*z3)
    return X, y

def get_estimate(what, X):
    '''
    xhat: K x d array
    '''
    z1 = X.dot(what[0])
    z2 = X.dot(what[1])
    z3 = X.dot(what[2])
    y = z1**2 + np.sign(z1*z2*z3)
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

def channel2(y, omega, V, mult):
    #NOTE: V here is the inverse of V from AMP       
    
    def func(z, omega, V, i0 = +1, i1 = +1, i2 = +1):
        segno = i0*i1*i2
        
        Va = V[:2, :2]
        Vb = V[:2, 1:]
        Vc = V[1:, 1:]
        X = np.sqrt(y - segno) - i0*omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - i0 * 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-segno)))
        argument = (V[1,2]*Y + i0 * V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1-i1*erf(argument))
            

    def dw_func(z, omega, V, ind, i0 = +1, i1 = +1, i2 = +1):
        segno = i0*i1*i2

        Va = V[:2, :2]
        Vb = V[:2, 1:]
        Vc = V[1:, 1:]
        X = np.sqrt(y-segno) - i0 * omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - i0 * 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-segno)))
        argument = (V[1,2]*Y + i0 * V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
            
        if ind == 0:
            d_exponent = -i0*(-X*np.linalg.det(Va) + i0*Y*np.linalg.det(Vb)) / V[1,1]
            d_argument = -(V[0,1]/ np.sqrt(2 * V[1,1]))
        elif ind == 1:
            d_exponent = 0
            d_argument = -np.sqrt(V[1,1]/2)
        elif ind == 2:
            d_exponent =  -(X*np.linalg.det(Vb)  - i0*Y*np.linalg.det(Vc)) / V[1,1]
            d_argument = -(V[1,2]/ np.sqrt(2 * V[1,1]))
        
        return coeff * np.exp(exponent) * ((1-i1*erf(argument))*d_exponent -i1 * (2/np.sqrt(np.pi))*np.exp(-argument**2)*d_argument)

    def dw00_func(z, omega, V, i0 = +1, i1 = +1, i2 = +1): 
        segno = i0*i1*i2

        Va = V[:2, :2]
        Vb = V[:2, 1:]
        Vc = V[1:, 1:]
        X = np.sqrt(y-segno) - i0 * omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - i0 * 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-segno)))
        argument = (V[1,2]*Y + i0 * V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
            
        d_exponent = -i0*(-X*np.linalg.det(Va) + i0*Y*np.linalg.det(Vb)) / V[1,1]
        d_argument = -(V[0,1]/ np.sqrt(2 * V[1,1]))
        
        term = (1-i1*erf(argument))*d_exponent -i1 * (2/np.sqrt(np.pi))*np.exp(-argument**2)*d_argument
        d_term1 = -i1*(2/np.sqrt(np.pi))*np.exp(-argument**2)*d_argument*d_exponent - (1-i1*erf(argument))*np.linalg.det(Va) / V[1,1]
        d_term2 =  -i1 * (2/np.sqrt(np.pi))*np.exp(-argument**2)*(-2*argument)*d_argument**2
        return coeff * np.exp(exponent) * (d_exponent*term + d_term1 + d_term2)

    Z_out = 1e-8
    if y > 1:
        q_PPP = quad(func, 0, mult, args = (omega, V, 1, 1, 1))[0]
        q_PMM = quad(func, -mult, 0, args = (omega, V, 1, -1, -1))[0]
        q_MPM = quad(func, -mult, 0, args = (omega, V, -1, 1, -1))[0]
        q_MMP = quad(func, 0, mult, args = (omega, V, -1, -1, 1))[0]
        Z_out += q_PPP + q_PMM + q_MPM + q_MMP

    q_PPM = quad(func, -mult,0, args = (omega, V, 1, 1, -1))[0]
    q_PMP = quad(func, 0, mult, args = (omega, V, 1, -1, 1))[0]
    q_MPP = quad(func, 0, mult, args = (omega, V, -1, 1, 1))[0]
    q_MMM = quad(func, -mult, 0, args = (omega, V, -1, -1, -1))[0]
    Z_out += q_PPM + q_PMP + q_MPP + q_MMM

    g_out0 = 0
    if y > 1:
        dq_PPP = quad(dw_func, 0, mult, args = (omega, V, 0, 1, 1, 1))[0]
        dq_PMM = quad(dw_func, -mult, 0, args = (omega, V, 0, 1, -1, -1))[0]
        dq_MPM = quad(dw_func, -mult, 0, args = (omega, V, 0, -1, 1, -1))[0]
        dq_MMP = quad(dw_func, 0, mult, args = (omega, V, 0, -1, -1, 1))[0]
        g_out0 += (dq_PPP + dq_PMM + dq_MPM + dq_MMP)/Z_out

    dq_PPM = quad(dw_func, -mult,0, args = (omega, V, 0, 1, 1, -1))[0]
    dq_PMP = quad(dw_func, 0, mult, args = (omega, V, 0, 1, -1, 1))[0]
    dq_MPP = quad(dw_func, 0, mult, args = (omega, V, 0, -1, 1, 1))[0]
    dq_MMM = quad(dw_func, -mult, 0, args = (omega, V, 0, -1, -1, -1))[0]
    g_out0 += (dq_PPM + dq_PMP + dq_MPP + dq_MMM)/Z_out
    
    def swap_V(V):
        M = V.copy()
        M[[1,2]] = M[[2,1]]
        M[:,[1,2]] = M[:,[2,1]]
        return M
    def swap_omega(ohm):
        v = ohm.copy()
        v[[1,2]] = v[[2,1]]
        return v
     
    sw_omega = swap_omega(omega)
    sw_V = swap_V(V)   
   
    def g_out_num2(omega, V): 
        omega1 = omega[1]
        omega2 = omega[2]
        g2_num = 0
        if y > 1:
            g2_num += func(0, omega, V, 1, 1, 1)
            g2_num += -func(0, omega, V, 1, -1, -1)
            g2_num += -func(0, omega, V, -1, 1, -1)
            g2_num += func(0, omega, V, -1, -1, 1)
        g2_num += -func(0, omega, V, 1, 1, -1)
        g2_num += func(0, omega, V, 1, -1, 1)
        g2_num += func(0, omega, V, -1, 1, 1)
        g2_num += -func(0, omega, V, -1, -1, -1)
        return g2_num
    
    g_out1 = g_out_num2(sw_omega, sw_V) / Z_out
    g_out2 = g_out_num2(omega, V) / Z_out

    
    dw10_gout_num = 0
    dw20_gout_num = 0
    dw11_gout_num = 0
    dw21_gout_num = 0
    dw22_gout_num = 0
    if y > 1:
        dw10_gout_num += dw_func(0,sw_omega, sw_V, 0, 1, 1, 1) - dw_func(0,sw_omega, sw_V, 0, 1, -1, -1) - dw_func(0,sw_omega, sw_V, 0, -1, 1, -1) + dw_func(0,sw_omega, sw_V, 0, -1, -1, 1)
        dw20_gout_num += dw_func(0,omega, V, 0, 1, 1, 1) - dw_func(0,omega, V, 0, 1, -1, -1) - dw_func(0,omega, V, 0, -1, 1, -1) + dw_func(0,omega, V, 0, -1, -1, 1)
        dw11_gout_num += dw_func(0,sw_omega, sw_V, 2, 1, 1, 1) - dw_func(0,sw_omega, sw_V, 2, 1, -1, -1) - dw_func(0,sw_omega, sw_V, 2, -1, 1, -1) + dw_func(0,sw_omega, sw_V, 2, -1, -1, 1)
        dw21_gout_num += dw_func(0,omega, V, 1, 1, 1, 1) - dw_func(0,omega, V, 1, 1, -1, -1) - dw_func(0,omega, V, 1, -1, 1, -1) + dw_func(0,omega, V, 1, -1, -1, 1)
        dw22_gout_num += dw_func(0,omega, V, 2, 1, 1, 1) - dw_func(0,omega, V, 2, 1, -1, -1) - dw_func(0,omega, V, 2, -1, 1, -1) + dw_func(0,omega, V, 2, -1, -1, 1)
        
    dw10_gout_num += -dw_func(0,sw_omega, sw_V, 0, 1, 1, -1) + dw_func(0,sw_omega, sw_V, 0, 1, -1, 1) + dw_func(0,sw_omega, sw_V, 0, -1, 1, 1) - dw_func(0,sw_omega, sw_V, 0, -1, -1, -1)
    dw20_gout_num += -dw_func(0,omega, V, 0, 1, 1, -1) + dw_func(0,omega, V, 0, 1, -1, 1) + dw_func(0,omega, V, 0, -1, 1, 1) - dw_func(0,omega, V, 0, -1, -1, -1)
    dw11_gout_num += -dw_func(0,sw_omega, sw_V, 2, 1, 1, -1) + dw_func(0,sw_omega, sw_V, 2, 1, -1, 1) + dw_func(0,sw_omega, sw_V, 2, -1, 1, 1) - dw_func(0,sw_omega, sw_V, 2, -1, -1, -1)
    dw21_gout_num += -dw_func(0,omega, V, 1, 1, 1, -1) + dw_func(0,omega, V, 1, 1, -1, 1) + dw_func(0,omega, V, 1, -1, 1, 1) - dw_func(0,omega, V, 1, -1, -1, -1)
    dw22_gout_num += -dw_func(0,omega, V, 2, 1, 1, -1) + dw_func(0,omega, V, 2, 1, -1, 1) + dw_func(0,omega, V, 2, -1, 1, 1) - dw_func(0,omega, V, 2, -1, -1, -1)

    dw00_gout_num = 0
    if y > 1:
        d2q_PPP = quad(dw00_func, 0, mult, args = (omega, V, 1, 1, 1))[0]
        d2q_PMM = quad(dw00_func, -mult, 0, args = (omega, V, 1, -1, -1))[0]
        d2q_MPM = quad(dw00_func, -mult, 0, args = (omega, V, -1, 1, -1))[0]
        d2q_MMP = quad(dw00_func, 0, mult, args = (omega, V, -1, -1, 1))[0]
        dw00_gout_num += d2q_PPP + d2q_PMM + d2q_MPM + d2q_MMP

    d2q_PPM = quad(dw00_func, -mult,0, args = (omega, V, 1, 1, -1))[0]
    d2q_PMP = quad(dw00_func, 0, mult, args = (omega, V, 1, -1, 1))[0]
    d2q_MPP = quad(dw00_func, 0, mult, args = (omega, V, -1, 1, 1))[0]
    d2q_MMM = quad(dw00_func, -mult, 0, args = (omega, V, -1, -1, -1))[0]
    dw00_gout_num += d2q_PPM + d2q_PMP + d2q_MPP + d2q_MMM
        
    #assembling g_out and dg_out
    g_out = np.array([g_out0, g_out1, g_out2])
    d2Z_out = np.zeros((3,3))
    d2Z_out[0,0] = dw00_gout_num
    d2Z_out[0,1] = dw10_gout_num
    d2Z_out[1,0] = dw10_gout_num
    d2Z_out[0,2] = dw20_gout_num
    d2Z_out[2,0] = dw20_gout_num
    d2Z_out[1,1] = dw11_gout_num
    d2Z_out[2,1] = dw21_gout_num
    d2Z_out[1,2] = dw21_gout_num
    d2Z_out[2,2] = dw22_gout_num
    dg_out = d2Z_out / Z_out - np.outer(g_out, g_out) 

    return g_out, dg_out


def channel(Y, W, V_, eps = 1e-6, mult = 10): #vectorized version
    s = len(Y)
    g = []
    dg = []
    for i in range(s):
        y = Y[i]
        w = W[i]
        V = np.linalg.inv(V_[i])
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

def iterate_gamp(X, y, w0=None, max_iter=50, tol=1e-4, 
                 damp=0, early_stopping=False, verbose=1, test_err = False, plant = 0, mult = 10):
    '''
    Note: implementation assuming A commutes with C
    '''
    # Preprocessing
    y_size, x_size = W.shape #W sono i dati
    X2 = X * X   
    K = 3
    alpha = y_size/x_size
        
    # Initialisation
    what = np.sqrt(plant)*w0.T + np.sqrt(1-plant)*np.random.randn(x_size, K)
    
    g = np.zeros((y_size, K))
    chat = np.stack(x_size*[np.identity(K)], axis=0)
    
    count = 0
    train_error = np.zeros(max_iter+1)
    test_error = np.zeros(max_iter+1)
    m = np.zeros((max_iter+1, K, K))
    
    m[0] =  np.tensordot(what, w0, axes=([0, 1])) / x_size     
    train_error[0] = get_error(y, X, what)
    
    if test_err:
        X_test, y_test = sample_labels(w0, alpha)
        test_error[0] = get_error(y_test, X_test, what)
    
    for t in range(max_iter):
        # First part
        V = np.tensordot(X2,chat, axes=([1,0]))
        tmp = np.tensordot(X2, chat, axes=([1,0]))
        omega = np.tensordot(X, what, axes=([1,0])) - np.einsum('ijk,ik->ij', tmp, g)
        
        g, dg = channel(y, omega, V, mult = mult)
        
        # Second part
        A = - np.tensordot(X2, dg, axes=([0, 0])) 
        b = np.einsum('ijk,ik->ij', A, what) + np.tensordot(X, g, axes=([0, 0]))
    
        what_old = what.copy()
        chat_old = chat.copy()
        what_t, chat_t = prior(b,A)      

        what = damping(what_t, what_old, damp)
        chat = damping(chat_t, chat_old, damp)
        
        diff = np.linalg.norm(np.abs(what-what_old))/np.sqrt(x_size)
        train_error[t+1] =  get_error(y, X, what)

        m[t+1] = np.tensordot(what, w0, axes=([0, 1])) / x_size
        print(m[t+1], 'Truth OVERLAP')
        if (early_stopping) and (t>1) and (train_error[t]-train_error[t-1] > 0):
            count += 1
        else:
            count = 0
        if test_err:
           X_fresh, y_fresh = sample_labels(w0, alpha)
           test_error[t+1] = get_error(y_fresh, X_fresh, what) 
            
        if verbose:
            if test_err:
                print('t: {}, diff: {}, train_error: {}, test_err: {}'.format(t, diff, train_error[t+1], test_error[t+1]))
            else:
                print('t: {}, diff: {}, error: {}'.format(t, diff, train_error[t+1]))
        
        if (diff < tol) or (train_error[t+1] < tol):
            break
        

    return what, m[:t+2], train_error[:t+2],test_error[:t+2], t+2







def main(d, alpha, max_iter=200, plant=0.9, damp=0.2, mult=7):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    w0, X, y = sample_instance(d, alpha)

    _, m, mses, tests, _ = iterate_gamp(X, y, w0=w0, max_iter=max_iter, tol = 1e-3, test_err = True, mult = mult, damp = damp, plant = plant)

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

        np.save(f"data_AMP_phase/MS_{d}_{alpha}", MS)
        np.save(f"data_AMP_phase/TESTS_{d}_{alpha}", TESTS)


if __name__=="__main__":
    d = 500
    alpha = float(sys.argv[1])
    max_iter = 100
 
    main(d, alpha, max_iter=max_iter)

 