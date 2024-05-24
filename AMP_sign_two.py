import numpy as np
from scipy.integrate import quad
from scipy.special import erf
import sys
from mpi4py import MPI

def norm(x, V=1):
    if V == 0:
        return 0
    return np.exp(-x**2/(2*V))/np.sqrt(2*np.pi*V)

def sample_instance(size_x, rows_to_columns, var_noise=0): 

    size_y = int(np.ceil(rows_to_columns * size_x))
    w = np.random.randn(2, size_x)
    X = np.random.randn(size_y, size_x) / np.sqrt(size_x)
    z1 = X.dot(w[0])
    z2 = X.dot(w[1])
    y = np.sign(z1*z2)
    return w, X, y

def sample_labels(w, rows_to_columns, var_noise=0): 
    '''
    w: 2 x d array
    '''
    # Some pre-processing
    K, size_x = w.shape
    size_y = int(np.ceil(rows_to_columns * size_x))
    
    # Generate F and y
    X = np.random.randn(size_y, size_x) / np.sqrt(size_x)
    y = np.sign(X.dot(w[0,:])*X.dot(w[1,:]))
    return X, y

def get_estimate(what, X):
    '''
    what: 2 x d array
    X: n x d array
    '''
    z1 = X.dot(what[0,:])
    z2 = X.dot(what[1,:])
    y = np.sign(z1*z2)
    return y
    
def get_error(y, X, what):
    yhat = get_estimate(what, X)
    return np.mean(((y-yhat))**2)/2

def prior(b, A): #Gaussian prior
    '''
    b: d x K dimensional tensor
    A: d x K x K tensor
    '''
    x_size, K = b.shape
    inv = np.linalg.inv(np.stack(x_size*[np.identity(K)], axis=0)+A)
    
    return np.einsum('ijk,ik->ij', inv, b), inv

def channel2(y,w,V, mult = 10, eps = 1e-6): #non vectorized version
    '''
    y: scalar
    w: K vector
    V: KxK matrix
    '''
    w.reshape(w.shape[0])
    V.reshape(w.shape[0],w.shape[0])
    K = w.shape[0]
    if K != 2:
        print('K neq 2')
        return None
    
    V = (V + V.T)/2 + eps*np.identity(K)

    Det = V[0,0]*V[1,1]-V[0,1]**2
    V_inv = np.linalg.inv(V)
    #Elements of the inverse
    a_ = V_inv[0,0]
    b_ = V_inv[0,1]
    c_ = V_inv[1,1]
    
    I = quad(lambda x: np.sign(x*V[1,1]+w[1])*norm(x, V = 1)*erf((b_*(x*np.sqrt(V[1,1]))-a_*w[0])/np.sqrt(2*a_)), -mult, mult)[0]
    Zout = (1 - y*I)/2
 
    def Z2(w1, w2, a, b, c): 
        return y*norm(w2, V = a*Det)*erf((b*w2+a*w1)/np.sqrt(2*a))
    def Z11(w1,w2,a,b,c): 
        if b == 0:
            return -y*norm(w1, 1/a)*w1*a*erf(w2*np.sqrt(c/2))
        return y*norm(w1, c*Det)*(2*norm(w1+c*w2/b, c/b**2)-w1/(c*Det)*erf((b*w1+c*w2)/np.sqrt(2*c)))
    
    z2 = Z2(w[0], w[1], a_, b_, c_)
    z1 = Z2(w[1], w[0], c_, b_, a_)
    z11 = Z11(w[0], w[1], a_, b_, c_)
    z22 = Z11(w[1], w[0], c_, b_, a_)
    z12 = 2*y*np.exp(-1/2 * (a_*w[0]**2 + 2*b_*w[0]*w[1] + c_*w[1]**2))/(2*np.pi*np.sqrt(Det))

    gout = np.array([z1, z2])/Zout
    dgout = np.array([np.array([z11, z12]), np.array([z12, z22])])/Zout - gout.reshape(2,1)@gout.reshape(1,2)
    return gout, dgout


def channel(Y, W, V_, eps = 1e-6, mult = 10): #vectorized version
    s = len(Y)
    g = []
    dg = []
    for i in range(s):
        y = Y[i]
        w = W[i]
        V = V_[i]
        v1, v2 = channel2(y,w,V, eps = eps, mult = mult)
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

def iterate_gamp(X, y, w0=None, max_iter=200, tol=1e-4, 
                 damp=0, early_stopping=False, verbose=0, test_err = False, plant = 0, mult = 10):
    '''
    Note: implementation assuming A commutes with C
    '''
    # Preprocessing
    y_size, x_size = X.shape #W sono i dati
    X2 = X * X   
    K = 2
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
        A = - np.tensordot(WX, dg, axes=([0, 0])) 
        b = np.einsum('ijk,ik->ij', A, xhat) + np.tensordot(X, g, axes=([0, 0]))
    
        what_old = what.copy() 
        chat_old = chat.copy()
        what_t, chat_t = prior(b,A)
        what = damping(what_t, what_old, damp)
        chat = damping(chat_t, chat_old, damp)
        
        diff = np.linalg.norm(what-what_old)/np.sqrt(x_size)
        train_error[t+1] = get_error(y, X, what)

        m[t+1] = np.tensordot(what, w0, axes=([0, 1])) / x_size
        
        if (early_stopping) and (t>1) and (train_error[t]-train_error[t-1] > 0):
            count += 1
        else:
            count = 0
            
        if count == 5:
            return xhat, m[:t+2], train_error[:t+2],test_error[:t+2], t+2
            
        if test_err:
           X_test, y_test = sample_labels(w0, alpha)
           test_error[t+1] = get_error(y_test, X_test, what) 
            
        if verbose:
            if test_err:
                print('t: {}, diff: {}, train_error: {}, test_err: {}'.format(t, diff, train_error[t+1], test_error[t+1]))
            else:
                print('t: {}, diff: {}, error: {}'.format(t, diff, train_error[t+1]))
        
        if (diff < tol) or (train_error[t+1] < tol):
            break
        

    return what, m[:t+2], train_error[:t+2],test_error[:t+2], t+2







def main(d, alpha, max_iter=200, plant=0.0, damp=0.35, mult=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    w0, X, y = sample_instance(d, alpha)

    _, m, mses, tests, _ = iterate_gamp(X, y, w0=w0, max_iter=max_iter, tol = 1e-4, test_err = True, mult = mult, damp = damp, plant = plant)

    if rank != 0:
        comm.send(m, dest=0)


    if rank == 0:
        MS = []
        MS.append(m)
        for i in range(1, size):
            MS.append(comm.recv(source=i))

        np.save(f"data_AMP/MS_{d}_{alpha}", MS)


if __name__=="__main__":
    d = 500
    alpha = float(sys.argv[1])
    max_iter = 200
 
    main(d, alpha, max_iter=max_iter)

 