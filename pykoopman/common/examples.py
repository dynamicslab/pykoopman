import numpy as np
from scipy.linalg import orth

def drss(n=2, p=2, m=2,
         p_int_first=0.1, p_int_others=0.01,
         p_repeat=0.05, p_complex=0.5):
    """
    Create discrete-time, random, stable, linear state space model.

    :math:`x_{k+1} = Ax_k + Bu_k`
    :math:`y_k = Cx_k`

    Parameters
    ----------
    n : int (default=2)
        Number of states.
    p : int (default=2)
        Number of control inputs.
    m : int (default=2)
        Number of output measurements.
        If m=0, C becomes the identity matrix, so that y=x.
    p_int_first : float (default=0.1)
        Probability of an integrator
    p_int_others : float (default=0.01)
        Probability of other integrators beyond the first
    p_repeat : float (default=0.05)
        Probability of repeated roots
    p_complex : float (default=0.5)
        Probability of complex roots

    Returns
    -------
    A : numpy.ndarray, shape (n, n)
        State transition matrix.
    B : numpy.ndarray, shape (n, p)
        Control matrix.
    C : numpy.ndarray, shape (m, n)
        Measurement matrix.
        If m = 0, C is identity matrix, so that output y = x.

    """

    # Number of integrators
    nint = int((np.random.rand(1)<p_int_first)+sum(np.random.rand(n-1)<p_int_others));
    # Number of repeated roots
    nrepeated = int(np.floor(sum(np.random.rand(n-nint)<p_repeat)/2));
    # Number of complex roots
    ncomplex = int(np.floor(sum(np.random.rand(n-nint-2*nrepeated,1)<p_complex)/2));
    nreal = n-nint-2*nrepeated-2*ncomplex;

    # Random poles
    rep = 2*np.random.rand(nrepeated)-1;
    if ncomplex != 0:
        mag = np.random.rand(ncomplex);
        cplx = np.zeros(ncomplex,dtype=complex)
        for i in range(ncomplex):
            tmp = np.exp(complex(0,np.pi*np.random.rand(1)))
            cplx[i] = mag[i]*np.exp(complex(0,np.pi*np.random.rand(1)))
        re = np.real(cplx);
        im = np.imag(cplx);

    # Generate random state space model
    A = np.zeros((n,n))
    if ncomplex != 0:
        for i in range(0,ncomplex):
            A[2*i:2*i+2,2*i:2*i+2] = np.array([[re[i],im[i]],[-im[i],re[i]]])

    if 2*ncomplex<n:
        list_poles = []
        if nint:
            list_poles = np.append(list_poles, np.ones(nint))
        if rep:
            list_poles = np.append(list_poles, rep)
            list_poles = np.append(list_poles, rep)
        if nreal:
            list_poles = np.append(list_poles, 2*np.random.rand(nreal)-1)

        A[2*ncomplex:,2*ncomplex:] = np.diag(list_poles)

    T = orth(np.random.rand(n,n));
    A = (np.transpose(T)@(A@T))

    # control matrix
    B = np.random.randn(n,p)
    # mask for nonzero entries in B
    mask = np.random.rand(B.shape[0], B.shape[1])
    B = np.squeeze(np.multiply(B, [(mask<0.75) != 0]))

    # Measurement matrix
    if m is 0:
        C = np.identity(n)
    else:
        C = np.random.randn(m, n)
        mask = np.random.rand(C.shape[0], C.shape[1])
        C = np.squeeze(C * [(mask < 0.75) != 0])

    return A,B,C

def advance_linear_system(x0,u,n,A=None,B=None,C=None):
    y = np.zeros([n,C.shape[0]])
    x = np.zeros([n,len(x0)])
    x[0,:] = x0
    y[0,:] = C.dot(x[0,:])
    for i in range(n-1):
        x[i+1,:] = A.dot(x[i,:]) + B.dot(u[:,i])
        y[i+1,:] = C.dot(x[i+1,:])
    return x,y