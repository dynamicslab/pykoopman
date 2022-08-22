import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orth


def drss(
    n=2, p=2, m=2, p_int_first=0.1, p_int_others=0.01, p_repeat=0.05, p_complex=0.5
):
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
    nint = int(
        (np.random.rand(1) < p_int_first) + sum(np.random.rand(n - 1) < p_int_others)
    )
    # Number of repeated roots
    nrepeated = int(np.floor(sum(np.random.rand(n - nint) < p_repeat) / 2))
    # Number of complex roots
    ncomplex = int(
        np.floor(sum(np.random.rand(n - nint - 2 * nrepeated, 1) < p_complex) / 2)
    )
    nreal = n - nint - 2 * nrepeated - 2 * ncomplex

    # Random poles
    rep = 2 * np.random.rand(nrepeated) - 1
    if ncomplex != 0:
        mag = np.random.rand(ncomplex)
        cplx = np.zeros(ncomplex, dtype=complex)
        for i in range(ncomplex):
            cplx[i] = mag[i] * np.exp(complex(0, np.pi * np.random.rand(1)))
        re = np.real(cplx)
        im = np.imag(cplx)

    # Generate random state space model
    A = np.zeros((n, n))
    if ncomplex != 0:
        for i in range(0, ncomplex):
            A[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.array(
                [[re[i], im[i]], [-im[i], re[i]]]
            )

    if 2 * ncomplex < n:
        list_poles = []
        if nint:
            list_poles = np.append(list_poles, np.ones(nint))
        if rep:
            list_poles = np.append(list_poles, rep)
            list_poles = np.append(list_poles, rep)
        if nreal:
            list_poles = np.append(list_poles, 2 * np.random.rand(nreal) - 1)

        A[2 * ncomplex :, 2 * ncomplex :] = np.diag(list_poles)

    T = orth(np.random.rand(n, n))
    A = np.transpose(T) @ (A @ T)

    # control matrix
    B = np.random.randn(n, p)
    # mask for nonzero entries in B
    mask = np.random.rand(B.shape[0], B.shape[1])
    B = np.squeeze(np.multiply(B, [(mask < 0.75) != 0]))

    # Measurement matrix
    if m == 0:
        C = np.identity(n)
    else:
        C = np.random.randn(m, n)
        mask = np.random.rand(C.shape[0], C.shape[1])
        C = np.squeeze(C * [(mask < 0.75) != 0])

    return A, B, C


def advance_linear_system(x0, u, n, A=None, B=None, C=None):
    if C is None:
        C = np.identity(len(x0))
    if u.ndim == 1:
        u = u[np.newaxis, :]

    y = np.zeros([n, C.shape[0]])
    x = np.zeros([n, len(x0)])
    x[0, :] = x0
    y[0, :] = C.dot(x[0, :])
    for i in range(n - 1):
        x[i + 1, :] = A.dot(x[i, :]) + B.dot(u[:, i])
        y[i + 1, :] = C.dot(x[i + 1, :])
    return x, y


class torus_dynamics:
    """
    Sparse dynamics in Fourier space on torus
    sparsity: degree of sparsity
    n_states : number of states
    freq_max = 15
    """

    def __init__(self, n_states=128, sparsity=5, freq_max=15, noisemag=0.0):
        self.n_states = n_states
        self.sparsity = sparsity
        self.freq_max = freq_max
        self.noisemag = noisemag
        self.setup()

    def setup(self):
        # Initialization in the Fourier space
        xhat = np.zeros((self.n_states, self.n_states), complex)
        # Index of nonzero frequency components
        self.J = np.zeros((self.sparsity, 2), dtype=int)
        IC = np.zeros(self.sparsity)  # Initial condition, real number
        frequencies = np.zeros(self.sparsity)
        damping = np.zeros(self.sparsity)

        IC = np.random.randn(self.sparsity)
        frequencies = np.sqrt(4 * np.random.rand(self.sparsity))
        damping = -np.random.rand(self.sparsity) * 0.1
        for k in range(self.sparsity):
            loopbreak = 0
            while loopbreak != 1:
                self.J[k, 0] = np.ceil(
                    np.random.rand(1) * self.n_states / (self.freq_max + 1)
                )
                self.J[k, 1] = np.ceil(
                    np.random.rand(1) * self.n_states / (self.freq_max + 1)
                )
                if xhat[self.J[k, 0], self.J[k, 1]] == 0.0:
                    loopbreak = 1

            xhat[self.J[k, 0], self.J[k, 1]] = IC[k]

        mask = np.zeros((self.n_states, self.n_states), int)
        for k in range(self.sparsity):
            mask[self.J[k, 0], self.J[k, 1]] = 1

        self.damping = damping
        self.frequencies = frequencies
        self.IC = IC
        self.xhat = xhat
        self.mask = mask

    def advance(self, n_samples, dt=1):
        print("Evolving continuous-time dynamics without control.")
        self.n_samples = n_samples
        self.dt = dt

        # Initilization
        # In physical space
        self.X = np.ndarray((self.n_states**2, self.n_samples))
        # In Fourier space
        self.Xhat = np.ndarray((self.n_states**2, self.n_samples), complex)
        self.time_vector = np.zeros(self.n_samples)

        # if self.noisemag != 0:
        #     self.XhatClean = np.ndarray((self.n_states**2, self.n_samples), complex)
        #     self.XClean = np.ndarray((self.n_states**2, self.n_samples))

        for step in range(self.n_samples):
            t = step * self.dt
            self.time_vector[step] = t
            xhat = np.zeros((self.n_states, self.n_states), complex)
            for k in range(self.sparsity):
                xhat[self.J[k, 0], self.J[k, 1]] = (
                    np.exp((self.damping[k] + 1j * 2 * np.pi * self.frequencies[k]) * t)
                    * self.IC[k]
                )

            if self.noisemag != 0:
                self.XhatClean[:, step] = xhat.reshape(self.n_states**2)
                xClean = np.real(np.fft.ifft2(xhat))
                self.XClean[:, step] = xClean.reshape(self.n_states**2)

            # xRMS = np.sqrt(np.mean(xhat.reshape((self.n_states**2,1))**2))
            # xhat = xhat + self.noisemag*xRMS\
            #           *np.random.randn(xhat.shape[0],xhat.shape[1]) \
            #         + 1j*self.noisemag*xRMS \
            #         *np.random.randn(xhat.shape[0],xhat.shape[1])
            self.Xhat[:, step] = xhat.reshape(self.n_states**2)
            x = np.real(np.fft.ifft2(xhat))
            self.X[:, step] = x.reshape(self.n_states**2)

    def advance_discrete_time(self, n_samples, dt, u=None):
        print("Evolving discrete-time dynamics with or without control.")
        if u is None:
            self.n_control_features_ = 0
            self.U = np.zeros(n_samples)
            self.U = self.U[np.newaxis, :]
            print("No control input provided. Evolving unforced system.")
        else:
            if u.ndim == 1:
                if len(u) > n_samples:
                    u = u[:-1]
                self.U = u[np.newaxis, :]
            elif u.ndim == 2:
                if u.shape[0] > n_samples:
                    u = u[:-1, :]
                self.U = u
            self.n_control_features_ = self.U.shape[1]

        if not hasattr(self, "B"):
            B = np.zeros((self.n_states, self.n_states))
            print(B.shape)
            self.set_control_matrix_physical(B)
            print("Control matrix is not set. Continue with unforced system.")

        self.n_samples = n_samples
        self.dt = dt

        # Initilization
        # In physical space
        self.X = np.ndarray((self.n_states**2, self.n_samples))
        # In Fourier space
        self.Xhat = np.ndarray((self.n_states**2, self.n_samples), complex)
        self.time_vector = np.zeros(self.n_samples)

        # Set initial condition
        xhat0 = np.zeros((self.n_states, self.n_states), complex)
        for k in range(self.sparsity):
            xhat0[self.J[k, 0], self.J[k, 1]] = self.IC[k]
        self.Xhat[:, 0] = xhat0.reshape(self.n_states**2)
        x0 = np.real(np.fft.ifft2(xhat0))
        self.X[:, 0] = x0.reshape(self.n_states**2)

        for step in range(1, self.n_samples, 1):
            t = step * self.dt
            self.time_vector[step] = t
            # self.Xhat[:, step] = np.reshape(self.Bhat * self.U[0,step - 1],\
            #                   self.n_states ** 2)
            # xhat = self.Xhat[:,step].reshape(self.n_states,self.n_states)
            # xhat_prev = \
            # self.Xhat[:, step - 1].reshape(self.n_states, self.n_states)

            # forced torus dynamics linearly evolve in the spectral space, sparsely
            xhat = np.array((self.n_states, self.n_states), complex)
            xhat = self.Xhat[:, step].reshape(self.n_states, self.n_states)
            xhat_prev = self.Xhat[:, step - 1].reshape(self.n_states, self.n_states)
            for k in range(self.sparsity):
                xhat[self.J[k, 0], self.J[k, 1]] = (
                    np.exp(
                        (self.damping[k] + 1j * 2 * np.pi * self.frequencies[k])
                        * self.dt
                    )
                    * xhat_prev[self.J[k, 0], self.J[k, 1]]
                    + self.Bhat[self.J[k, 0], self.J[k, 1]] * self.U[0, step - 1]
                )

            # xhat_prev = self.Xhat[:,step-1].reshape(self.n_states, self.n_states)
            # for k in range(self.sparsity):
            #     xhat[self.J[k,0], self.J[k,1]] += np.exp((self.damping[k] \
            #     + 1j * 2 * np.pi * self.frequencies[k]) * self.dt) \
            #     * xhat_prev[self.J[k,0], self.J[k,1]]

            self.Xhat[:, step] = xhat.reshape(self.n_states**2)
            x = np.real(np.fft.ifft2(xhat))
            self.X[:, step] = x.reshape(self.n_states**2)

    def set_control_matrix_physical(self, B):
        if np.allclose(B.shape, np.array([self.n_states, self.n_states])) is False:
            raise TypeError("Control matrix B has wrong shape.")
        self.B = B
        self.Bhat = np.fft.fft2(B)

    def set_control_matrix_fourier(self, Bhat):
        if np.allclose(Bhat.shape, np.array([self.n_states, self.n_states])) is False:
            raise TypeError("Control matrix Bhat has wrong shape.")
        self.Bhat = Bhat
        self.B = np.real(np.fft.ifft2(self.Bhat))

    def set_point_actuator(self, position=None):
        if position is None:
            position = np.random.randint(0, self.n_states, 2)
        try:
            for i in range(len(position)):
                position[i] = int(position[i])
        except ValueError:
            print("position was not a valid integer.")

        is_position_in_valid_domain = (position >= 0) & (position < self.n_states)
        if all(is_position_in_valid_domain) is False:
            raise ValueError(
                "Actuator position was not a valid integer inside of domain."
            )

        # Control matrix in physical space (single point actuator)
        B = np.zeros((self.n_states, self.n_states))
        B[position[0], position[1]] = 1
        self.set_control_matrix_physical(B)

    def viz_setup(self):
        self.cmap_torus = plt.cm.jet  # bwr #plt.cm.RdYlBu
        self.n_colors = self.n_states
        r1 = 2
        r2 = 1
        [T1, T2] = np.meshgrid(
            np.linspace(0, 2 * np.pi, self.n_states),
            np.linspace(0, 2 * np.pi, self.n_states),
        )
        R = r1 + r2 * np.cos(T2)
        self.Zgrid = r2 * np.sin(T2)
        self.Xgrid = R * np.cos(T1)
        self.Ygrid = R * np.sin(T1)

    def viz_torus(self, ax, x):

        if not hasattr(self, "viz"):
            self.viz_setup()

        norm = mpl.colors.Normalize(vmin=-abs(x).max(), vmax=abs(x).max())
        surface = ax.plot_surface(
            self.Xgrid,
            self.Ygrid,
            self.Zgrid,
            facecolors=self.cmap_torus(norm(x)),
            shade=False,
            rstride=1,
            cstride=1,
        )
        #     m = cm.ScalarMappable(cmap=cmap_torus, norm=norm)
        #     m.set_array([])
        #     plt.colorbar(m)
        # ax.figure.colorbar(surf, ax=ax)
        ax.set_zlim(-3.01, 3.01)
        return surface

    def viz_all_modes(self, modes=None):

        if modes is None:
            modes = self.modes

        if not hasattr(self, "viz"):
            self.viz_setup()

        fig = plt.figure(figsize=(20, 10))
        for k in range(self.sparsity):
            ax = plt.subplot2grid((1, self.sparsity), (0, k), projection="3d")
            self.viz_torus(ax, modes[:, k].reshape(self.n_states, self.n_states))
            plt.axis("off")
        return fig

    @property
    def modes(self):
        modes = np.zeros((self.n_states**2, self.sparsity))

        for k in range(self.sparsity):
            mode_in_fourier = np.zeros((self.n_states, self.n_states))
            mode_in_fourier[self.J[k, 0], self.J[k, 1]] = 1
            modes[:, k] = np.real(
                np.fft.ifft2(mode_in_fourier).reshape(self.n_states**2)
            )

        return modes

    @property
    def B_effective(self):
        Bhat_effective = np.zeros((self.n_states, self.n_states), complex)
        for k in range(self.sparsity):
            control_mode = np.zeros((self.n_states, self.n_states), complex)
            control_mode[self.J[k, 0], self.J[k, 1]] = self.Bhat[
                self.J[k, 0], self.J[k, 1]
            ]
            Bhat_effective += control_mode
        B_effective = np.fft.ifft2(Bhat_effective)

        return B_effective


def vdp_osc(t, x, u):  # Dynamics of Van der Pol oscillator
    y = np.zeros(x.shape)
    y[0, :] = 2 * x[1, :]
    y[1, :] = -0.8 * x[0, :] + 2 * x[1, :] - 10 * (x[0, :] ** 2) * x[1, :] + u
    return y


def rk4(t, x, u, _dt=0.01, func=vdp_osc):
    # 4th order Runge-Kutta
    k1 = func(t, x, u)
    k2 = func(t, x + k1 * _dt / 2, u)
    k3 = func(t, x + k2 * _dt / 2, u)
    k4 = func(t, x + k1 * _dt, u)
    return x + (_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def square_wave(step):
    return (-1.0) ** (round(step / 30.0))


def lorenz(x, t, sigma=10, beta=8 / 3, rho=28):
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]
