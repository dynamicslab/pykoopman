"""module for example dynamics data"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orth


def drss(
    n=2, p=2, m=2, p_int_first=0.1, p_int_others=0.01, p_repeat=0.05, p_complex=0.5
):
    """
    Create a discrete-time, random, stable, linear state space model.

    Args:
        n (int, optional): Number of states. Default is 2.
        p (int, optional): Number of control inputs. Default is 2.
        m (int, optional): Number of output measurements.
            If m=0, C becomes the identity matrix, so that y=x. Default is 2.
        p_int_first (float, optional): Probability of an integrator as the first pole.
            Default is 0.1.
        p_int_others (float, optional): Probability of other integrators beyond the
            first. Default is 0.01.
        p_repeat (float, optional): Probability of repeated roots. Default is 0.05.
        p_complex (float, optional): Probability of complex roots. Default is 0.5.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: A tuple containing the
        state transition matrix (A), control matrix (B), and measurement matrix (C).

        A (numpy.ndarray): State transition matrix of shape (n, n).
        B (numpy.ndarray): Control matrix of shape (n, p).
        C (numpy.ndarray): Measurement matrix of shape (m, n). If m = 0, C is the
            identity matrix.

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
    """
    Simulate the linear system dynamics for a given number of steps.

    Args:
        x0 (numpy.ndarray): Initial state vector of shape (n,).
        u (numpy.ndarray): Control input array of shape (p,) or (p, n-1).
            If 1-dimensional, it will be converted to a row vector.
        n (int): Number of steps to simulate.
        A (numpy.ndarray, optional): State transition matrix of shape (n, n).
            If not provided, it defaults to None.
        B (numpy.ndarray, optional): Control matrix of shape (n, p).
            If not provided, it defaults to None.
        C (numpy.ndarray, optional): Measurement matrix of shape (m, n).
            If not provided, it defaults to None.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the state trajectory
        (x) and the output trajectory (y).

        x (numpy.ndarray): State trajectory of shape (n, len(x0)).
        y (numpy.ndarray): Output trajectory of shape (n, C.shape[0]).

    """
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


def vdp_osc(t, x, u):
    """
    Compute the dynamics of the Van der Pol oscillator.

    Args:
        t (float): Time.
        x (numpy.ndarray): State vector of shape (2,).
        u (float): Control input.

    Returns:
        numpy.ndarray: Updated state vector of shape (2,).

    """
    y = np.zeros(x.shape)
    y[0, :] = 2 * x[1, :]
    y[1, :] = -0.8 * x[0, :] + 2 * x[1, :] - 10 * (x[0, :] ** 2) * x[1, :] + u
    return y


def rk4(t, x, u, _dt=0.01, func=vdp_osc):
    """
    Perform a 4th order Runge-Kutta integration.

    Args:
        t (float): Time.
        x (numpy.ndarray): State vector of shape (2,).
        u (float): Control input.
        _dt (float, optional): Time step. Defaults to 0.01.
        func (function, optional): Function defining the dynamics. Defaults to vdp_osc.

    Returns:
        numpy.ndarray: Updated state vector of shape (2,).

    """
    # 4th order Runge-Kutta
    k1 = func(t, x, u)
    k2 = func(t, x + k1 * _dt / 2, u)
    k3 = func(t, x + k2 * _dt / 2, u)
    k4 = func(t, x + k1 * _dt, u)
    return x + (_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def square_wave(step):
    """
    Generate a square wave with a period of 60 time steps.

    Args:
        step (int): Current time step.

    Returns:
        float: Square wave value at the given time step.

    """
    return (-1.0) ** (round(step / 30.0))


def sine_wave(step):
    """
    Generate a sine wave with a period of 60 time steps.

    Args:
        step (int): Current time step.

    Returns:
        float: Sine wave value at the given time step.

    """
    return np.sin(round(step / 30.0))


def lorenz(x, t, sigma=10, beta=8 / 3, rho=28):
    """
    Compute the derivative of the Lorenz system at a given state.

    Args:
        x (list): Current state of the Lorenz system [x, y, z].
        t (float): Current time.
        sigma (float, optional): Parameter sigma. Default is 10.
        beta (float, optional): Parameter beta. Default is 8/3.
        rho (float, optional): Parameter rho. Default is 28.

    Returns:
        list: Derivative of the Lorenz system [dx/dt, dy/dt, dz/dt].

    """
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]


def rev_dvdp(t, x, u=0, dt=0.1):
    """
    Reverse dynamics of the Van der Pol oscillator.

    Args:
        t (float): Time.
        x (numpy.ndarray): Current state of the system [x1, x2].
        u (float, optional): Input. Default is 0.
        dt (float, optional): Time step. Default is 0.1.

    Returns:
        numpy.ndarray: Updated state of the system [x1', x2'].

    """
    return np.array(
        [
            x[0, :] - x[1, :] * dt,
            x[1, :] + (x[0, :] - x[1, :] + x[0, :] ** 2 * x[1, :]) * dt,
        ]
    )


class Linear2Ddynamics:
    def __init__(self):
        """
        Initializes a Linear2Ddynamics object.

        """
        self.n_states = 2  # Number of states

    def linear_map(self, x):
        """
        Applies the linear mapping to the input state.

        Args:
            x (numpy.ndarray): Input state.

        Returns:
            numpy.ndarray: Resulting mapped state.

        """
        return np.array([[0.8, -0.05], [0, 0.7]]) @ x

    def collect_data(self, x, n_int, n_traj):
        """
        Collects data by integrating the linear dynamics.

        Args:
            x (numpy.ndarray): Initial state.
            n_int (int): Number of integration steps.
            n_traj (int): Number of trajectories.

        Returns:
            numpy.ndarray: Input data.
            numpy.ndarray: Output data.

        """
        # Init
        X = np.zeros((self.n_states, n_int * n_traj))
        Y = np.zeros((self.n_states, n_int * n_traj))

        # Integrate
        for step in range(n_int):
            y = self.linear_map(x)
            X[:, (step) * n_traj : (step + 1) * n_traj] = x
            Y[:, (step) * n_traj : (step + 1) * n_traj] = y
            x = y

        return X, Y

    def visualize_modes(self, x, phi, eigvals, order=None):
        """
        Visualizes the modes of the linear dynamics.

        Args:
            x (numpy.ndarray): State data.
            phi (numpy.ndarray): Eigenvectors.
            eigvals (numpy.ndarray): Eigenvalues.
            order (list, optional): Order of the modes to visualize. Default is None.

        """
        n_modes = min(10, phi.shape[1])
        fig, axs = plt.subplots(2, n_modes, figsize=(3 * n_modes, 6))
        if order is None:
            index_list = range(n_modes)
        else:
            index_list = order
        j = 0
        for i in index_list:
            axs[0, j].scatter(
                x[0, :],
                x[1, :],
                c=np.real(phi[:, i]),
                marker="o",
                cmap=plt.get_cmap("jet"),
            )
            axs[1, j].scatter(
                x[0, :],
                x[1, :],
                c=np.imag(phi[:, i]),
                marker="o",
                cmap=plt.get_cmap("jet"),
            )
            axs[0, j].set_title(r"$\lambda$=" + "{:2.3f}".format(eigvals[i]))
            j += 1


class torus_dynamics:
    """
    Sparse dynamics in Fourier space on torus.

    Attributes:
        n_states (int): Number of states.
        sparsity (int): Degree of sparsity.
        freq_max (int): Maximum frequency.
        noisemag (float): Magnitude of noise.

    Methods:
        __init__(self, n_states=128, sparsity=5, freq_max=15, noisemag=0.0):
            Initializes a torus_dynamics object.

        setup(self):
            Sets up the dynamics.

        advance(self, n_samples, dt=1):
            Advances the continuous-time dynamics without control.

        advance_discrete_time(self, n_samples, dt, u=None):
            Advances the discrete-time dynamics with or without control.

        set_control_matrix_physical(self, B):
            Sets the control matrix in physical space.

        set_control_matrix_fourier(self, Bhat):
            Sets the control matrix in Fourier space.

        set_point_actuator(self, position=None):
            Sets a single point actuator.

        viz_setup(self):
            Sets up the visualization.

        viz_torus(self, ax, x):
            Visualizes the torus dynamics.

        viz_all_modes(self, modes=None):
            Visualizes all modes.

        modes(self):
            Returns the modes of the dynamics.

        B_effective(self):
            Returns the effective control matrix.

    """

    def __init__(self, n_states=128, sparsity=5, freq_max=15, noisemag=0.0):
        """
        Initializes a torus_dynamics object.

        Args:
            n_states (int, optional): Number of states. Default is 128.
            sparsity (int, optional): Degree of sparsity. Default is 5.
            freq_max (int, optional): Maximum frequency. Default is 15.
            noisemag (float, optional): Magnitude of noise. Default is 0.0.

        """
        self.n_states = n_states
        self.sparsity = sparsity
        self.freq_max = freq_max
        self.noisemag = noisemag
        self.setup()

    def setup(self):
        """
        Sets up the dynamics.

        """
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
        """
        Advances the continuous-time dynamics without control.

        Args:
            n_samples (int): Number of samples to advance.
            dt (float, optional): Time step. Default is 1.

        """
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
        """
        Advances the discrete-time dynamics with or without control.

        Args:
            n_samples (int): Number of samples to advance.
            dt (float): Time step.
            u (array-like, optional): Control input. Default is None.

        """
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
        """
        Sets the control matrix in physical space.

        Args:
            B (array-like): Control matrix in physical space.

        """
        if np.allclose(B.shape, np.array([self.n_states, self.n_states])) is False:
            raise TypeError("Control matrix B has wrong shape.")
        self.B = B
        self.Bhat = np.fft.fft2(B)

    def set_control_matrix_fourier(self, Bhat):
        """
        Sets the control matrix in Fourier space.

        Args:
            Bhat (array-like): Control matrix in Fourier space.

        """
        if np.allclose(Bhat.shape, np.array([self.n_states, self.n_states])) is False:
            raise TypeError("Control matrix Bhat has wrong shape.")
        self.Bhat = Bhat
        self.B = np.real(np.fft.ifft2(self.Bhat))

    def set_point_actuator(self, position=None):
        """
        Sets a single point actuator.

        Args:
            position (array-like, optional): Position of the actuator. Default is None.

        """
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
        """
        Sets up the visualization.

        """
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
        """
        Visualizes the torus dynamics.

        Args:
            ax: Axes object for plotting.
            x (array-like): Dynamics to be visualized.

        Returns:
            surface: Surface plot of the torus dynamics.

        """
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
        """
        Visualizes all modes.

        Args:
            modes (array-like, optional): Modes to be visualized. Default is None.

        Returns:
            fig: Figure object containing the visualizations.

        """
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
        """
        Returns the modes of the dynamics.

        Returns:
            modes (array-like): Modes of the dynamics.

        """
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
        """
        Returns the effective control matrix.

        Returns:
            B_effective (array-like): Effective control matrix.

        """
        Bhat_effective = np.zeros((self.n_states, self.n_states), complex)
        for k in range(self.sparsity):
            control_mode = np.zeros((self.n_states, self.n_states), complex)
            control_mode[self.J[k, 0], self.J[k, 1]] = self.Bhat[
                self.J[k, 0], self.J[k, 1]
            ]
            Bhat_effective += control_mode
        B_effective = np.fft.ifft2(Bhat_effective)

        return B_effective


class slow_manifold:
    """
    Represents the slow manifold class.

    Args:
        mu (float, optional): Parameter mu. Default is -0.05.
        la (float, optional): Parameter la. Default is -1.0.
        dt (float, optional): Time step size. Default is 0.01.

    Attributes:
        mu (float): Parameter mu.
        la (float): Parameter la.
        b (float): Value computed from mu and la.
        dt (float): Time step size.
        n_states (int): Number of states.

    Methods:
        sys(t, x, u): Computes the system dynamics.
        output(x): Computes the output based on the state.
        simulate(x0, n_int): Simulates the system dynamics.
        collect_data_continuous(x0): Collects data from continuous-time dynamics.
        collect_data_discrete(x0, n_int): Collects data from discrete-time dynamics.
        visualize_trajectories(t, X, n_traj): Visualizes the trajectories.
        visualize_state_space(X, Y, n_traj): Visualizes the state space.
    """

    def __init__(self, mu=-0.05, la=-1.0, dt=0.01):
        self.mu = mu
        self.la = la
        self.b = self.la / (self.la - 2 * self.mu)
        self.dt = dt
        self.n_states = 2

    def sys(self, t, x, u):
        """
        Computes the system dynamics.

        Args:
            t (float): Time.
            x (array-like): State.
            u (array-like): Control input.

        Returns:
            array-like: Computed system dynamics.

        """
        return np.array([self.mu * x[0, :], self.la * (x[1, :] - x[0, :] ** 2)])

    def output(self, x):
        """
        Computes the output based on the state.

        Args:
            x (array-like): State.

        Returns:
            array-like: Computed output.

        """
        return x[0, :] ** 2 + x[1, :]

    def simulate(self, x0, n_int):
        """
        Simulates the system dynamics.

        Args:
            x0 (array-like): Initial state.
            n_int (int): Number of integration steps.

        Returns:
            array-like: Simulated trajectory.

        """
        n_traj = x0.shape[1]
        x = x0
        u = np.zeros((n_int, 1))
        X = np.zeros((self.n_states, n_int * n_traj))
        for step in range(n_int):
            y = rk4(0, x, u[step, :], self.dt, self.sys)
            X[:, (step) * n_traj : (step + 1) * n_traj] = y
            x = y
        return X

    def collect_data_continuous(self, x0):
        """
        Collects data from continuous-time dynamics.

        Args:
            x0 (array-like): Initial state.

        Returns:
            tuple: Collected data (X, Y).

        """
        n_traj = x0.shape[1]
        u = np.zeros((1, n_traj))
        X = x0
        Y = self.sys(0, x0, u)
        return X, Y

    def collect_data_discrete(self, x0, n_int):
        """
        Collects data from discrete-time dynamics.

        Args:
            x0 (array-like): Initial state.
            n_int (int): Number of integration steps.

        Returns:
            tuple: Collected data (X, Y).

        """
        n_traj = x0.shape[1]
        x = x0
        u = np.zeros((n_int, n_traj))
        X = np.zeros((self.n_states, n_int * n_traj))
        Y = np.zeros((self.n_states, n_int * n_traj))
        for step in range(n_int):
            y = rk4(0, x, u[step, :], self.dt, self.sys)
            X[:, (step) * n_traj : (step + 1) * n_traj] = x
            Y[:, (step) * n_traj : (step + 1) * n_traj] = y
            x = y
        return X, Y

    def visualize_trajectories(self, t, X, n_traj):
        """
        Visualizes the trajectories.

        Args:
            t (array-like): Time vector.
            X (array-like): State trajectories.
            n_traj (int): Number of trajectories.

        """
        fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=(12, 4))
        for traj_idx in range(n_traj):
            x = X[:, traj_idx::n_traj]
            axs.plot(t[0:100], x[1, 0:100], "k")
        axs.set(ylabel=r"$x_2$", xlabel=r"$t$")

    def visualize_state_space(self, X, Y, n_traj):
        """
        Visualizes the state space.

        Args:
            X (array-like): State trajectories.
            Y (array-like): Output trajectories.
            n_traj (int): Number of trajectories.

        """
        fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=(4, 4))
        for traj_idx in range(n_traj):
            axs.plot(
                [X[0, traj_idx::n_traj], Y[0, traj_idx::n_traj]],
                [X[1, traj_idx::n_traj], Y[1, traj_idx::n_traj]],
                "-k",
            )
        axs.set(ylabel=r"$x_2$", xlabel=r"$x_1$")


class forced_duffing:
    """
    Forced Duffing Oscillator.

    dx1/dt = x2
    dx2/dt = -d*x2-alpha*x1-beta*x1^3 + u

    [1] S. Peitz, S. E. Otto, and C. W. Rowley,
    “Data-driven model predictive control using interpolated koopman generators,”
    SIAM J. Appl. Dyn. Syst., vol. 19, no. 3, pp. 2162–2193, Mar. 2020.
    """

    def __init__(self, dt, d, alpha, beta):
        """
        Initializes the Forced Duffing Oscillator.

        Args:
            dt (float): Time step.
            d (float): Damping coefficient.
            alpha (float): Coefficient of x1.
            beta (float): Coefficient of x1^3.
        """
        self.dt = dt
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.n_states = 2

    def sys(self, t, x, u):
        """
        Defines the system dynamics of the Forced Duffing Oscillator.

        Args:
            t (float): Time.
            x (array-like): State vector.
            u (array-like): Control input.

        Returns:
            array-like: Rate of change of the state vector.
        """
        y = np.array(
            [
                x[1, :],
                -self.d * x[1, :] - self.alpha * x[0, :] - self.beta * x[0, :] ** 3 + u,
            ]
        )
        return y

    def simulate(self, x0, n_int, u):
        """
        Simulates the Forced Duffing Oscillator.

        Args:
            x0 (array-like): Initial state vector.
            n_int (int): Number of time steps.
            u (array-like): Control inputs.

        Returns:
            array-like: State trajectories.
        """
        n_traj = x0.shape[1]
        x = x0
        X = np.zeros((self.n_states, n_int * n_traj))
        for step in range(n_int):
            y = rk4(0, x, u[step, :], self.dt, self.sys)
            X[:, (step) * n_traj : (step + 1) * n_traj] = y
            x = y
        return X

    def collect_data_continuous(self, x0, u):
        """
        Collects continuous data for the Forced Duffing Oscillator.

        Args:
            x0 (array-like): Initial state vector.
            u (array-like): Control inputs.

        Returns:
            tuple: State and output trajectories.
        """
        X = x0
        Y = self.sys(0, x0, u)
        return X, Y

    def collect_data_discrete(self, x0, n_int, u):
        """
        Collects discrete-time data for the Forced Duffing Oscillator.

        Args:
            x0 (array-like): Initial state vector.
            n_int (int): Number of time steps.
            u (array-like): Control inputs.

        Returns:
            tuple: State and output trajectories.
        """
        n_traj = x0.shape[1]
        x = x0
        X = np.zeros((self.n_states, n_int * n_traj))
        Y = np.zeros((self.n_states, n_int * n_traj))
        for step in range(n_int):
            y = rk4(0, x, u[step, :], self.dt, self.sys)
            X[:, (step) * n_traj : (step + 1) * n_traj] = x
            Y[:, (step) * n_traj : (step + 1) * n_traj] = y
            x = y
        return X, Y

    def visualize_trajectories(self, t, X, n_traj):
        """
        Visualizes the state trajectories of the Forced Duffing Oscillator.

        Args:
            t (array-like): Time vector.
            X (array-like): State trajectories.
            n_traj (int): Number of trajectories to visualize.
        """
        fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(12, 4))
        for traj_idx in range(n_traj):
            x = X[:, traj_idx::n_traj]
            axs[0].plot(t, x[0, :], "k")
            axs[1].plot(t, x[1, :], "b")
        axs[0].set(ylabel=r"$x_1$", xlabel=r"$t$")
        axs[1].set(ylabel=r"$x_2$", xlabel=r"$t$")

    def visualize_state_space(self, X, Y, n_traj):
        """
        Visualizes the state space trajectories of the Forced Duffing Oscillator.

        Args:
            X (array-like): State trajectories.
            Y (array-like): Output trajectories.
            n_traj (int): Number of trajectories to visualize.
        """
        fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=(4, 4))
        for traj_idx in range(n_traj):
            axs.plot(
                [X[0, traj_idx::n_traj], Y[0, traj_idx::n_traj]],
                [X[1, traj_idx::n_traj], Y[1, traj_idx::n_traj]],
                "-k",
            )
        axs.set(ylabel=r"$x_2$", xlabel=r"$x_1$")
