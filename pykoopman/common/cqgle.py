"""Module for cubic-quintic Ginzburg-Landau equation."""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft
from scipy.fft import fftfreq
from scipy.fft import ifft

from pykoopman.common.examples import rk4


class cqgle:
    """
    Cubic-quintic Ginzburg-Landau equation solver.

    Solves the equation:
    i*u_t + (0.5 - i * tau) u_{xx} - i * kappa u_{xxxx} + (1-i * beta)|u|^2 u +
    (nu - i * sigma)|u|^4 u - i * gamma u = 0

    Solves the periodic boundary conditions PDE using spectral methods.

    Attributes:
        n_states (int): Number of states.
        x (numpy.ndarray): x-coordinates.
        dt (float): Time step.
        tau (float): Parameter tau.
        kappa (float): Parameter kappa.
        beta (float): Parameter beta.
        nu (float): Parameter nu.
        sigma (float): Parameter sigma.
        gamma (float): Parameter gamma.
        k (numpy.ndarray): Wave numbers.
        dk (float): Wave number spacing.

    Methods:
        sys(t, x, u): System dynamics function.
        simulate(x0, n_int, n_sample): Simulate the system for a given initial
            condition.
        collect_data_continuous(x0): Collect training data pairs in continuous sense.
        collect_one_step_data_discrete(x0): Collect training data pairs in discrete
            sense.
        collect_one_trajectory_data(x0, n_int, n_sample): Collect data for one
            trajectory.
        visualize_data(x, t, X): Visualize the data in physical space.
        visualize_state_space(X): Visualize the data in state space.
    """

    def __init__(
        self,
        n,
        x,
        dt,
        tau=0.08,
        kappa=0,
        beta=0.66,
        nu=-0.1,
        sigma=-0.1,
        gamma=-0.1,
        L=2 * np.pi,
    ):
        self.n_states = n
        self.x = x

        self.tau = tau
        self.kappa = kappa
        self.beta = beta
        self.nu = nu
        self.sigma = sigma
        self.gamma = gamma

        dk = 2 * np.pi / L
        self.k = fftfreq(self.n_states, 1.0 / self.n_states) * dk
        self.dt = dt

    def sys(self, t, x, u):
        xk = fft(x)

        # 1/3 truncation rule
        xk[self.n_states // 6 : 5 * self.n_states // 6] = 0j
        x = ifft(xk)

        tmp_1_k = (0.5 - 1j * self.tau) * (-self.k**2) * xk
        tmp_2_k = -1j * self.kappa * self.k**4 * xk
        tmp_3_k = fft(
            (1 - 1j * self.beta) * abs(x) ** 2 * x
            + (self.nu - 1j * self.sigma) * abs(x) ** 4 * x
        )
        tmp_4_k = -1j * self.gamma * xk

        # return back to physical space
        y = ifft(1j * (tmp_1_k + tmp_2_k + tmp_3_k + tmp_4_k))
        return y

    def simulate(self, x0, n_int, n_sample):
        # n_traj = x0.shape[1]
        x = x0
        u = np.zeros((n_int, 1), dtype=complex)
        X = np.zeros((n_int // n_sample, self.n_states), dtype=complex)
        t = 0
        j = 0
        t_list = []
        for step in range(n_int):
            t += self.dt
            y = rk4(0, x, u[step], self.dt, self.sys)
            if (step + 1) % n_sample == 0:
                X[j] = y
                j += 1
                t_list.append(t)
            x = y
        return X, np.array(t_list)

    def collect_data_continuous(self, x0):
        """
        collect training data pairs - continuous sense.

        given x0, with shape (n_dim, n_traj), the function
        returns dx/dt with shape (n_dim, n_traj)
        """

        n_traj = x0.shape[0]
        u = np.zeros((n_traj, 1))
        X = x0
        Y = []
        for i in range(n_traj):
            y = self.sys(0, x0[i], u[i])
            Y.append(y)
        Y = np.vstack(Y)
        return X, Y

    def collect_one_step_data_discrete(self, x0):
        """
        collect training data pairs - discrete sense.

        given x0, with shape (n_dim, n_traj), the function
        returns system state x1 after self.dt with shape
        (n_dim, n_traj)
        """

        n_traj = x0.shape[0]
        X = x0
        Y = []
        for i in range(n_traj):
            y, _ = self.simulate(x0[i], n_int=1, n_sample=1)
            Y.append(y)
        Y = np.vstack(Y)
        return X, Y

    def collect_one_trajectory_data(self, x0, n_int, n_sample):
        x = x0
        y, _ = self.simulate(x, n_int, n_sample)
        return y

    def visualize_data(self, x, t, X):
        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection=Axes3D.name)
        for i in range(X.shape[0]):
            ax.plot(x, abs(X[i]), zs=t[i], zdir="t", label="time = " + str(i * self.dt))
        # plt.legend(loc='best')
        ax.view_init(elev=35.0, azim=-65, vertical_axis="y")
        ax.set(ylabel=r"$mag. of. u(x,t)$", xlabel=r"$x$", zlabel=r"time $t$")
        plt.title("CQGLE (Kutz et al., Complexity, 2018)")
        plt.show()

    def visualize_state_space(self, X):
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        # this is a pde problem so the number of snapshots are smaller than dof
        pca_1_r, pca_1_i = np.real(u[:, 0]), np.imag(u[:, 0])
        pca_2_r, pca_2_i = np.real(u[:, 1]), np.imag(u[:, 1])
        pca_3_r, pca_3_i = np.real(u[:, 2]), np.imag(u[:, 2])

        plt.figure(figsize=(6, 6))
        plt.semilogy(s)
        plt.xlabel("number of SVD terms")
        plt.ylabel("singular values")
        plt.title("PCA singular value decays")
        plt.show()

        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection=Axes3D.name)
        ax.plot3D(pca_1_r, pca_2_r, pca_3_r, "k-o")
        ax.set(xlabel="pc1", ylabel="pc2", zlabel="pc3")
        plt.title("PCA visualization (real)")
        plt.show()

        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection=Axes3D.name)
        ax.plot3D(pca_1_i, pca_2_i, pca_3_i, "k-o")
        ax.set(xlabel="pc1", ylabel="pc2", zlabel="pc3")
        plt.title("PCA visualization (imag)")
        plt.show()


if __name__ == "__main__":
    n = 512
    x = np.linspace(-10, 10, n, endpoint=False)
    u0 = np.exp(-((x) ** 2))
    # u0 = 2.0 / np.cosh(x)
    # u0 = u0.reshape(-1,1)
    n_int = 9000
    n_snapshot = 300
    dt = 40.0 / n_int
    n_sample = n_int // n_snapshot

    model = cqgle(n, x, dt, L=20)
    X, t = model.simulate(u0, n_int, n_sample)

    print(X.shape)
    print(X[:, -1].max())

    # usage: visualize the data in physical space
    model.visualize_data(x, t, X)
    print(t)

    # usage: visualize the data in state space
    model.visualize_state_space(X)

    # usage: collect continuous data pair: x and dx/dt
    x0_array = np.vstack([u0, u0, u0])
    X, Y = model.collect_data_continuous(x0_array)

    print(X.shape)
    print(Y.shape)

    # usage: collect discrete data pair
    x0_array = np.vstack([u0, u0, u0])
    X, Y = model.collect_one_step_data_discrete(x0_array)

    print(X.shape)
    print(Y.shape)

    # usage: collect one trajectory data
    X = model.collect_one_trajectory_data(u0, n_int, n_sample)
    print(X.shape)
