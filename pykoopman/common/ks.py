"""module for 1D KS equation"""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft
from scipy.fft import fftfreq
from scipy.fft import ifft


class ks:
    """
    Solving 1D KS equation

    u_t = -u*u_x + u_{xx} + nu*u_{xxxx}

    Periodic B.C. between 0 and 2*pi. This PDE is solved
    using spectral methods.
    """

    def __init__(self, n, x, nu, dt, M=16):
        self.n_states = n
        self.dt = dt
        self.x = x
        dk = 1
        k = fftfreq(self.n_states, 1.0 / self.n_states) * dk
        k[n // 2] = 0.0
        L = k**2 - nu * k**4
        self.E = np.exp(self.dt * L)
        self.E2 = np.exp(self.dt * L / 2.0)
        # self.M = M
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
        r = r.reshape(1, -1)
        r_on_circle = np.repeat(r, n, axis=0)
        LR = self.dt * L
        LR = LR.reshape(-1, 1)
        LR = LR.astype("complex")
        LR = np.repeat(LR, M, axis=1)
        LR += r_on_circle
        self.g = -0.5j * k

        self.Q = self.dt * np.real(np.mean((np.exp(LR / 2.0) - 1) / LR, axis=1))
        self.f1 = self.dt * np.real(
            np.mean(
                (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=1
            )
        )
        self.f2 = self.dt * np.real(
            np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=1)
        )
        self.f3 = self.dt * np.real(
            np.mean(
                (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=1
            )
        )

    @staticmethod
    def compute_u2k_zeropad_dealiased(uk_):
        # three over two law
        N = uk_.size
        # map uk to uk_fine
        uk_fine = (
            np.hstack((uk_[0 : int(N / 2)], np.zeros((int(N / 2))), uk_[int(-N / 2) :]))
            * 3.0
            / 2.0
        )
        # convert uk_fine to physical space
        u_fine = np.real(ifft(uk_fine))
        # compute square
        u2_fine = np.square(u_fine)
        # compute fft on u2_fine
        u2k_fine = fft(u2_fine)
        # convert u2k_fine to u2k
        u2k = np.hstack((u2k_fine[0 : int(N / 2)], u2k_fine[int(-N / 2) :])) / 3.0 * 2.0
        return u2k

    def sys(self, t, x, u):
        raise NotImplementedError

    def simulate(self, x0, n_int, n_sample):
        xk = fft(x0)
        u = np.zeros((n_int, 1))
        X = np.zeros((n_int // n_sample, self.n_states))
        t = 0
        j = 0
        t_list = []
        for step in range(n_int):
            t += self.dt
            Nv = self.g * self.compute_u2k_zeropad_dealiased(xk)
            a = self.E2 * xk + self.Q * Nv
            Na = self.g * self.compute_u2k_zeropad_dealiased(a)
            b = self.E2 * xk + self.Q * Na
            Nb = self.g * self.compute_u2k_zeropad_dealiased(b)
            c = self.E2 * a + self.Q * (2.0 * Nb - Nv)
            Nc = self.g * self.compute_u2k_zeropad_dealiased(c)
            xk = self.E * xk + Nv * self.f1 + 2.0 * (Na + Nb) * self.f2 + Nc * self.f3

            if (step + 1) % n_sample == 0:
                y = np.real(ifft(xk)) + self.dt * u[j]
                X[j, :] = y
                j += 1
                t_list.append(t)
                xk = fft(y)

        return X, np.array(t_list)

    def collect_data_continuous(self, x0):
        raise NotImplementedError

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
            ax.plot(x, X[i], zs=t[i], zdir="t", label="time = " + str(i * self.dt))
        ax.view_init(elev=35.0, azim=-65, vertical_axis="y")
        ax.set(ylabel=r"$u(x,t)$", xlabel=r"$x$", zlabel=r"time $t$")
        plt.title("1D K-S equation")
        plt.show()

    def visualize_state_space(self, X):
        u, s, vt = np.linalg.svd(X, full_matrices=False)
        plt.figure(figsize=(6, 6))
        plt.semilogy(s)
        plt.xlabel("number of SVD terms")
        plt.ylabel("singular values")
        plt.title("PCA singular value decays")
        plt.show()

        # this is a pde problem so the number of snapshots are smaller than dof
        pca_1, pca_2, pca_3 = u[:, 0], u[:, 1], u[:, 2]
        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection=Axes3D.name)
        ax.plot3D(pca_1, pca_2, pca_3, "k-o")
        ax.set(xlabel="pc1", ylabel="pc2", zlabel="pc3")
        plt.title("PCA visualization")
        plt.show()


if __name__ == "__main__":
    n = 256
    x = np.linspace(0, 2.0 * np.pi, n, endpoint=False)
    u0 = np.sin(x)
    nu = 0.01
    n_int = 1000
    n_snapshot = 500
    dt = 4.0 / n_int
    n_sample = n_int // n_snapshot

    model = ks(n, x, nu=nu, dt=dt)
    X, t = model.simulate(u0, n_int, n_sample)
    print(X.shape)
    model.visualize_data(x, t, X)

    # usage: visualize the data in state space
    model.visualize_state_space(X)

    # usage: collect discrete data pair
    x0_array = np.vstack([u0, u0, u0])
    X, Y = model.collect_one_step_data_discrete(x0_array)

    print(X.shape)
    print(Y.shape)

    # usage: collect one trajectory data
    X = model.collect_one_trajectory_data(u0, n_int, n_sample)
    print(X.shape)
