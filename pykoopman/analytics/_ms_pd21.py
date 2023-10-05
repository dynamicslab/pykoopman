"""Module for implementing Pan-Duraisamy modes selection algorithm"""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.linear_model import enet_path

from ._base_analyzer import BaseAnalyzer
from ._pruned_koopman import PrunedKoopman
from pykoopman.koopman import Koopman


class ModesSelectionPAD21(BaseAnalyzer):
    """Koopman V selection algorithm from Pan, et al.,JFM (2021).

    Aims to extract a low-dimensional Koopman invariant subspace
    in a model-agnostic way, i.e., applies to any algorithms for
    approximating the Koopman operator.

    See the following reference for more details:
        Pan, S., Arnold-Medabalimi, N., & Duraisamy, K. (2021).
        Sparsity-promoting algorithms for the discovery of informative
        Koopman-invariant subspaces. Journal of Fluid Mechanics, 917.
        <https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/
        article/sparsitypromoting-algorithms-for-the-discovery-of-informative-
        koopmaninvariant-subspaces/F52F03ED181023369A661EF09B57281A>

    Parameters:
        model (Koopman): An instance of `pykoopman.koopman.Koopman`.
        validate_data_traj (list): A list of dictionaries that contains validation
            trajectories. Each dictionary has two keys: `t` and `x`, which correspond to
             time stamps and data matrix.
        truncation_threshold (float): When sweeping the `alpha` in the sparse linear
            regression solver over the data, we consider any term with an absolute
            coefficient smaller than the truncation threshold as zero. Default is 1e-3.
        max_terms_allowed (int): The maximum number of terms used to perform sparse
            linear regression. It can be inferred by the Q-R plot. Default is 10.
        plot (bool): True if the figures should be plotted. Default is False.

    Attributes:
        L (int): Total number of terms considered in sparse linear regression.
        dir (str): The path where figures are saved.
        eigenfunction_on_traj_total_top_k (numpy.ndarray): Evaluations of the best k
            eigenfunctions evaluated on all of the validation trajectories.
        small_to_large_error_eigen_index (numpy.ndarray): Indices of eigenmodes from
            best to worst in terms of linear evolution error.
        sweep_index_list (list): A list of a list of bool. It is the final result of
            sweeping in the sparse linear regression. It contains which V are
            selected at a certain `alpha`.
        validate_data_traj (list): A list of dictionaries that contains validation
            trajectories. Each dictionary has two keys: `t` and `x`, which correspond
            to time stamps and data matrix.
        truncation_threshold (float): When sweeping the `alpha` in the sparse linear
            regression solver over the data, we consider any term with an absolute
            coefficient smaller than the truncation threshold as zero.
    """

    def __init__(
        self,
        model: Koopman,
        validate_data_traj: list,
        truncation_threshold=1e-3,
        max_terms_allowed=10,
        plot=False,
    ):
        """Initialize the ModesSelectionPAD21 object.

        Args:
            model (Koopman): An instance of `pykoopman.koopman.Koopman`.
            validate_data_traj (list): A list of dictionaries that contains
                validation trajectories. Each dictionary should have the keys 't'
                and 'x', corresponding to time stamps and data matrix, respectively.
            truncation_threshold (float, optional): When sweeping the alpha in the
                sparse linear regression solver over the data, we consider any term
                with an absolute coefficient smaller than the truncation threshold as
                zero. Defaults to 1e-3.
            max_terms_allowed (int, optional): The maximum number of terms used to
                perform sparse linear regression. It can be inferred by the Q-R plot.
                Defaults to 10.
            plot (bool, optional): Set to True to plot the figures. Defaults to False.
        """

        super().__init__(model)

        self.validate_data_traj = validate_data_traj
        self.truncation_threshold = truncation_threshold
        self.dir = "/"

        if not isinstance(validate_data_traj, list):
            raise NotImplementedError("validate_data_traj should be a list.")

        # loop over each validation trajectory
        Q_i = []
        for validate_data_one_traj in validate_data_traj:
            validate_data = validate_data_one_traj["x"]
            validate_time = validate_data_one_traj["t"]

            # 1. residual of linearity equation
            linear_residual_list = self._compute_phi_minus_phi_evolved(
                validate_time, validate_data
            )

            # 1.1 normalization factor for each psi
            eigenfunction_evaluated_on_traj = self.eigenfunction(validate_data.T).T

            tmp = np.abs(eigenfunction_evaluated_on_traj) ** 2  # pointwise square only
            dt_arr = np.diff(validate_time, prepend=validate_time[0] - validate_time[1])
            tmp = np.dot(dt_arr, tmp) / (validate_time[-1] - validate_time[0])
            normal_constant = np.sqrt(tmp)

            # 1.2 normalized residual and pick the maximized one over t
            tmp = [
                np.max(np.abs(tmp) / normal_constant[i])
                for i, tmp in enumerate(linear_residual_list)
            ]
            Q_i.append(tmp)

        # compute the mean Q_i for all of the trajectories
        Q_i_mean = np.array(Q_i).mean(axis=0)

        # sort Q to get i_1 to i_L
        self.small_to_large_error_eigen_index = np.argsort(Q_i_mean)[
            : max_terms_allowed + 1
        ]

        # loop over each validation trajectory - for the second time
        R_i = []
        for validate_data_one_traj in validate_data_traj:
            validate_data = validate_data_one_traj["x"]
            eigenfunction_evaluated_on_traj = self.eigenfunction(validate_data.T).T

            # get reconstruction error with increasing number of V
            R_i_each = []
            for k in range(1, max_terms_allowed + 1):
                eigenfunction_evaluated_on_traj_top_k = eigenfunction_evaluated_on_traj[
                    :, self.small_to_large_error_eigen_index[:k]
                ]
                sparse_measurement_matrix = np.linalg.lstsq(
                    eigenfunction_evaluated_on_traj_top_k, validate_data
                )[0]
                residual = (
                    eigenfunction_evaluated_on_traj_top_k @ sparse_measurement_matrix
                    - validate_data
                )
                normalized_err_top_k = np.linalg.norm(residual) / np.linalg.norm(
                    validate_data
                )
                R_i_each.append(normalized_err_top_k)
            R_i_each = np.array(R_i_each)
            R_i.append(R_i_each)
        R_i_mean = np.array(R_i).mean(axis=0)

        # print out the Q-R
        QR_table = PrettyTable()

        QR_table.field_names = ["Index", "Eigenvalue", "Q", "R"]
        tmp = self.eigenvalues_discrete[self.small_to_large_error_eigen_index]
        for i in range(len(R_i_mean)):
            QR_table.add_row(
                [
                    self.small_to_large_error_eigen_index[i],
                    tmp[i],
                    Q_i_mean[i],
                    R_i_mean[i],
                ]
            )
        print(QR_table)

        # prepare top max k selected eigentraj
        eigenfunction_evaluated_on_traj_total = np.vstack(
            [self.eigenfunction(tmp1["x"].T).T for tmp1 in validate_data_traj]
        )
        self.eigenfunction_on_traj_total_top_k = eigenfunction_evaluated_on_traj_total[
            :, self.small_to_large_error_eigen_index[: max_terms_allowed + 1]
        ]

        if plot:
            fig = plt.figure(figsize=(6, 6))
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax1.plot(
                range(1, len(Q_i_mean) + 1),
                Q_i_mean[self.small_to_large_error_eigen_index],
                "b-^",
                label="Max relative psi error",
            )
            ax1.set_xlabel("Number of eigenmodes included", fontsize=16)
            ax1.set_yscale("log")
            ax1.set_ylabel(
                "Max linear evolving normalized error",
                color="b",
                fontsize=16,
            )
            ax2.plot(
                np.arange(1, len(R_i_mean) + 1),
                R_i_mean,
                "r-o",
                label="Reconstruction normalized error",
            )
            ax2.set_ylabel("Reconstruction normalized error", color="r", fontsize=16)
            ax2.set_yscale("log")
            plt.grid("both")
            # annotate the lamda
            for i in range(len(Q_i_mean)):
                ax1.text(
                    i,
                    Q_i_mean[self.small_to_large_error_eigen_index][i],
                    "{:.2f}".format(
                        self.eigenvalues_discrete[
                            self.small_to_large_error_eigen_index
                        ][i]
                    ),
                    size=10,
                    rotation=25,
                )
            plt.tight_layout()
            plt.show()

    def sweep_among_best_L_modes(
        self,
        L: int,
        ALPHA_RANGE=np.logspace(-3, 1, 100),
        MAX_ITER=1e5,
        save_figure=True,
        plot=True,
    ):
        """Perform sweeping among the best L modes using multi-task elastic net.

        Computes multi-task elastic net over a list of alpha values and saves the
        coefficients for each path.

        Parameters:
            L (int): The number of eigenmodes considered for sparse linear regression.
            ALPHA_RANGE (numpy.ndarray, optional): An array of alpha values on which to
                perform sparse linear regression. Defaults to np.logspace(-3, 1, 100).
            MAX_ITER (int, optional): Maximum iterations allowed in the coordinate
                descent algorithm. Defaults to 1e5.
            save_figure (bool, optional): True if the figures should be saved. Defaults
                to True.
            plot (bool, optional): True if the figures should be plotted. Defaults to
                True.
        """

        self.L = L
        # options
        TOL = 1e-12
        L1_RATIO = 0.99

        phi_tilde = self.eigenfunction_on_traj_total_top_k[:, :L]
        X = np.vstack([tmp["x"] for tmp in self.validate_data_traj])
        num_alpha = len(ALPHA_RANGE)

        # 1. normalize the features by making modal amplitude to 1 for all features
        phi_tilde_scaled = phi_tilde / np.abs(phi_tilde[0, :])
        assert phi_tilde_scaled.shape == phi_tilde.shape

        # 2. augment the complex AX=B problem into an AX=B problem with real entries
        #    since the current package only supports real number arrays

        a = np.hstack([np.real(phi_tilde_scaled), -np.imag(phi_tilde_scaled)])
        b = np.hstack([np.imag(phi_tilde_scaled), np.real(phi_tilde_scaled)])
        phi_tilde_aug = np.vstack([a, b])
        X_aug = np.vstack([X, np.zeros(X.shape)])
        num_data = X.shape[0]
        alphas_enet, coefs_enet_aug, _ = enet_path(
            phi_tilde_aug,
            X_aug,
            l1_ratio=L1_RATIO,
            tol=TOL,
            max_iter=MAX_ITER,
            alphas=ALPHA_RANGE,
            check_input=True,
            verbose=0,
        )
        num_total_eigen_func = int(coefs_enet_aug.shape[1] / 2)

        # get the real and imaginary parts from the complex solution
        coefs_enet_real = coefs_enet_aug[:, :num_total_eigen_func, :]
        coefs_enet_imag = coefs_enet_aug[:, num_total_eigen_func:, :]
        assert coefs_enet_imag.shape == coefs_enet_real.shape

        # combine them into a complex array for final results
        coefs_enet_comp = coefs_enet_real + 1j * coefs_enet_imag

        # 2.5 remove features that are smaller than 'self.truncation_threshold'
        # of the max, because most often
        for i_alpha in range(coefs_enet_comp.shape[2]):
            for i_target in range(coefs_enet_comp.shape[0]):
                coef_cutoff_value = self.truncation_threshold * np.max(
                    abs(coefs_enet_comp[i_target, :, i_alpha])
                )
                index_remove = (
                    abs(coefs_enet_comp[i_target, :, i_alpha]) < coef_cutoff_value
                )
                coefs_enet_comp[i_target, index_remove, i_alpha] = 0 + 0j

        # 2.7 given the selected features, do LS-refit to remove the bias of any kind
        # of regularization
        for i_alpha in range(coefs_enet_comp.shape[2]):
            bool_non_zero = np.linalg.norm(coefs_enet_comp[:, :, i_alpha], axis=0) > 0
            phi_tilde_scaled_reduced = phi_tilde_scaled[:, bool_non_zero]
            coef_enet_comp_reduced_i_alpha = np.linalg.lstsq(
                phi_tilde_scaled_reduced, X
            )[0]
            coefs_enet_comp[
                :, bool_non_zero, i_alpha
            ] = coef_enet_comp_reduced_i_alpha.T
            coefs_enet_comp[:, np.invert(bool_non_zero), i_alpha] = 0

        # 3. compute residual for parameter sweep to draw the trade-off
        coefs_enet = np.abs(coefs_enet_comp)
        residual_array = []
        for i in range(num_alpha):
            residual = np.linalg.norm(
                X
                - np.matmul(phi_tilde_scaled, coefs_enet_comp[:, :, i].T)[:num_data]
                # computed the augmented but only compare the first half rows
            )
            residual /= np.linalg.norm(X)
            residual_array.append(residual)
        residual_array = np.array(residual_array)

        # compute the number of non-zeros
        num_non_zero_all_alpha = []
        num_target_components = coefs_enet_comp.shape[0]
        for ii in range(coefs_enet_comp.shape[2]):
            non_zero_index_per_alpha = []
            for i_component in range(num_target_components):
                non_zero_index_per_alpha_per_target = abs(
                    coefs_enet_comp[i_component, :, ii]
                ) > 0 * np.max(abs(coefs_enet_comp[i_component, :, ii]))
                non_zero_index_per_alpha.append(non_zero_index_per_alpha_per_target)
            non_zero_index_per_alpha_all_targets = np.logical_or.reduce(
                non_zero_index_per_alpha
            )
            num_non_zero_all_alpha.append(np.sum(non_zero_index_per_alpha_all_targets))
        num_non_zero_all_alpha = np.array(num_non_zero_all_alpha)

        # print a table for non-zero alpha
        sparse_error_table = PrettyTable()
        sparse_error_table.field_names = [
            "Index",
            "Alpha",
            "# Non-zero",
            "Reconstruction Error",
        ]
        for i in range(len(ALPHA_RANGE)):
            sparse_error_table.add_row(
                [
                    i,
                    ALPHA_RANGE[i],
                    num_non_zero_all_alpha[i],
                    residual_array[i],
                ]
            )
        print(sparse_error_table)

        # plot figures
        num_target_components = coefs_enet_comp.shape[0]
        alphas_enet_log_negative = -np.log10(alphas_enet)
        top_k_modes_list = np.arange(L)

        # figure set 1 -- sparsity of Koopman mode in reconstructing each target
        if plot:
            for i_component in range(num_target_components):
                plt.figure(figsize=(6, 6))
                for i in top_k_modes_list:
                    i_s = self.small_to_large_error_eigen_index[i]
                    plt.plot(
                        alphas_enet_log_negative,
                        abs(coefs_enet[i_component, i, :]),
                        "-*",
                        label=r"$\lambda_{}$ = {:.2f}".format(
                            i_s, self.eigenvalues_discrete[i_s]
                        ),
                    )
                max_val = np.max(abs(coefs_enet[i_component, :, -1]))
                min_val = np.min(abs(coefs_enet[i_component, :, -1]))
                diss = (max_val - min_val) / 2
                mean = (max_val + min_val) / 2
                plt.xlabel(r"-$\log_{10}(\alpha)$", fontsize=16)
                plt.ylabel(
                    "Absolute Value of Coefficients",
                    fontsize=16,
                )
                plt.ylim([mean - diss * 3, mean + diss * 3])
                plt.title(r"$x_{}$".format(i_component + 1))
                plt.legend(loc="best")
                if save_figure:
                    plt.savefig(
                        self.dir
                        + "multi-elastic-net-coef-"
                        + str(i_component + 1)
                        + ".png",
                        bbox_inches="tight",
                    )
                    plt.close()
                else:
                    plt.tight_layout()
                    plt.show()

            # figure set 2 -- reconstruction MSE vs alpha
            fig = plt.figure(figsize=(6, 6))
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax1.plot(alphas_enet_log_negative, residual_array, "k*-")
            ax1.set_xlabel(r"-$\log_{10}(\alpha)$", fontsize=16)
            ax1.set_ylabel(
                "Normalized Reconstruction MSE",
                color="k",
                fontsize=16,
            )

            ax2.plot(alphas_enet_log_negative, num_non_zero_all_alpha, "r*-")
            ax2.set_ylabel(
                "Number of Selected Koopman V",
                color="r",
                fontsize=16,
            )

            if save_figure:
                plt.savefig(
                    self.dir + "/multi-elastic-net-mse.png", bbox_inches="tight"
                )
                plt.close()
            else:
                plt.tight_layout()
                plt.show()

        # 4. find the selected index within the top L best eigenmodes for each alpha
        sweep_index_list = []
        for ii, alpha in enumerate(alphas_enet):
            non_zero_index_bool_array = (
                np.linalg.norm(coefs_enet_comp[:, :, ii], axis=0) > 0
            )
            sweep_index_list.append(non_zero_index_bool_array)
        self.sweep_index_list = sweep_index_list

    def prune_model(self, i_alpha, x_train, dt=1):
        """Prune the `pykoopman.koopman.Koopman` model.

        Returns a pruned model that contains most of the functionality of
        the original model.

        Parameters:
            i_alpha (int): The chosen index from the result of sparse linear regression
                to prune the model.
            x_train (numpy.ndarray): The training data, but only the `x`. Used to refit
                the Koopman V since the Koopman eigenmodes are sparsified.
            dt (float, optional): Time step used in the original model. Defaults to 1.

        Returns:
            pruned_model (PrunedKoopman): The pruned model with fewer Koopman V, but
                similar accuracy.
        """
        sweep_bool_index = self.sweep_index_list[i_alpha]
        sweep_index = self.small_to_large_error_eigen_index[: self.L][sweep_bool_index]

        pruned_model = PrunedKoopman(self.model, sweep_index, dt)
        pruned_model = pruned_model.fit(x_train)
        return pruned_model
