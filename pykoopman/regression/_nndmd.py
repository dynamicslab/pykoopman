"""module for implementing a neural network DMD"""
from __future__ import annotations

import pickle
from abc import abstractmethod
from warnings import warn

import lightning as L
import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from pykoopman.regression._base import BaseRegressor


# todo: add the control version


class MaskedMSELoss(nn.Module):
    """
    Calculates the mean squared error (MSE) loss between `output` and `target`, with
    masking based on `target_lens`. The `max_look_forward` will determine the

    Args:
        max_look_forward

    Returns:
        The MSE loss as a scalar tensor.
    """

    def __init__(self, max_look_forward):
        super().__init__()
        self.max_look_forward = torch.tensor(max_look_forward, dtype=torch.int)

    def forward(self, output, target, target_lens):
        """
        Calculates the MSE loss between `output` and `target`, with masking based on
        `target_lens`.

        Args:
            output (torch.Tensor): The output tensor of shape (batch_size,
            sequence_length, features).
            target (torch.Tensor): The target tensor of shape (batch_size,
            sequence_length, features).
            target_lens (torch.Tensor): A tensor of shape (batch_size,) containing the
            sequence lengths for each item in the batch.

        Returns:
            The MSE loss as a scalar tensor.
        """

        # if target is shorter than output, just cut output off
        if target.size(1) < self.max_look_forward:
            output = output[:, : target.size(1), :]

        # Create mask using target_lens
        mask = torch.zeros_like(output, dtype=torch.bool)
        for i, length in enumerate(target_lens):
            if length > self.max_look_forward:
                length_used = self.max_look_forward
            else:
                length_used = length
            mask[i, :length_used, :] = 1

        # Compute squared differences and apply mask
        squared_diff = torch.pow(output - target, 2)
        squared_diff_masked = torch.where(
            mask, squared_diff, torch.zeros_like(squared_diff)
        )

        # Compute the MSE loss
        mse_loss = squared_diff_masked.sum() / mask.sum()

        return mse_loss


class FFNN(nn.Module):
    """A feedforward neural network with customizable architecture and activation
    functions.

    Args:
        input_size (int): The size of the input layer.
        hidden_sizes (list): A list of the sizes of the hidden layers.
        output_size (int): The size of the output layer.
        activations (str): A string for activation functions for every layer.

    Attributes:
        layers (nn.ModuleList): A list of the neural network layers.
    """

    def __init__(self, input_size, hidden_sizes, output_size, activations):
        super(FFNN, self).__init__()

        activations_dict = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "swish": nn.SiLU(),
            "elu": nn.ELU(),
            "mish": nn.Mish(),
            "linear": nn.Identity(),
        }

        # Define the activation
        act = activations_dict[activations]

        # Define the input layer
        self.layers = nn.ModuleList()

        # if linear layer, remove bias
        if activations == "linear":
            bias = False
        else:
            bias = True

        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_size, output_size, bias))
        else:
            self.layers.append(nn.Linear(input_size, hidden_sizes[0], bias))
            if activations != "linear":
                self.layers.append(act)

            # Define the hidden layers
            for i in range(1, len(hidden_sizes)):
                self.layers.append(
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], bias)
                )
                if activations != "linear":
                    self.layers.append(act)

            # Define the last output layer
            bias_last = False  # True  # last layer with bias
            self.layers.append(nn.Linear(hidden_sizes[-1], output_size, bias_last))

    def forward(self, x):
        """Performs a forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor to the neural network.

        Returns:
            torch.Tensor: The output tensor of the neural network.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class BaseKoopmanOperator(nn.Module):
    """Base class for Koopman operator models.

    Args:
        dim (int): The dimension of the state space.
        dt (float, optional): The time step size. Defaults to 1.0.
        init_std (float, optional): The standard deviation of the initializer.
            Defaults to 0.1.

    Attributes:
        dim (int): The dimension of the state space.
        dt (torch.Tensor): The time step size.
        init_std (float): The standard deviation of the initializer.

    Note:
        rule for self.init_std: a number between 0.1 and 10 over dt

    """

    def __init__(
        self,
        dim: int,
        dt: float = 1.0,
        init_std: float = 0.1,
    ):
        """
        Initializes the `BaseKoopmanOperator` instance.
        """
        super().__init__()
        self.dim = dim
        self.register_buffer("dt", torch.tensor(dt))
        self.init_std = init_std

    def forward(self, x):
        """
        Computes the forward pass of the `BaseKoopmanOperator`.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        koopman_operator = self.get_discrete_time_Koopman_Operator()
        xnext = torch.matmul(x, koopman_operator.t())  # following pytorch convention
        return xnext

    def get_discrete_time_Koopman_Operator(self):
        """
        Computes the discrete-time Koopman operator.

        Returns:
            torch.Tensor: The discrete-time Koopman operator.
        """
        return torch.matrix_exp(self.dt * self.get_K())

    @abstractmethod
    def get_K(self):
        """
        Computes the matrix K.

        Returns:
            torch.Tensor: The matrix K.
        """
        pass


class StandardKoopmanOperator(BaseKoopmanOperator):
    """
    Standard Koopman operator that only has a diagonal matrix for the Koopman operator.
    """

    def __init__(self, **kwargs):
        """
        Initializes the StandardKoopmanOperator.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.register_parameter(
            "K",
            torch.nn.Parameter(
                torch.nn.init.trunc_normal_(
                    torch.zeros(self.dim, self.dim), std=self.init_std
                )
            ),
        )

    def get_K(self):
        """
        Computes the Koopman operator.

        Returns:
            The Koopman operator.
        """
        return self.K


class HamiltonianKoopmanOperator(BaseKoopmanOperator):
    """
    Hamiltonian Koopman operator that has an off-diagonal matrix for the Koopman
    operator.
    """

    def __init__(self, **kwargs):
        """
        Initializes the HamiltonianKoopmanOperator.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.register_parameter(
            "off_diagonal",
            torch.nn.Parameter(
                torch.nn.init.trunc_normal_(
                    torch.zeros(self.dim, self.dim), std=self.init_std
                )
            ),
        )

    def get_K(self):
        """
        Computes the Koopman operator.

        Returns:
            The Koopman operator.
        """
        return self.off_diagonal - self.off_diagonal.t()


class DissipativeKoopmanOperator(BaseKoopmanOperator):
    """
    Dissipative Koopman operator that has an off-diagonal and a diagonal matrix for the
    Koopman operator.
    """

    def __init__(self, **kwargs):
        """
        Initializes the DissipativeKoopmanOperator.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.register_parameter(
            "off_diagonal",
            torch.nn.Parameter(
                torch.nn.init.trunc_normal_(
                    torch.zeros(self.dim, self.dim), std=self.init_std
                )
            ),
        )
        self.register_parameter(
            "diagonal",
            torch.nn.Parameter(
                -torch.pow(
                    torch.nn.init.trunc_normal_(
                        torch.zeros(self.dim), std=self.init_std
                    ),
                    2,
                )
            ),
        )

    def get_K(self):
        """
        Computes the Koopman operator.

        Returns:
            The Koopman operator.
        """
        return torch.diag(self.diagonal) + self.off_diagonal - self.off_diagonal.t()


class DLKoopmanRegressor(L.LightningModule):
    """
    Deep Learning Koopman Regressor module using a Feedforward Neural Network
    encoder and decoder to learn the Koopman operator for a given dynamical system.

    Args:
        mode (str): Type of Koopman operator to use - "Standard", "Hamiltonian" or
            "Dissipative". Defaults to None.
        dt (float): Time step of the Koopman operator. Defaults to 1.0.
        look_forward (int): Number of time steps to predict in the future.
            Defaults to 1.
        config_encoder (dict): Dictionary containing encoder configurations
            - input_size, output_size, hidden_sizes, activations. Defaults to {}.
        config_decoder (dict): Dictionary containing decoder configurations
            - input_size, output_size, hidden_sizes, activations. Defaults to {}.
        lbfgs (bool): Use L-BFGS optimizer. Defaults to False.

    Attributes:
        input_size (int): Size of input to the encoder.
        output_size (int): Size of output from the encoder.
        _encoder (FFNN): Feedforward Neural Network encoder.
        _decoder (FFNN): Feedforward Neural Network decoder.
        _koopman_propagator (BaseKoopmanOperator): Type of Koopman operator used.
        look_forward (int): Number of time steps to predict in the future.
        using_lbfgs (bool): Use L-BFGS optimizer.
        masked_loss_metric (MaskedMSELoss): Mean Squared Error Loss function.
    """

    def __init__(
        self,
        mode=None,
        dt=1.0,
        look_forward=1,
        config_encoder=dict(),
        config_decoder=dict(),
        lbfgs=False,
        std_koopman=1e-1,
    ):
        super(DLKoopmanRegressor, self).__init__()

        self.input_size = config_encoder["input_size"]
        self.output_size = config_encoder["output_size"]

        self._encoder = FFNN(
            input_size=config_encoder["input_size"],
            hidden_sizes=config_encoder["hidden_sizes"],
            output_size=config_encoder["output_size"],
            activations=config_encoder["activations"],
        )

        self._decoder = FFNN(
            input_size=config_decoder["input_size"],
            hidden_sizes=config_decoder["hidden_sizes"],
            output_size=config_decoder["output_size"],
            activations=config_decoder["activations"],
        )

        if mode == "Dissipative":
            self._koopman_propagator = DissipativeKoopmanOperator(
                dim=config_encoder["output_size"], dt=dt, init_std=std_koopman
            )
        elif mode == "Hamiltonian":
            self._koopman_propagator = HamiltonianKoopmanOperator(
                dim=config_encoder["output_size"], dt=dt, init_std=std_koopman
            )
        else:
            self._koopman_propagator = StandardKoopmanOperator(
                dim=config_encoder["output_size"], dt=dt, init_std=std_koopman
            )

        self.look_forward = look_forward
        self.using_lbfgs = lbfgs

        # self.masked_loss_metric = MaskedMSELoss(1)
        self.masked_loss_metric = MaskedMSELoss(self.look_forward)

        if self.using_lbfgs:
            self.automatic_optimization = False

            def training_step(batch, batch_idx):
                optimizer = self.optimizers()

                def closure():

                    # unpack batch data
                    x, y, ys = batch

                    # get the max look forward in this batch
                    batch_look_forward = ys.max()

                    # encode x
                    encoded_x = self._encoder(x)

                    # future unroll look_forward
                    phi_seq = self._propagate_encoded_n_steps(
                        encoded_x, n=batch_look_forward
                    )

                    # standard RNN loss
                    decoded_y_seq_rnn = torch.zeros(
                        (x.size(0), self.look_forward, self.input_size),
                        device=self.device,
                    )

                    for i in range(batch_look_forward):
                        decoded_y_seq_rnn[:, i, :] = self._decoder(phi_seq[:, i, :])
                    rnn_loss = self.masked_loss_metric(decoded_y_seq_rnn, y, ys)

                    # autoencoder reconstruction loss
                    # for x
                    decoded_x = self._decoder(encoded_x)
                    rec_loss = torch.nn.functional.mse_loss(decoded_x, x)

                    # for y_seq
                    decoded_y_seq_rec = torch.zeros(
                        (x.size(0), self.look_forward, self.input_size),
                        device=self.device,
                    )
                    for i in range(batch_look_forward):
                        decoded_y_seq_rec[:, i, :] = self._decoder(
                            self._encoder(y[:, i, :])
                        )
                    rec_loss += self.masked_loss_metric(decoded_y_seq_rec, y, ys)

                    loss = rnn_loss + rec_loss

                    optimizer.zero_grad()
                    self.manual_backward(loss)

                    self.log("loss", loss, prog_bar=True)
                    self.log("rec_loss", rec_loss, prog_bar=True)
                    self.log("rnn_loss", rnn_loss, prog_bar=True)

                    return loss

                optimizer.step(closure=closure)

            self.training_step = training_step

        self.save_hyperparameters()

    def forward(self, x, n=1):
        """
        Propagates input tensor through the model to obtain predicted output tensor
        after n steps.

        Args:
            x: Input tensor with shape (batch_size, input_size).
            n (int): Number of steps to propagate.

        Returns:
            decoded: Output tensor with shape (batch_size, output_size).

        """
        encoded = self._encoder(x)
        phi_seq = self._propagate_encoded_n_steps(encoded, n)
        decoded = self._decoder(phi_seq[:, -1, :])
        return decoded

    def forward_all(self, x, n):
        """
        Forward pass of the Koopman Regressor for a given sequence of input states `x`.
        This method returns the decoded sequence for all steps within the horizon `n`.

        Args:
            x (torch.Tensor): The input state sequence with shape `(batch_size, seq_len,
                input_size)`.
            n (int): The maximum horizon for which to generate the output sequence.

        Returns:
            decoded (torch.Tensor): The decoded sequence with shape `(batch_size, n,
                input_size)`.
        """
        encoded = self._encoder(x)
        phi_seq = self._propagate_encoded_n_steps(encoded, n)
        decoded = torch.zeros(x.size(0), n, self.input_size)
        for i in range(n):
            decoded[:, i, :] = self._decoder(phi_seq[:, i, :])
        return decoded

    def _propagate_encoded_n_steps(self, encoded, n):
        """
        Propagates the encoded tensor linearly in the encoded space for n steps.

        Args:
            encoded (torch.Tensor): The encoded tensor of shape (batch_size,
            encoded_size). n (int): The number of steps to propagate.

        Returns:
            torch.Tensor: The propagated encoded tensor of shape (batch_size, n,
            encoded_size).
        """
        encoded_future = []
        for i in range(n):
            encoded = self._koopman_propagator(encoded)
            encoded_future.append(encoded)
        return torch.stack(encoded_future, 1)

    def training_step(self, batch, batch_idx):
        """
        Defines a training step for the DL Koopman Regressor.

        Args:
            batch: tuple of (x, y, ys), representing the input data,
                the true output data, and the sequence length for
                each sample in the batch.
            batch_idx: integer, the index of the batch.

        Returns:
            tensor representing the loss value for this training step.
        """
        # unpack batch data
        x, y, ys = batch

        # get the max look forward in this batch
        batch_look_forward = ys.max()

        # encode x
        encoded_x = self._encoder(x)

        # future unroll look_forward
        phi_seq = self._propagate_encoded_n_steps(encoded_x, n=batch_look_forward)

        # standard RNN loss
        decoded_y_seq_rnn = torch.zeros(
            (x.size(0), self.look_forward, self.input_size), device=self.device
        )

        for i in range(batch_look_forward):
            decoded_y_seq_rnn[:, i, :] = self._decoder(phi_seq[:, i, :])
        rnn_loss = self.masked_loss_metric(decoded_y_seq_rnn, y, ys)

        # autoencoder reconstruction loss
        # for x
        decoded_x = self._decoder(encoded_x)
        rec_loss = torch.nn.functional.mse_loss(decoded_x, x)

        # for y_seq
        decoded_y_seq_rec = torch.zeros(
            (x.size(0), self.look_forward, self.input_size), device=self.device
        )
        for i in range(batch_look_forward):
            decoded_y_seq_rec[:, i, :] = self._decoder(self._encoder(y[:, i, :]))
        rec_loss += self.masked_loss_metric(decoded_y_seq_rec, y, ys)

        loss = rnn_loss + rec_loss

        self.log("loss", loss, prog_bar=True)
        self.log("rec_loss", rec_loss, prog_bar=True)
        self.log("rnn_loss", rnn_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configures and returns the optimizer to use for training.

        If using LBFGS optimizer, set `using_lbfgs` attribute to True when
        initializing the DLKoopmanRegressor instance.

        Returns:
            An instance of torch.optim.Optimizer to use for training.
        """
        if self.using_lbfgs:
            optimizer = torch.optim.LBFGS(
                self.parameters(),
                lr=1,
                history_size=100,
                max_iter=20,
                line_search_fn="strong_wolfe",
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class SeqDataDataset(Dataset):
    """
    A PyTorch Dataset class to handle sequential data in the format of (x, y, ys),
    where x is the input sequence, y is the target output sequence and ys is a vector
    indicating the maximum look-ahead distance.

    Args:
        x (torch.Tensor): The input sequence tensor of shape (batch_size,
            sequence_length, input_size).
        y (torch.Tensor): The output sequence tensor of shape (batch_size,
            sequence_length, output_size).
        ys (torch.Tensor): The maximum look-ahead distance tensor of shape
            (batch_size,).
        transform (callable, optional): Optional normalization function to apply to
            x and y.

    Returns:
        torch.Tensor: The preprocessed input sequence tensor.
        torch.Tensor: The preprocessed target output sequence tensor.
        torch.Tensor: The maximum look-ahead distance tensor.
    """

    def __init__(self, x, y, ys, transform=None):
        self.x = x.squeeze(1)
        self.y = y
        self.ys = ys
        self.normalization = transform

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x = self.x[idx].clone()
        y = self.y[idx].clone()
        ys = self.ys[idx].clone()

        if self.normalization:
            x = self.normalization(x)
            y = self.normalization(y)

        return x, y, ys


class TensorNormalize(nn.Module):
    """
    Normalizes the input tensor by subtracting the mean and dividing by the standard
    deviation.

    Args:
        mean (float or tensor): The mean value to be subtracted from the input tensor.
        std (float or tensor): The standard deviation value to divide the input tensor
            by.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor: torch.Tensor):
        """
        Forward pass of the normalization module.

        Args:
            tensor (tensor): The input tensor to be normalized.

        Returns:
            The normalized tensor.
        """
        return torch.divide((tensor - self.mean), self.std)
        # return # tensor.copy_(tensor.sub_(self.mean).div_(self.std))

    def __repr__(self) -> str:
        """
        Returns a string representation of the TensorNormalize module.

        Returns:
            A string representation of the module.
        """
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class InverseTensorNormalize(nn.Module):
    """
    A PyTorch module that performs inverse normalization on input tensors using
    a given mean and standard deviation.

    Args:
        mean (float or sequence): The mean used for normalization.
        std (float or sequence): The standard deviation used for normalization.

    Example:
        >>> mean = [0.5, 0.5, 0.5]
        >>> std = [0.5, 0.5, 0.5]
        >>> inv_norm = InverseTensorNormalize(mean, std)
        >>> normalized_tensor = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]])
        >>> output = inv_norm(normalized_tensor)

    Attributes:
        mean (float or sequence): The mean used for normalization.
        std (float or sequence): The standard deviation used for normalization.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor: torch.Tensor):
        return torch.multiply(tensor, self.std) + self.mean
        # return tensor.copy_(tensor.mul_(self.std).add_(self.mean))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class SeqDataModule(L.LightningDataModule):
    """
    Class for creating sequence data dataloader for training and validation.

    Args:
        data_tr: List of 2D numpy.ndarray representing training data trajectories.
        data_val: List of 2D numpy.ndarray representing validation data trajectories.
            Can be None.
        look_forward: Number of time steps to predict forward.
        batch_size: Size of each batch of data.
        normalize: Whether to normalize the input data or not. Default is True.
        normalize_mode: The type of normalization to use. Either "equal" or "max".
            Default is "equal".
        normalize_std_factor: Scaling factor for standard deviation during
            normalization. Default is 2.0.

    Methods:
        prepare_data(): Prepares the data by converting to time-delayed data and
            computing mean and std if normalize is True.
        setup(stage=None): Sets up training and validation datasets.
        train_dataloader(): Returns a DataLoader for training data.
        val_dataloader(): Returns a DataLoader for validation data.
        convert_seq_list_to_delayed_data(data_list, look_back, look_forward): Converts
            list of sequences to time-delayed data.
        collate_fn(batch): Custom collate function to be used with DataLoader.

    Returns:
        A SeqDataModule object.
    """

    def __init__(
        self,
        data_tr,
        data_val,
        look_forward=10,
        batch_size=32,
        normalize=True,
        normalize_mode="equal",
        normalize_std_factor=2.0,
    ):
        """
        Initialize a SeqDataModule.

        Args:
            data_tr (Union[str, List[np.ndarray]]): Training data. Can be either a
                list of 2D numpy arrays, each 2D numpy array representing a trajectory,
                or the path to a pickle file containing such a list.
            data_val (Optional[Union[str, List[np.ndarray]]]): Validation data.
                Can be either a list of 2D numpy arrays, each 2D numpy array
                    representing a trajectory, or the path to a pickle file
                    containing such a list.
            look_forward (int): Number of time steps to predict into the future.
            batch_size (int): Number of samples per batch.
            normalize (bool): Whether to normalize the data. Default is True.
            normalize_mode (str): Mode for normalization. Can be either "equal"
                or "max". "equal" divides by the standard deviation, while "max"
                divides by the maximum absolute value of the data. Default is "equal".
            normalize_std_factor (float): Scaling factor for the standard deviation in
                normalization. Default is 2.0.

        Returns:
            None.
        """
        super().__init__()
        # input data_tr or data_val is a list of 2D np.ndarray. each 2d
        # np.ndarray is a trajectory, and the axis 0 is number of samples, axis 1 is
        # the number of system state
        self.data_tr = data_tr
        self.data_val = data_val
        self.look_forward = look_forward
        self.batch_size = batch_size
        self.look_back = 1
        self.normalize = normalize
        self.normalize_mode = normalize_mode
        self.normalization = None
        self.inverse_transform = None
        self.normalize_std_factor = normalize_std_factor

    def prepare_data(self):
        """
        Preprocesses the input training and validation data by checking their types,
        checking for normalization, finding the mean and standard deviation of
        the training data (if normalization is enabled), and creating time-delayed data
        from the input data.

        Raises:
            ValueError: If the training data is None or has an invalid type.
            ValueError: If the validation data has an invalid type.
            TypeError: If the data is complex or not float.

        """
        # train data
        if self.data_tr is None:
            raise ValueError("You must feed training data!")
        if isinstance(self.data_tr, list):
            data_list = self.data_tr
        elif isinstance(self.data_tr, str):
            f = open(self.data_tr, "rb")
            data_list = pickle.load(f)
        else:
            raise ValueError("Wrong type of `self.data_tr`")

        # check train data
        data_list = self.check_list_of_nparray(data_list)

        # find the mean, std
        if self.normalize:
            stacked_data_list = np.vstack(data_list)
            mean = stacked_data_list.mean(axis=0)
            std = stacked_data_list.std(axis=0)

            # zero mean so easier for downstream
            self.mean = torch.FloatTensor(mean) * 0
            # default = 2.0, more stable
            self.std = torch.FloatTensor(std) * self.normalize_std_factor

            if self.normalize_mode == "max":
                self.std = torch.ones_like(self.std) * self.std.max()

            # prevent divide by zero error
            for i in range(len(self.std)):
                if self.std[i] < 1e-6:
                    self.std[i] += 1e-3

            # get transform
            self.normalization = TensorNormalize(self.mean, self.std)

            # get inverse transform
            self.inverse_transform = InverseTensorNormalize(self.mean, self.std)

        # create time-delayed data
        self._tr_x, self._tr_yseq, self._tr_ys = self.convert_seq_list_to_delayed_data(
            data_list, self.look_back, self.look_forward
        )

        # validation data
        if self.data_val is not None:
            # raise ValueError("You need to feed validation data!")
            if isinstance(self.data_val, list):
                data_list = self.data_val
            elif isinstance(self.data_val, str):
                f = open(self.data_val, "rb")
                data_list = pickle.load(f)
            else:
                raise ValueError("Wrong type of `self.data_val`")

            # check val data
            data_list = self.check_list_of_nparray(data_list)

            # create time-delayed data
            (
                self._val_x,
                self._val_yseq,
                self._val_ys,
            ) = self.convert_seq_list_to_delayed_data(
                data_list, self.look_back, self.look_forward
            )
        else:
            warn("Warning: no validation data prepared")

    def setup(self, stage=None):
        """
        Prepares the train and validation datasets for the Lightning module.
        The train dataset is created from the training data specified in the
        constructor by creating time-delayed versions of the input/output sequences.
        If `normalize` is True, the data is normalized using the mean and standard
        deviation of the training data. The validation dataset is created from the
        validation data specified in the constructor in the same way as the training
        dataset. If `normalize` is True, it is also normalized using the mean and
        standard deviation of the training data. If `stage` is not "fit",
        an exception is raised as the `setup()` method has not been implemented
        for other stages.

        Args:
            stage: The stage of training, validation or testing (default is None).

        Raises:
            NotImplementedError: If `stage` is not "fit".
        """
        # Load data and split into train and validation sets here
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.tr_dataset = SeqDataDataset(
                self._tr_x, self._tr_yseq, self._tr_ys, self.normalization
            )
            if self.data_val is not None:
                self.val_dataset = SeqDataDataset(
                    self._val_x, self._val_yseq, self._val_ys, self.normalization
                )
        else:
            raise NotImplementedError("We didn't implement for stage not `fit`")

    def train_dataloader(self):
        return DataLoader(
            self.tr_dataset, self.batch_size, shuffle=True, collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, self.batch_size, shuffle=True, collate_fn=self.collate_fn
        )

    def convert_seq_list_to_delayed_data(self, data_list, look_back, look_forward):
        """
        Converts a list of sequences to time-delayed data by extracting subsequences
        of length `look_back` and `look_forward` from each sequence in the list.

        Args:
            data_list (List[np.ndarray]): A list of 2D numpy arrays. Each array
                represents a trajectory, with axis 0 representing the number of samples
                and axis 1 representing the number of system states.
            look_back (int): The number of previous time steps to include in each
                subsequence.
            look_forward (int): The number of future time steps to include in each
                subsequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three
                tensors:
            1) The time-delayed input data, with shape (num_samples, look_back,
                num_system_states).
            2) The time-delayed output data, with shape (num_samples, look_forward,
                num_system_states).
            3) The sequence lengths of the output data, with shape (num_samples,).
        """
        time_delayed_x_list = []
        time_delayed_yseq_list = []
        for seq in data_list:
            # if self.look_forward + self.look_back > len(seq):
            #     raise ValueError("look_forward too large")
            n_sub_traj = len(seq) - look_back - look_forward + 1
            if n_sub_traj >= 1:
                for i in range(len(seq) - look_back - look_forward + 1):
                    time_delayed_x_list.append(seq[i : i + look_back])
                    time_delayed_yseq_list.append(
                        seq[i + look_back : i + look_back + look_forward]
                    )
            else:
                # only 1 traj, just to predict to its end
                time_delayed_x_list.append(seq[0:1])
                time_delayed_yseq_list.append(seq[1:])
        time_delayed_yseq_lens_list = [x.shape[0] for x in time_delayed_yseq_list]

        # convert data to tensor
        time_delayed_x = torch.FloatTensor(np.array(time_delayed_x_list))
        time_delayed_yseq = pad_sequence(
            [torch.FloatTensor(x) for x in time_delayed_yseq_list], True
        )
        time_delayed_yseq_lens = torch.LongTensor(time_delayed_yseq_lens_list)
        return time_delayed_x, time_delayed_yseq, time_delayed_yseq_lens

    def collate_fn(self, batch):
        """
        Collates a batch of data.

        Args:
            batch: A list of tuples where each tuple represents a sample containing
                the input sequence `x`, the output sequence `y`, and the maximum
                number of steps to predict `ys`.

        Returns:
            A tuple containing the input sequences as a stacked tensor, the output
            sequences as a stacked tensor, and the maximum number of steps to predict
            as a stacked tensor.

        """
        x_batch, y_batch, ys_batch = zip(*batch)
        xx = torch.stack(x_batch, 0)
        yy = torch.stack(y_batch, 0)
        ys = torch.stack(ys_batch, 0)
        return xx, yy, ys

    @classmethod
    def check_list_of_nparray(cls, data_list):
        """
        Check if the input is a list of numpy arrays, and convert data to float32 if
        float64.

        Args:
            data_list (List[np.ndarray]): A list of numpy arrays representing system
                states.

        Returns:
            List[np.ndarray]: The input list of numpy arrays converted to float32.

        Raises:
            TypeError: If the input data is complex or not float.
        """
        # check if data is complex
        if any(np.iscomplexobj(x) for x in data_list):
            raise TypeError("Complex data is not supported")

        # check if data has float64
        if any(x.dtype is np.float64 for x in data_list):
            warn("Found float64 data. Will convert to float32")

        # convert data to float32 if float64
        for i, data_traj in enumerate(data_list):
            if "float" not in data_traj.dtype.name:
                raise TypeError("Found data is not float")
            if data_traj.dtype.name == "float64":
                data_list[i] = data_traj.astype("float32")

        return data_list


class NNDMD(BaseRegressor):
    """Implementation of Nonlinear Dynamic Mode Decomposition (NNDMD).

    Args:
        mode (str): NNDMD mode, `Dissipative` or `Hamiltonian` or else (default: None).
        dt (float): Time step (default: 1.0).
        look_forward (int): Number of steps to look forward (default: 1).
        config_encoder (dict): Configuration for the encoder network
            (default: dict(input_size=2, hidden_sizes=[32]*2, output_size=6,
            activations='tanh')).
        config_decoder (dict): Configuration for the decoder network
            (default: dict(input_size=6, hidden_sizes=[32]*2, output_size=2,
            activations='linear')).
        batch_size (int): Batch size (default: 16).
        lbfgs (bool): Whether to use L-BFGS optimizer (default: False).
        normalize (bool): Whether to normalize data (default: True).
        normalize_mode (str): Normalization mode, `max` or `equal`
            (default: 'equal').
        normalize_std_factor (float): Standard deviation factor for normalization
            (default: 2.0).
        trainer_kwargs (dict): Arguments for the `pytorch_lightning.Trainer`
            (default: {}).

    Attributes:
        coef_ (np.ndarray): Koopman operator coefficients.
        state_matrix_ (np.ndarray): State matrix of the Koopman operator.
        eigenvalues_ (np.ndarray): Eigenvalues of the Koopman operator.
        eigenvectors_ (np.ndarray): Eigenvectors of the Koopman operator.
        ur (np.ndarray): Effective linear transformation.
        unnormalized_modes (np.ndarray): Unnormalized modes.

    Note:
        The `n_samples_` attribute is meaningless for this class.
        The `dt` argument is only included to please the regressor class and has no
            real use.

    """

    def __init__(
        self,
        mode=None,
        dt=1.0,
        look_forward=1,
        config_encoder=dict(
            input_size=2, hidden_sizes=[32] * 2, output_size=6, activations="tanh"
        ),
        config_decoder=dict(
            input_size=6, hidden_sizes=[32] * 2, output_size=2, activations="linear"
        ),
        batch_size=16,
        lbfgs=False,
        normalize=True,
        normalize_mode="equal",
        normalize_std_factor=2.0,
        std_koopman=1e-1,
        trainer_kwargs={},
    ):
        """Initializes the NNDMD model."""
        self.mode = mode
        self.look_forward = look_forward
        self.config_encoder = config_encoder
        self.config_decoder = config_decoder
        self.lbfgs = lbfgs
        self.normalize = normalize
        self.normalize_mode = normalize_mode
        self.dt = dt
        self.trainer_kwargs = trainer_kwargs
        self.normalize_std_factor = normalize_std_factor
        self.batch_size = batch_size
        self.std_koopman = std_koopman

        # build DLK regressor
        self._regressor = DLKoopmanRegressor(
            mode, dt, look_forward, config_encoder, config_decoder, lbfgs, std_koopman
        )

    def fit(self, x, y=None, dt=None):
        """fit the NNDMD model with data x,y

        Args:
            x (np.ndarray or list): The training input data. If a 2D numpy array,
                then it represents a single time-series  and each row represents a
                state, otherwise it should be a list of 2D numpy arrays.
            y (np.ndarray or list, optional): The target output data,
                corresponding to `x`. If `None`, `x` is assumed to contain the target
                data in its second half. Defaults to `None`.
            dt (float, optional): The time step used to generate `x`.
                Defaults to `None`.

        Returns:
            None. The fitted model is stored in the class attribute `_regressor`.
        """
        # build trainer
        self.trainer = L.Trainer(**self.trainer_kwargs)

        self.n_input_features_ = self.config_encoder["input_size"]

        # create the data module
        # case  1: a single traj, x is 2D np.ndarray, no validation
        if y is None and isinstance(x, np.ndarray) and x.ndim == 2:
            t0, t1 = x[:-1], x[1:]
            list_of_traj = [np.stack((t0[i], t1[i]), 0) for i in range(len(x) - 1)]
            self.dm = SeqDataModule(
                list_of_traj,
                None,
                self.look_forward,
                self.batch_size,
                self.normalize,
                self.normalize_mode,
                self.normalize_std_factor,
            )
            self.n_samples_ = len(list_of_traj)

        # case 2: x, y are 2D np.ndarray, no validation
        elif (
            isinstance(x, np.ndarray)
            and isinstance(y, np.ndarray)
            and x.ndim == 2
            and y.ndim == 2
        ):
            t0, t1 = x, y
            list_of_traj = [np.stack((t0[i], t1[i]), 0) for i in range(len(x) - 1)]
            self.dm = SeqDataModule(
                list_of_traj,
                None,
                self.look_forward,
                self.batch_size,
                self.normalize,
                self.normalize_mode,
                self.normalize_std_factor,
            )
            self.n_samples_ = len(list_of_traj)

        # case 3: only training data, x is a list of trajectories, y is None
        elif isinstance(x, list) and y is None:
            self.dm = SeqDataModule(
                x,
                None,
                self.look_forward,
                self.batch_size,
                self.normalize,
                self.normalize_mode,
                self.normalize_std_factor,
            )
            self.n_samples_ = len(x)

        # case 4: x, y are two lists of trajectories, we have validation data
        elif isinstance(x, list) and isinstance(y, list):
            self.dm = SeqDataModule(
                x,
                y,
                self.look_forward,
                self.batch_size,
                self.normalize,
                self.normalize_mode,
                self.normalize_std_factor,
            )
            self.n_samples_ = len(x)
        else:
            raise ValueError("check `x` and `y` for `self.fit`")

        # trainer starts to train
        self.trainer.fit(self._regressor, self.dm)

        # compute Koopman operator information
        self._state_matrix_ = (
            self._regressor._koopman_propagator.get_discrete_time_Koopman_Operator()
            .detach()
            .numpy()
        )
        [self._eigenvalues_, self._eigenvectors_] = np.linalg.eig(self._state_matrix_)

        self._coef_ = self._state_matrix_

        # obtain effective linear transformation
        decoder_weight_list = []
        for i in range(len(self._regressor._decoder.layers)):
            decoder_weight_list.append(
                self._regressor._decoder.layers[i].weight.detach().numpy()
            )
        if len(decoder_weight_list) > 1:
            self._ur = np.linalg.multi_dot(decoder_weight_list[::-1])
        else:
            self._ur = decoder_weight_list[0]

        if self.normalize:
            std = self.dm.inverse_transform.std
            self._ur = np.diag(std) @ self._ur

        self._unnormalized_modes = self._ur @ self._eigenvectors_

    def predict(self, x, n=1):
        """
        Predict the system state after n steps away from x_0 = x.

        Args:
            x (numpy.ndarray or torch.Tensor): Input data of shape
                (n_samples, n_features).
            n (int): Number of steps to predict the system state into the future.

        Returns:
            numpy.ndarray: Predicted system state after n steps, of shape
                (n_samples, n_features).

        Note:
            By default, the model is stored on the CPU for inference.
        """
        self._regressor.eval()
        x = self._convert_input_ndarray_to_tensor(x)

        with torch.no_grad():
            # print("inference device = ", self._regressor.device)

            if self.normalize:
                y = self.dm.normalization(x)
                y = self._regressor(y, n)
                y = self.dm.inverse_transform(y).numpy()
            else:
                y = self._regressor(x, n).numpy()
            return y

    def simulate(self, x, n_steps):
        """
        Simulate the system forward in time for `n_steps` steps starting from `x`.

        Args:
            x (np.ndarray or torch.Tensor): The initial state of the system.
            Should be a 2D array/tensor.
            n_steps (int): The number of time steps to simulate the system forward.

        Returns:
            np.ndarray: The simulated states of the system. Will be of shape
            `(n_steps+1, n_features)`.
        """
        self._regressor.eval()
        x = self._convert_input_ndarray_to_tensor(x)
        x_future = torch.zeros([n_steps + 1, x.size(1)])
        x_future[0] = x
        with torch.no_grad():
            for i in range(n_steps):
                if self.normalize:
                    y = self.dm.normalization(x)
                    y = self._regressor(y, i + 1)
                    x_future[i + 1] = self.dm.inverse_transform(y)
                else:
                    x_future[i + 1] = self._regressor(x, i + 1)

            return x_future.numpy()

    @property
    def A(self):
        """Returns the state transition matrix A of the NNDMD model.

        Returns
        -------
        A : numpy.ndarray
            The state transition matrix of shape (n_states, n_states), where
            n_states is the number of states in the model.
        """
        return self._state_matrix_

    @property
    def B(self):
        # todo: we don't have control considered in nndmd for now
        pass

    @property
    def C(self):
        """
        Returns the matrix C representing the effective linear transformation
        from the observables to the Koopman modes. The matrix C is computed during
        the fit process as the product of the decoder weights of the trained
        autoencoder network.

        Returns:
        --------
        numpy.ndarray of shape (n_koopman, n_features)
            The matrix C.
        """
        return self._ur

    @property
    def W(self):
        """
        Returns the matrix W representing the Koopman modes. The matrix W is computed
        during the fit process as the eigenvectors of the Koopman operator.

        Returns:
        --------
        numpy.ndarray of shape (n_koopman, n_koopman)
            The matrix V, where each column represents a Koopman mode.
        """
        return self._unnormalized_modes

    def phi(self, x_col):
        return self._compute_phi(x_col)

    def psi(self, x_col):
        return self._compute_psi(x_col)

    def _compute_phi(self, x_col):
        """
        Computes the Koopman observable vector `phi(x)` for input `x`.

        Args:
            x (np.ndarray or torch.Tensor): The input state vector or tensor.

        Returns:
            phi (np.ndarray): The Koopman observable vector `phi(x)` for input `x`.
        """
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        x = x_col.T

        self._regressor.eval()
        x = self._convert_input_ndarray_to_tensor(x)

        if self.normalize:
            x = self.dm.normalization(x)
        phi = self._regressor._encoder(x).detach().numpy().T
        return phi

    def _compute_psi(self, x_col):
        """
        Computes the Koopman eigenfunction expansion coefficients `psi(x)` given `x`.

        Args:
            x (numpy.ndarray): Input data of shape `(n_samples, n_features)`.

        Returns:
            numpy.ndarray: Koopman eigenfunction expansion coefficients `psi(x)`
                of shape `(n_koopman, n_samples)`.
        """
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        # x = x_col.T

        phi = self._compute_phi(x_col)
        psi = np.linalg.inv(self._eigenvectors_) @ phi
        return psi

    def _convert_input_ndarray_to_tensor(self, x):
        """
        Converts input numpy ndarray to PyTorch tensor with appropriate dtype and
        device.

        Args:
            x (np.ndarray or torch.Tensor): Input data as numpy ndarray or PyTorch
                tensor.

        Returns:
            torch.Tensor: Input data as PyTorch tensor.

        Raises:
            TypeError: If input data is not a numpy ndarray or PyTorch tensor.
            ValueError: If input array has more than 2 dimensions.
        """
        if isinstance(x, np.ndarray):
            if x.ndim > 2:
                raise ValueError("input array should be 1 or 2D")
            if x.ndim == 1:
                x = x.reshape(1, -1)
            # convert to a float32
            # if x.dtype == np.float64:
            x = torch.FloatTensor(x)
        elif isinstance(x, torch.Tensor):
            if x.ndim != 2:
                raise ValueError("input tensor `x` must be a 2d tensor")
        return x

    @property
    def coef_(self):
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        return self._state_matrix_

    @property
    def eigenvalues_(self):
        check_is_fitted(self, "_eigenvalues_")
        return self._eigenvalues_

    @property
    def eigenvectors_(self):
        check_is_fitted(self, "_eigenvectors_")
        return self._eigenvectors_

    @property
    def unnormalized_modes(self):
        check_is_fitted(self, "_unnormalized_modes")
        return self._unnormalized_modes

    @property
    def ur(self):
        check_is_fitted(self, "_ur")
        return self._ur


if __name__ == "__main__":
    pass
