import torch
import torch.nn.functional as F

from .._utils import pull_info_from_hf_hub, unstandardize_y
from ..functions.mlp import MLPFunction
from .base import LLMTestProblem


class HPO(LLMTestProblem):
    r"""Hyperparameter problem.

    7 parameters:
        1. learning rate (float, [0, 1])
        2. batch size (int, [2, 4])
        3. lora rank (int, [2, 5])
        4. lora alpha (int, [2, 5])
        5. lora dropout (float, [0, 1])
        6. lora layers (int, [1, 30])
        7. lora target module (categorical int, [0, 3])

    Example usage:
    ```python
    import torch
    from bolt import HPO

    prob = HPO(noise_std=0.001, negate=False)
    X = torch.Tensor([[0, 2, 2, 2, 0.5, 30, 2]])
    y = prob(X)

    ```

    """

    name = "hpo"
    hf_repo = "chewwt/hpo_qwen8b_emulator"

    dim = 7
    _bounds = [
        (0.0, 1.0),  # lr
        (2, 4),  # batch
        (2, 5),  # lora rank
        (2, 5),  # lora alpha
        (0.0, 1.0),  # lora dropout
        (1, 30),  # lora layers
        (0, 3),  # lora target module (categorical)
    ]
    _check_grad_at_opt: bool = True
    continuous_inds = [0, 4]
    discrete_inds = [1, 2, 3, 5]
    categorical_inds = [6]

    _optimal_value = 0.34647  # empirically found
    _optimizers = [(0.31100, 2, 4, 2, 0.87056, 30, 1)]

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Hyperparameter optimization for Qwen3-8B-Base LoRA fine-tuning

        Args:
            noise_std: Standard deviation of the observation noise. If a list is
                provided, specifies separate noise standard deviations for each
                objective in a multiobjective problem.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """

        super().__init__(
            noise_std=noise_std,
            negate=negate,
            dtype=dtype,
        )

        self.model_path, self.model_config, self.y_mean, self.y_std = (
            pull_info_from_hf_hub(self.hf_repo)
        )

        self.obj_func = MLPFunction(
            self.model_config["input_dim"],
            self.model_path,
            categorical_inds=[],
            categorical_sizes=[],
            hidden_dim=self.model_config["hidden_dim"],
            output_dim=self.model_config["output_dim"],
        )

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the objective using the pretrained emulator.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: Objective tensor of shape ``(N, 1)``.
        """

        X_copy = X.clone()

        # normalize integers to (0, 1) bound
        for i in self.discrete_inds:
            i_min, i_max = self._bounds[i]
            X_copy[:, i] = (X_copy[:, i] - i_min) / (i_max - i_min)

        # expand categorical lora_target to one-hot (exact 0/1, matching training data)
        one_hot = F.one_hot(X_copy[:, 6].long(), num_classes=4).to(dtype=X_copy.dtype)
        X_enc = torch.cat(
            [
                X_copy[:, :6],
                one_hot,
                torch.ones(len(X_copy), 1, dtype=X_copy.dtype, device=X_copy.device),
            ],
            dim=1,
        )  # 11-dim

        # unstandardize output to get the real values
        y_st = self.obj_func.evaluate_true(X_enc)
        y = unstandardize_y(y_st, self.y_mean, self.y_std)
        return y


class HPOMultiFidelityToken(LLMTestProblem):
    """Multifidelity hyperparameter problem. Same model with varying number of training tokens

    8 parameters:
        1. learning rate (float, [0, 1])
        2. batch size (int, [2, 4])
        3. lora rank (int, [2, 5])
        4. lora alpha (int, [2, 5])
        5. lora dropout (float, [0, 1])
        6. lora layers (int, [1, 30])
        7. lora target module (categorical int, [0, 3])
        8. [fidelity parameter] Number of training tokens seen (float, [0, 1],
           normalized over 1e5–9.1e6 tokens)

    Evaluation samples the emulator at ``fidelity_step`` intervals (≈500k tokens)
    up to the queried fidelity and returns the cumulative max, ensuring monotonicity
    for MF-BO methods.

    Example usage:
    ```python
    import torch
    from bolt import HPOMultiFidelityToken

    prob = HPOMultiFidelityToken(noise_std=0.001, negate=False)
    X = torch.Tensor([[0, 2, 2, 2, 0.5, 30, 2, 0.5]])
    y = prob(X)
    ```

    """

    name = "hpo_multifidelity_step"
    hf_repo = "chewwt/hpo_qwen8b_emulator"

    dim = 8
    _bounds = [
        (0.0, 1.0),  # lr
        (2, 4),  # batch
        (2, 5),  # lora rank
        (2, 5),  # lora alpha
        (0.0, 1.0),  # lora dropout
        (1, 30),  # lora layers
        (0, 3),  # lora target module (categorical)
        (0.0, 1.0),  # fidelity (num training tokens seen, (normalized 1e5 to 9.1e6))
    ]
    _check_grad_at_opt: bool = True
    continuous_inds = [0, 4, 7]
    discrete_inds = [1, 2, 3, 5]
    categorical_inds = [6]
    fidelity_step: float = (
        0.056  # normalized interval (~500k tokens); cummax ensures monotonicity
    )

    _optimal_value = 0.35480  # empirically found
    _optimizers = [(0.23094, 4, 3, 3, 0.33724, 30, 2, 1.0)]

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Hyperparameter optimization for Qwen3-8B-Base LoRA fine-tuning, with
        fidelity controlled by number of training tokens

        Args:
            noise_std: Standard deviation of the observation noise. If a list is
                provided, specifies separate noise standard deviations for each
                objective in a multiobjective problem.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """

        super().__init__(
            noise_std=noise_std,
            negate=negate,
            dtype=dtype,
        )

        self.model_path, self.model_config, self.y_mean, self.y_std = (
            pull_info_from_hf_hub(self.hf_repo)
        )

        self.obj_func = MLPFunction(
            self.model_config["input_dim"],
            self.model_path,
            categorical_inds=[],
            categorical_sizes=[],
            hidden_dim=self.model_config["hidden_dim"],
            output_dim=self.model_config["output_dim"],
        )

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the objective using the pretrained emulator.

        For each query, the emulator is evaluated at fidelity intervals of
        ``fidelity_step`` from ``fidelity_step`` up to (and including) the queried
        fidelity. The cumulative max across those samples is returned,
        guaranteeing that higher-fidelity queries produce values >= lower-fidelity
        queries with the same hyperparameters.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: Objective tensor of shape ``(N, 1)``.
        """

        X_copy = X.clone()

        # normalize integers to (0, 1) bound
        for i in self.discrete_inds:
            i_min, i_max = self._bounds[i]
            X_copy[:, i] = (X_copy[:, i] - i_min) / (i_max - i_min)

        # Build expanded batch: for each query, sample at fidelity_step intervals up to
        # the queried fidelity, then take cumulative max over those samples.
        n = X_copy.shape[0]
        x_list, counts = [], []
        for i in range(n):
            fidelity = X_copy[i, -1].item()

            if fidelity > self.fidelity_step:
                steps = torch.arange(
                    self.fidelity_step,
                    fidelity,
                    self.fidelity_step,
                    dtype=X_copy.dtype,
                    device=X_copy.device,
                )
                steps = torch.cat([steps, X_copy[i, -1:]])
            else:
                # skip if query fidelity is smaller than self.fidelity_step
                steps = X_copy[i, -1:]  # queried fidelity only

            x_rep = X_copy[i].unsqueeze(0).expand(steps.numel(), -1).clone()
            x_rep[:, -1] = steps
            x_list.append(x_rep)
            counts.append(steps.numel())

        X_exp = torch.cat(x_list, dim=0)  # (sum(counts), 8)

        # expand categorical lora_target to one-hot (exact 0/1, matching training data)
        one_hot = F.one_hot(X_exp[:, 6].long(), num_classes=4).to(dtype=X_exp.dtype)
        X_enc = torch.cat(
            [X_exp[:, :6], one_hot, X_exp[:, 7:]], dim=1
        )  # 11-dim, fidelity last

        y_st = self.obj_func.evaluate_true(X_enc)
        y = unstandardize_y(y_st, self.y_mean, self.y_std)

        # Cumulative max per original query
        out = torch.zeros(n, 1, dtype=y.dtype, device=y.device)
        idx = 0
        for i, c in enumerate(counts):
            out[i] = y[idx : idx + c].max(dim=0).values
            idx += c
        return out

    def cost(self, X: torch.Tensor) -> torch.Tensor:
        r"""Return the evaluation cost at the fidelity encoded in `X`.

        Cost is proportional to the fidelity level. The highest fidelity
        has a cost of 1 and the lowest fidelity a cost of 0.1.

        Args:
            X (torch.Tensor): Input tensor of shape `(N, dim)`, where the last dimension
                encodes the fidelity parameter.

        Returns:
            torch.Tensor: Cost tensor of shape `(N, )`.
        """

        fidelity = X[..., -1]
        return fidelity * 0.9 + 0.1


class HPOMultiFidelityModel(LLMTestProblem):
    r"""Multifidelity hyperparameter problem. Different model sizes

    8 parameters:
        1. learning rate (float, [0, 1])
        2. batch size (int, [2, 4])
        3. lora rank (int, [2, 5])
        4. lora alpha (int, [2, 5])
        5. lora dropout (float, [0, 1])
        6. lora layers ([int, 1, 30])
        7. lora target module (categorical int, [0, 3])
        8. [fidelity parameter] model size (0 for 4B, 1 for 8B) (int, [0, 1])

    Example usage:
    ```python
    import torch
    from bolt import HPOMultiFidelityModel

    prob = HPOMultiFidelityModel(noise_std=0.001, negate=False)
    X = torch.Tensor([[0, 2, 2, 2, 0.5, 30, 2, 1]])
    y = prob(X)
    ```

    """

    name = "hpo_multifidelity_model"
    hf_repo_high_fid = "chewwt/hpo_qwen8b_emulator"  # high fid
    hf_repo_low_fid = "chewwt/hpo_qwen4b_emulator"

    dim = 8
    _bounds = [
        (0.0, 1.0),  # lr
        (2, 4),  # batch
        (2, 5),  # lora rank
        (2, 5),  # lora alpha
        (0.0, 1.0),  # lora dropout
        (1, 30),  # lora layers
        (0, 3),  # lora target module (categorical)
        (0, 1),  # [fidelity] model size (0 for 4B, 1 for 8B)
    ]
    _check_grad_at_opt: bool = True
    continuous_inds = [0, 4]
    discrete_inds = [1, 2, 3, 5, 7]
    categorical_inds = [6]

    _optimal_value = 0.34647  # empirically found
    _optimizers = [(0.31100, 2, 4, 2, 0.87056, 30, 1, 1)]

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Hyperparameter optimization for Qwen3-8B-Base LoRA fine-tuning, with fidelity controlled by
        model size (Qwen3-4B-Base, Qwen3-8B-Base)

        Args:
            noise_std: Standard deviation of the observation noise. If a list is
                provided, specifies separate noise standard deviations for each
                objective in a multiobjective problem.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """

        super().__init__(
            noise_std=noise_std,
            negate=negate,
            dtype=dtype,
        )

        (
            self.model_path_high_fid,
            self.model_config_high_fid,
            self.y_mean_high_fid,
            self.y_std_high_fid,
        ) = pull_info_from_hf_hub(self.hf_repo_high_fid)
        (
            self.model_path_low_fid,
            self.model_config_low_fid,
            self.y_mean_low_fid,
            self.y_std_low_fid,
        ) = pull_info_from_hf_hub(self.hf_repo_low_fid)

        self.high_fidelity_obj_function = MLPFunction(
            self.model_config_high_fid["input_dim"],
            self.model_path_high_fid,
            categorical_inds=[],
            categorical_sizes=[],
            hidden_dim=self.model_config_high_fid["hidden_dim"],
            output_dim=self.model_config_high_fid["output_dim"],
        )
        self.low_fidelity_obj_function = MLPFunction(
            self.model_config_low_fid["input_dim"],
            self.model_path_low_fid,
            categorical_inds=[],
            categorical_sizes=[],
            hidden_dim=self.model_config_low_fid["hidden_dim"],
            output_dim=self.model_config_low_fid["output_dim"],
        )

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the objective using the pretrained emulator corresponding
        to the selected fidelity.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: Objective tensor of shape ``(N, 1)``.
        """

        # check last parameter match fidelity values
        if not torch.bitwise_or(
            ((1 - X[:, -1]).abs() < 0.05), ((X[:, -1] - 0).abs() < 0.05)
        ).all():
            raise ValueError(
                "Set fidelity values according to _bounds. Should be 0 or 1"
            )

        X_copy = X.clone().detach()

        # normalize integers to (0, 1) bound
        for i in self.discrete_inds:
            i_min, i_max = self._bounds[i]
            X_copy[:, i] = (X_copy[:, i].float() - i_min) / (i_max - i_min)

        X_splice_idxs_all = []
        outs = []

        for i in range(int(self._bounds[-1][0]), int(self._bounds[-1][1]) + 1):
            mask = X_copy[:, -1] == i

            # splice to feed into correct model
            X_splice = X_copy[mask, :]

            X_splice_idxs = torch.arange(len(X_copy), device=X_copy.device)[
                mask
            ].tolist()

            # expand categorical lora_target to one-hot (exact 0/1, matching training data)
            one_hot = F.one_hot(X_splice[:, 6].long(), num_classes=4).to(
                dtype=X_splice.dtype
            )
            X_enc = torch.cat(
                [
                    X_splice[:, :6],
                    one_hot,
                    torch.ones(
                        len(X_splice), 1, dtype=X_splice.dtype, device=X_splice.device
                    ),  # at max training tokens
                ],
                dim=1,
            )  # 11-dim

            # evaluate each fidelity, unstandardize output to get the real values
            if i == 1:
                y_st = self.high_fidelity_obj_function.evaluate_true(X_enc)
                outs.append(
                    unstandardize_y(y_st, self.y_mean_high_fid, self.y_std_high_fid)
                )
            elif i == 0:
                y_st = self.low_fidelity_obj_function.evaluate_true(X_enc)
                outs.append(
                    unstandardize_y(y_st, self.y_mean_low_fid, self.y_std_low_fid)
                )
            else:
                raise ValueError(f"Unknown fidelity {i}")

            X_splice_idxs_all += X_splice_idxs

        # reorder outputs to match original input
        outs_t = torch.vstack(outs)
        outs_order = torch.empty_like(outs_t)
        outs_order[X_splice_idxs_all, :] = outs_t

        return outs_order

    def cost(self, X: torch.Tensor) -> torch.Tensor:
        r"""Return the evaluation cost at the fidelity encoded in ``X``.

        Cost is proportional to the fidelity level. The highest fidelity
        has a cost of 1 and the lowest fidelity a cost of 0.1.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``, where the last
                dimension encodes the fidelity parameter.

        Returns:
            torch.Tensor: Cost tensor of shape ``(N, )``.
        """

        fidelity = X[..., -1]
        return fidelity * 0.9 + 0.1
