import torch
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem

from .._utils import (
    pull_info_from_hf_hub,
    pull_noise_info_from_hf_hub,
    unstandardize_y,
)
from ..functions.mlp import MLPFunction
from ..functions.nonparam import KRFunction
from .base import HeteroscedasticTestProblem, LLMTestProblem


class DMCurriculum(LLMTestProblem):
    r"""Data mixture selection with curriculum.

    First set of data mixture is used to train from 0 to 5M tokens.
    Second set of data mixture is used to train from 5M to 10M tokens.

    3 types of datasets are used:
        - instruction following (IF): allenai/tulu-3-sft-personas-instruction-following
        - math: allenai/tulu-3-sft-personas-math
        - code: allenai/tulu-3-sft-personas-code

    6 parameters:
        1. IF proportion 1 (float, [0, 1])
        2. Math proportion 1 (float, [0, 1])
        3. Code proportion 1 (float, [0, 1])
        4. IF proportion 2 (float, [0, 1])
        5. Math proportion 2 (float, [0, 1])
        6. Code proportion 2 (float, [0, 1])

    Parameters 1,2,3 must sum to 1. So must parameters 4,5,6. (Simplex)

    Single objective that averages:
        - evaluation results on IFEval (strict)
        - evaluation results on MATH-500 (minerva format)
        - evaluation results on MBPP plus

    Example usage:
    ```python
    import torch
    from bolt import DMCurriculum

    prob = DMCurriculum(noise_std=0.001)

    X = torch.Tensor([[0.15, 0.2, 0.65, 0.6, 0.2, 0.2]])
    y = prob(X)
    ```

    """

    name = "dm_curriculum"
    hf_repo = "chewwt/dm_qwen4b_emulator"
    hf_revision = "v0.2.0"

    dim = 6
    _bounds = [
        (0.0, 1.0),  # if_prop1
        (0.0, 1.0),  # math_prop1
        (0.0, 1.0),  # code_prop1
        (0.0, 1.0),  # if_prop2
        (0.0, 1.0),  # math_prop2
        (0.0, 1.0),  # code_prop2
    ]

    _check_grad_at_opt: bool = True
    continuous_inds = [0, 1, 2, 3, 4, 5]

    _optimal_value = 0.61395  # empirically found
    _optimizers = [(0.59699, 0.40301, 0.00000, 0.29968, 0.65149, 0.04883)]

    # Measured mean std of the 3-benchmark average, over 100 configs x 5 seeds.
    _measured_std = 0.0114

    def __init__(
        self,
        noise_std: None | float | list[float] = _measured_std,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Data mixture curriculum optimization for Qwen3-4B-Base

        Args:
            noise_std: Standard deviation of the observation noise. Defaults to
                the empirically measured ``_measured_std``; pass ``None`` for a
                noiseless problem. A list gives per-objective values.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """

        super().__init__(
            noise_std=noise_std,
            negate=negate,
            dtype=dtype,
        )

        self.model_path, self.model_config, self.y_mean, self.y_std = (
            pull_info_from_hf_hub(self.hf_repo, revision=self.hf_revision)
        )

        self.obj_func = MLPFunction(
            self.model_config["input_dim"],
            self.model_path,
            hidden_dim=self.model_config["hidden_dim"],
            output_dim=self.model_config["output_dim"],
            # absent from pre-v0.2.0 configs, which are all 2-layer
            n_layers=self.model_config.get("n_layers", 2),
        )

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the objective using the pretrained emulator.

        Args:
            X (torch.Tensor): Input tensor of shape `(N, dim)`.

        Returns:
            torch.Tensor: Objective tensor of shape `(N, 1)`.
        """

        validate_simplex_product(X=X, eps=1e-5)

        y_mo = self.obj_func.evaluate_true(X)
        y_mo_st = unstandardize_y(y_mo, self.y_mean, self.y_std)
        return y_mo_st.mean(axis=1)[..., None]


class DMCurriculumMO(MultiObjectiveTestProblem, LLMTestProblem):
    r"""Data mixture selection with curriculum and multi-objective.

    First set of data mixture is used to train from 0 to 5M tokens.
    Second set of data mixture is used to train from 5M to 10M tokens.

    3 types of datasets are used:
        - instruction following (IF): allenai/tulu-3-sft-personas-instruction-following
        - math: allenai/tulu-3-sft-personas-math
        - code: allenai/tulu-3-sft-personas-code

    6 parameters:
        1. IF proportion 1 (float, [0, 1])
        2. Math proportion 1 (float, [0, 1])
        3. Code proportion 1 (float, [0, 1])
        4. IF proportion 2 (float, [0, 1])
        5. Math proportion 2 (float, [0, 1])
        6. Code proportion 2 (float, [0, 1])

    Parameters 1,2,3 must sum to 1. So must parameters 4,5,6. (Simplex)

    Multiobjective with 3 outputs:
        1. evaluation results on IFEval (strict)
        2. evaluation results on MATH-500 (minerva format)
        3. evaluation results on MBPP plus

    Example usage:
    ```python
    import torch
    from bolt import DMCurriculumMO

    prob = DMCurriculumMO(noise_std=0.001)

    X = torch.Tensor([[0.15, 0.2, 0.65, 0.6, 0.2, 0.2]])
    y = prob(X)
    ```

    """

    name = "dm_curriculum_mo"
    hf_repo = "chewwt/dm_qwen4b_emulator"
    hf_revision = "v0.2.0"

    dim = 6
    _bounds = [
        (0.0, 1.0),  # if_prop1
        (0.0, 1.0),  # math_prop1
        (0.0, 1.0),  # code_prop1
        (0.0, 1.0),  # if_prop2
        (0.0, 1.0),  # math_prop2
        (0.0, 1.0),  # code_prop2
    ]

    _check_grad_at_opt: bool = True
    continuous_inds = [0, 1, 2, 3, 4, 5]

    num_objectives: int = 3
    _ref_point = [0.38854, 0.32387, 0.74238]
    _max_hv = 0.0026072

    # Measured mean std per objective (IFEval/MATH/MBPP+), over 100 configs x 5 seeds.
    _measured_std = [0.0131, 0.0274, 0.0101]

    def __init__(
        self,
        noise_std: None | float | list[float] = _measured_std,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Data mixture curriculum optimization for Qwen3-4B-Base

        Args:
            noise_std: Standard deviation of the observation noise. Defaults to
                the empirically measured ``_measured_std``; pass ``None`` for a
                noiseless problem. A list gives per-objective values.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """

        super().__init__(
            noise_std=noise_std,
            negate=negate,
            dtype=dtype,
        )

        self.model_path, self.model_config, self.y_mean, self.y_std = (
            pull_info_from_hf_hub(self.hf_repo, revision=self.hf_revision)
        )

        self.obj_func = MLPFunction(
            self.model_config["input_dim"],
            self.model_path,
            hidden_dim=self.model_config["hidden_dim"],
            output_dim=self.model_config["output_dim"],
            # absent from pre-v0.2.0 configs, which are all 2-layer
            n_layers=self.model_config.get("n_layers", 2),
        )

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the objective using the pretrained emulator.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: Objective tensor of shape ``(N, 3)``.
        """

        validate_simplex_product(X=X, eps=1e-5)

        y_mo = self.obj_func.evaluate_true(X)
        y_mo_st = unstandardize_y(y_mo, self.y_mean, self.y_std)
        return y_mo_st


class DMCurriculumHet(HeteroscedasticTestProblem, LLMTestProblem):
    r"""Data mixture selection with curriculum and heteroscedastic noise.

    First set of data mixture is used to train from 0 to 5M tokens.
    Second set of data mixture is used to train from 5M to 10M tokens.

    3 types of datasets are used:
        - instruction following (IF): allenai/tulu-3-sft-personas-instruction-following
        - math: allenai/tulu-3-sft-personas-math
        - code: allenai/tulu-3-sft-personas-code

    6 parameters:
        1. IF proportion 1 (float, [0, 1])
        2. Math proportion 1 (float, [0, 1])
        3. Code proportion 1 (float, [0, 1])
        4. IF proportion 2 (float, [0, 1])
        5. Math proportion 2 (float, [0, 1])
        6. Code proportion 2 (float, [0, 1])

    Parameters 1,2,3 must sum to 1. So must parameters 4,5,6. (Simplex)

    Single objective averaging three benchmarks:
        - evaluation results on IFEval (strict)
        - evaluation results on MATH-500 (minerva format)
        - evaluation results on MBPP plus

    Only MATH-500 has an input-dependent noise emulator; the IFEval and MBPP+
    stds are much flatter over the input space, so they are held at their
    measured constants. The objective is the mean of the three benchmarks and
    the replicate deviations are near-uncorrelated across them, so the noise on
    the average is

        sigma(x) = sqrt(sigma_if^2 + sigma_math(x)^2 + sigma_code^2) / 3

    Example usage:
    ```python
    import torch
    from bolt import DMCurriculumHet

    prob = DMCurriculumHet()

    X = torch.Tensor([[0.15, 0.2, 0.65, 0.6, 0.2, 0.2]])
    y = prob(X)
    ```

    """

    name = "dm_curriculum_heteroscedastic"
    hf_repo = "chewwt/dm_qwen4b_emulator"
    hf_revision = "v0.2.0"
    hf_repo_noise = "chewwt/dm_qwen4b_noise_emulator"
    hf_revision_noise = "v0.2.0"

    dim = 6
    _bounds = [
        (0.0, 1.0),  # if_prop1
        (0.0, 1.0),  # math_prop1
        (0.0, 1.0),  # code_prop1
        (0.0, 1.0),  # if_prop2
        (0.0, 1.0),  # math_prop2
        (0.0, 1.0),  # code_prop2
    ]

    _check_grad_at_opt: bool = True
    continuous_inds = [0, 1, 2, 3, 4, 5]

    _optimal_value = 0.61395  # empirically found
    _optimizers = [(0.59699, 0.40301, 0.00000, 0.29968, 0.65149, 0.04883)]

    # Measured mean std of IFEval and MBPP+, the two with no noise emulator, over
    # 100 configs x 5 seeds. Same measurement as DMCurriculumMO._measured_std[0]
    # and [2], unrounded.
    _sigma_if = 0.0131165
    _sigma_code = 0.0100820

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Data mixture curriculum optimization for Qwen3-4B-Base

        Args:
            noise_std: Standard deviation of the observation noise. Ignored
                in this problem as noise_std is output from noise model.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """

        super().__init__(
            noise_std=noise_std,
            negate=negate,
            dtype=dtype,
        )

        self.model_path, self.model_config, self.y_mean, self.y_std = (
            pull_info_from_hf_hub(self.hf_repo, revision=self.hf_revision)
        )
        self.model_path_noise, self.model_config_noise = pull_noise_info_from_hf_hub(
            self.hf_repo_noise, revision=self.hf_revision_noise
        )

        self.obj_func = MLPFunction(
            self.model_config["input_dim"],
            self.model_path,
            hidden_dim=self.model_config["hidden_dim"],
            output_dim=self.model_config["output_dim"],
            # absent from pre-v0.2.0 configs, which are all 2-layer
            n_layers=self.model_config.get("n_layers", 2),
        )

        # noise func only for math std
        self.noise_func = KRFunction(
            self.model_path_noise,
            gamma=self.model_config_noise.get("gamma", 0.1),
            scale_factor=1.0,
        )

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the objective using the pretrained emulator.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: Objective tensor of shape ``(N, 1)``.
        """

        validate_simplex_product(X=X, eps=1e-5)

        y_mo = self.obj_func.evaluate_true(X)
        y_mo_st = unstandardize_y(y_mo, self.y_mean, self.y_std)
        return y_mo_st.mean(axis=1)[..., None]

    def _evaluate_noise(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the noise function using the pretrained noise emulator.

        The objective averages 3 benchmarks whose replicate deviations are
        near-uncorrelated, so the variance of the average is the sum of the three
        variances over 9. Only the MATH-500 std is input-dependent.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: Noise std tensor of shape ``(N, 1)``.

        """

        validate_simplex_product(X=X, eps=1e-5)

        # The noise emulator takes the full curriculum vector and predicts the
        # MATH-500 std only.
        sigma_math = self.noise_func.evaluate_true(X).clamp(min=1e-6)

        sigma_avg = torch.sqrt(
            (self._sigma_if**2 + sigma_math**2 + self._sigma_code**2) / 9
        )

        return sigma_avg[:, None]


def validate_simplex_product(X: torch.Tensor, eps: float = 1e-5) -> None:
    """Validate that the first three and last three parameters each sum to 1.

    Data-mixture inputs live on a product of two simplices: parameters 0–2
    define stage-1 proportions and parameters 3–5 define stage-2 proportions.

    Args:
        X: Input tensor of shape ``(N, 6)``.
        eps: Tolerance for the simplex constraint check.

    Raises:
        ValueError: If any row violates the simplex constraint in either group.
    """

    # check first 3 and last 3 params are simplex
    if ((X[:, :3].sum(dim=1) - 1).abs() > eps).any() or (
        (X[:, 3:].sum(dim=1) - 1).abs() > eps
    ).any():
        raise ValueError(
            "first 3 and last 3 parameters need be from a simplex (sum to 1)"
        )
