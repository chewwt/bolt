import torch
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from huggingface_hub import hf_hub_download

from .._utils import pull_info_from_hf_hub, unstandardize_y
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

    _optimal_value = 0.61116  # empirically found
    _optimizers = [(0.5663, 0.4337, 0.0000, 0.4139, 0.4901, 0.0960)]

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Data mixture curriculum optimization for Qwen3-4B-Base

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
            hidden_dim=self.model_config["hidden_dim"],
            output_dim=self.model_config["output_dim"],
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
    _ref_point = [0.40563, 0.34200, 0.75031]
    _max_hv = 0.0018603

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Data mixture curriculum optimization for Qwen3-4B-Base

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
            hidden_dim=self.model_config["hidden_dim"],
            output_dim=self.model_config["output_dim"],
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

    Single objective: evaluation results on MATH-500 (minerva format)

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
    hf_repo_noise = "chewwt/dm_qwen4b_noise_emulator"

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

    _optimal_value = 0.52161
    _optimizers = [(0.5311, 0.4689, 0.0000, 0.2176, 0.7824, 0.0000)]

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
            pull_info_from_hf_hub(self.hf_repo)
        )
        self.model_path_noise = hf_hub_download(
            self.hf_repo_noise, "noise_model.safetensors"
        )

        self.obj_func = MLPFunction(
            self.model_config["input_dim"],
            self.model_path,
            hidden_dim=self.model_config["hidden_dim"],
            output_dim=self.model_config["output_dim"],
        )

        # noise func only for math std
        self.noise_func = KRFunction(self.model_path_noise, scale_factor=0.1)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the objective using the pretrained emulator.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: Objective tensor of shape ``(N, 3)``.
        """

        validate_simplex_product(X=X, eps=1e-5)

        y_mo = self.obj_func.evaluate_true(X)[:, 1]
        y_mo_st = unstandardize_y(y_mo, self.y_mean[1], self.y_std[1])
        return y_mo_st[:, None]

    def _evaluate_noise(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the noise function using the pretrained noise emulator.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: Noise std tensor of shape ``(N, 3)``.

        """

        validate_simplex_product(X=X, eps=1e-5)

        # X for noise model takes in [if_prop1, math_prop1, math_prop2] only
        # predicts math std only
        y_noise = self.noise_func.evaluate_true(X[:, [0, 1, 4]]).clamp(min=1e-6)

        return y_noise[:, None]


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
