"""Composed FKC problems: analytic experts, drift, potential, exact targets.

For constant exponents ``beta_i > 0`` with ``B = sum_i beta_i``, the composed
target is ``pi_t ∝ prod_i (q_t^i)^{beta_i}``. For an affine schedule
``(alpha_t, sigma_t)`` with SDE diffusion ``nu_t`` the fields are

    ``a_t(x) = (alpha_dot_t / alpha_t) x``,
    ``nu_t**2 = 2 (alpha_dot_t / alpha_t) sigma_t**2 - 2 sigma_t sigma_dot_t``,
    ``S_beta = sum_i beta_i s_t^i``,  ``s_t^i = grad log q_t^i``,
    ``u_beta = a + (nu_t**2 / 2) S_beta``,   (composed velocity)
    ``b_beta = 2 u_beta - a = a + nu_t**2 S_beta``,   (SDE drift)
    ``g_beta = (nu_t**2 / 2) (||S_beta||**2 - sum_i beta_i ||s_t^i||**2)``.

``g_beta`` drops only spatially constant terms, which do not affect
self-normalized weights. The single-expert exponents admit exact finite
mixtures at every time (including cross terms), so ``target(t)`` returns a
:class:`GaussianMixture2D` whose time-1 marginal is ``pi_t`` exactly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.models.components.schedule import AffineSchedule, LinearSchedule

from .gmm import GaussianMixture2D, _component_log_prob, multiply_mixtures
from .weights import systematic_resample

#: Initial composition case with an exactly normalizeable finite-mixture target.
FKC_CASES = ('gaussian40',)

#: Reference 40-Gaussian target parameters (FKC temperature-annealing notebook).
_REFERENCE_LOC_SCALING = 40.0
_REFERENCE_LOG_VAR_SCALING = 2.0
_REFERENCE_NORMALIZATION = 50.0


def _softplus(x: float) -> float:
    """Numerically stable ``log(1 + exp(x))`` for a Python float."""
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


class FKCProblem(nn.Module):
    """A constant-exponent composition of analytic 2D Gaussian-mixture experts."""

    def __init__(
        self,
        experts: Sequence[GaussianMixture2D],
        powers: Sequence[int] = (1,),
        schedule: AffineSchedule | None = None,
    ) -> None:
        """Initialize the problem.

        :param experts: Expert mixtures ``q_t^i`` (all in one dtype/device).
        :param powers: Positive integer exponents ``beta_i``. The exact target
            is a finite mixture with ``prod_i K_i ** beta_i`` components, so the
            cost grows as ``K ** sum(beta_i)``; we cap ``sum(beta_i)`` at 3.
        :param schedule: Affine noise schedule; defaults to the linear schedule.
        """
        super().__init__()
        if len(experts) < 1:
            raise ValueError('a problem needs at least one expert.')
        if len(powers) != len(experts):
            raise ValueError(
                f'expected one power per expert; got {len(powers)} powers for '
                f'{len(experts)} experts.'
            )
        for power in powers:
            if not isinstance(power, int) or power < 1:
                raise ValueError(f'powers must be positive integers; got {power!r}.')
        if sum(powers) > 3:
            raise ValueError(
                f'sum of powers must be at most 3 for exact targets; got {sum(powers)}.'
            )
        self.experts = nn.ModuleList(experts)
        self.powers = tuple(int(p) for p in powers)
        self.schedule: AffineSchedule = schedule or LinearSchedule()
        # The composed target's marginalization must use the same path as the
        # fields, so every expert inherits the problem's schedule.
        for expert in self.experts:
            expert.schedule = self.schedule
        self._target_cache: dict[float, GaussianMixture2D] = {}

    def scores(self, x: Tensor, t: Tensor) -> list[Tensor]:
        """Return each expert marginal score ``s_t^i(x)`` of shape ``(B, 2)``."""
        return [expert.score(x, t) for expert in self.experts]

    def fields(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return the composed velocity ``u_beta`` and potential ``g_beta``.

        :param x: States ``(B, 2)``.
        :param t: Physical times, scalar or ``(B,)`` in ``(0, 1]``.
        :return: ``(velocity (B, 2), g (B,))``.
        """
        t_b = _time_like(t, x)
        a = (self.schedule.alpha_dot(t_b) / self.schedule.alpha(t_b))[:, None] * x
        nu2 = self.schedule.nu_squared(t_b)
        score_list = self.scores(x, t_b)
        combined = sum(power * score for power, score in zip(self.powers, score_list, strict=True))
        velocity = a + 0.5 * nu2[:, None] * combined
        sq_combined = (combined**2).sum(-1)
        sq_experts = sum(
            power * (score**2).sum(-1)
            for power, score in zip(self.powers, score_list, strict=True)
        )
        g = 0.5 * nu2 * (sq_combined - sq_experts)
        return velocity, g

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Return the composed SDE drift ``b_beta = 2 u_beta - a`` of shape ``(B, 2)``."""
        t_b = _time_like(t, x)
        a = (self.schedule.alpha_dot(t_b) / self.schedule.alpha(t_b))[:, None] * x
        nu2 = self.schedule.nu_squared(t_b)
        score_list = self.scores(x, t_b)
        combined = sum(power * score for power, score in zip(self.powers, score_list, strict=True))
        return a + nu2[:, None] * combined

    def potential(self, x: Tensor, t: Tensor) -> Tensor:
        """Return the composed weight rate ``g_beta`` of shape ``(B,)``."""
        return self.fields(x, t)[1]

    def _single_expert_target(self, index: int, t: float) -> GaussianMixture2D:
        """Build the exact time-``t`` law of one expert raised to its power."""
        means, covariances, weights = self.experts[index].marginal_parameters(t)
        base = GaussianMixture2D(means, covariances, weights, schedule=self.schedule)
        result = base
        for _ in range(self.powers[index] - 1):
            result = multiply_mixtures(result, base)
        return result

    def target(self, t: float) -> GaussianMixture2D:  # noqa: D102
        """Return a mixture whose time-1 law is exactly ``pi_t``.

        :param t: Scalar physical time in ``(0, 1]``.
        :return: A :class:`GaussianMixture2D` at endpoints equal to ``pi_t``.
        """
        t_value = float(t)
        if not 0.0 <= t_value <= 1.0:
            raise ValueError(f't must lie in [0, 1]; got {t}.')
        cached = self._target_cache.get(t_value)
        if cached is not None:
            return cached
        result = self._single_expert_target(0, t_value)
        for index in range(1, len(self.experts)):
            result = multiply_mixtures(result, self._single_expert_target(index, t_value))
        if len(self._target_cache) >= 8:
            self._target_cache.clear()
        self._target_cache[t_value] = result
        return result

    def sample_target(
        self, n: int, t: float, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Draw ``n`` exact samples from ``pi_t`` of shape ``(n, 2)``."""
        return self.target(t).sample(n, 1.0, generator=generator)

    def sample_target_batch(
        self, s: Tensor, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Draw one exact sample from ``pi_s`` per entry of per-sample times ``s``.

        Vectorized counterpart of :meth:`sample_target` so the training loss can
        use per-sample start times (matching the repo's WZ time sampling).

        :param s: Per-sample times ``(n,)`` in ``(0, 1]``.
        :param generator: Optional RNG.
        :return: Exact samples ``(n, 2)`` from ``pi_s``.
        """
        s = s.reshape(-1)
        n = s.shape[0]
        if bool(torch.any(s < 0.0)) or bool(torch.any(s > 1.0)):
            raise ValueError('times must satisfy 0 <= s <= 1.')
        eye = torch.eye(2, device=s.device, dtype=s.dtype)
        alpha_s = self.schedule.alpha(s)[:, None, None]
        sigma_sq = (self.schedule.sigma(s) ** 2)[:, None, None, None]
        components: list[tuple[Tensor, Tensor, Tensor]] = []
        for index, expert in enumerate(self.experts):
            means = alpha_s * expert.means[None]
            covariances = (
                alpha_s.unsqueeze(-1) ** 2 * expert.covariances[None] + sigma_sq * eye
            )
            log_weights = torch.log(expert.weights)[None].expand(n, -1)
            base = (means, covariances, log_weights)
            result = base
            for _ in range(self.powers[index] - 1):
                result = _product_batch(result, base)
            components.append(result)
        result = components[0]
        for extra in components[1:]:
            result = _product_batch(result, extra)
        means, covariances, log_weights = result
        probabilities = torch.softmax(log_weights, dim=-1)
        index = torch.multinomial(probabilities, 1, generator=generator)
        chosen_mean = torch.gather(means, 1, index[:, :, None].expand(-1, -1, 2)).squeeze(1)
        chosen_cov = torch.gather(
            covariances, 1, index[:, :, None, None].expand(-1, -1, 2, 2)
        ).squeeze(1)
        noise = torch.randn(n, 2, device=s.device, dtype=s.dtype, generator=generator)
        lower = torch.linalg.cholesky(chosen_cov)
        return chosen_mean + (lower @ noise.unsqueeze(-1)).squeeze(-1)

    def sample_expert_batch(
        self, s: Tensor, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Draw states from the first expert's marginal ``q_s`` (cheap proposal).

        The Eulerian identity is pointwise, so training states may come from any
        distribution covering the target's support. ``q_s`` covers ``pi_s``
        (since ``pi ∝ q**beta`` is more concentrated) and costs ``K`` components
        instead of ``prod_i K_i ** beta_i``, which matters for the 40-mode
        ``beta = 3`` case (40 vs 64,000 components).

        :param s: Per-sample times ``(n,)`` in ``[0, 1]``.
        :param generator: Optional RNG.
        :return: Samples ``(n, 2)`` from ``q_s``.
        """
        s = s.reshape(-1)
        return self.experts[0].sample(s.shape[0], s, generator=generator)

    def sample_target_cheap(
        self, s: Tensor, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Approximate ``pi_s`` samples by importance-resampling the expert.

        Since ``pi_s ∝ prod_i (q_s^i)^{beta_i}``, the ratio to the first expert
        is ``prod_i (q_s^i)^{beta_i} - 1`` on the first expert, i.e. a product of
        the experts' own log densities -- cheap (``K`` components) and available
        per sample. Draw from ``q_s^1`` and systematically resample, which keeps
        per-sample start times while avoiding the ``prod_i K_i ** beta_i``
        (64,000) exact-target build.

        :param s: Per-sample times ``(n,)`` in ``[0, 1]``.
        :param generator: Optional RNG.
        :return: Samples ``(n, 2)`` approximately from ``pi_s``.
        """
        s = s.reshape(-1)
        n = s.shape[0]
        x = self.experts[0].sample(n, s, generator=generator)
        log_ratio = (self.powers[0] - 1) * self.experts[0].log_prob(x, s)
        for index in range(1, len(self.experts)):
            log_ratio = log_ratio + self.powers[index] * self.experts[index].log_prob(x, s)
        indices = systematic_resample(log_ratio, generator=generator)
        return x[indices]

    def build_state_pool(
        self,
        num_s: int = 64,
        per_s: int = 2048,
        s_range: tuple[float, float] = (0.02, 0.9),
        generator: torch.Generator | None = None,
    ) -> None:
        """Precompute exact ``pi_s`` rollouts on a discrete ``s`` grid.

        Builds ``num_s * per_s`` (state, time) pairs from the composed target
        once; the loss then samples from this pool instead of rebuilding the
        (64k-component) target every training step.

        :param num_s: Number of start times on the grid.
        :param per_s: States drawn per grid time.
        :param s_range: Grid endpoints ``(s_min, s_max)``.
        :param generator: Optional RNG.
        """
        dtype, device = self.experts[0].means.dtype, self.experts[0].means.device
        s_vals = torch.linspace(s_range[0], s_range[1], num_s, device=device, dtype=dtype)
        chunks = [self.target(float(s)).sample(per_s, 1.0, generator=generator) for s in s_vals]
        self._state_pool = (torch.cat(chunks, 0), s_vals.repeat_interleave(per_s))

    def sample_pool(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        late_fraction: float = 0.0,
        late_start: float = 0.75,
    ) -> tuple[Tensor, Tensor]:
        """Draw ``(x, s)`` pairs from the cached exact-target pool.

        With ``late_fraction > 0`` that fraction of the batch is drawn uniformly
        from pool entries with ``s >= late_start`` and the rest uniformly from the
        whole pool, so the late interval is oversampled (not loss-weighted: more
        gradient samples there without inflating per-sample variance).
        """
        pool = getattr(self, '_state_pool', None)
        if pool is None:
            self.build_state_pool(generator=generator)
            pool = self._state_pool
        x, s = pool
        if not 0.0 <= late_fraction <= 1.0:
            raise ValueError(f'late_fraction must lie in [0, 1]; got {late_fraction}.')
        n_late = int(round(late_fraction * n))
        index = torch.randint(x.shape[0], (n - n_late,), device=x.device, generator=generator)
        if n_late > 0:
            late = torch.nonzero(s >= late_start, as_tuple=False).squeeze(-1)
            if late.numel() == 0:
                raise ValueError(f'no pool states with s >= {late_start}.')
            pick = torch.randint(late.numel(), (n_late,), device=x.device, generator=generator)
            index = torch.cat([index, late[pick]])
        return x[index], s[index]

    def log_target(self, x: Tensor, t: float) -> Tensor:
        """Return ``log pi_t(x)`` (up to the normalization constant) of shape ``(B,)``."""
        return self.target(t).log_prob(x, 1.0)


def _product_batch(
    a: tuple[Tensor, Tensor, Tensor], b: tuple[Tensor, Tensor, Tensor]
) -> tuple[Tensor, Tensor, Tensor]:
    """Exact per-sample product of two batched Gaussian mixtures.

    Each tuple is ``(means (n, K, 2), covariances (n, K, 2, 2), log_weights
    (n, K))``. Returns the renormalized product with ``K_a * K_b`` components
    per sample, using the Gaussian product identity including the overlap term.
    """
    means_a, covs_a, logw_a = a
    means_b, covs_b, logw_b = b
    ma, ca = means_a[:, :, None], covs_a[:, :, None]
    mb, cb = means_b[:, None, :], covs_b[:, None, :]
    log_overlap, _ = _component_log_prob(ca + cb, ma - mb)
    precision = torch.linalg.inv(ca) + torch.linalg.inv(cb)
    cov = torch.linalg.inv(precision)
    natural = torch.linalg.solve(ca, ma.unsqueeze(-1)).squeeze(-1) + torch.linalg.solve(
        cb, mb.unsqueeze(-1)
    ).squeeze(-1)
    shift = (cov @ natural.unsqueeze(-1)).squeeze(-1)
    log_weight = logw_a[:, :, None] + logw_b[:, None, :] + log_overlap
    n = means_a.shape[0]
    return shift.reshape(n, -1, 2), cov.reshape(n, -1, 2, 2), log_weight.reshape(n, -1)


def _time_like(t: Tensor | float, x: Tensor) -> Tensor:
    """Coerce a scalar or ``(B,)`` time to ``(B,)`` matching ``x``'s dtype/device."""
    t_b = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    if t_b.ndim == 0:
        t_b = t_b.expand(x.shape[0])
    elif t_b.ndim != 1 or t_b.shape[0] != x.shape[0]:
        raise ValueError(f't must be scalar or of shape ({x.shape[0]},); got {tuple(t_b.shape)}.')
    if bool(torch.any(t_b < 0.0)) or bool(torch.any(t_b > 1.0)):
        raise ValueError('times must satisfy 0 <= t <= 1.')
    return t_b


def forty_gaussian_experts(
    num_modes: int = 40,
    std: float | None = None,
    box: float | None = None,
    seed: int = 0,
    schedule: AffineSchedule | None = None,
) -> list[GaussianMixture2D]:
    """Return the FKC temperature-annealing 40-Gaussian expert.

    Defaults reproduce the reference notebook exactly
    (``applications/temperature_annealing/runner/notebooks/
    gmm_temp_annealed_birth_death.ipynb``)::

        GMM(dim=2, n_mixes=40, loc_scaling=40.0, log_var_scaling=2.0,
            should_unnormalize=True, data_normalization_factor=50)

    constructed after ``torch.manual_seed(0)``. The network/score operates in
    the ``/50``-normalized coordinates, where the modes are uniform on
    ``[-40, 40] / 50 = [-0.8, 0.8]`` and each per-mode standard deviation is
    ``softplus(2.0) / 50 ~= 0.04254`` (FAB's ``scale_tril =
    diag_embed(softplus(log_var))``).

    :param num_modes: Number of modes ``K``.
    :param std: Common per-mode standard deviation; defaults to
        ``softplus(log_var_scaling) / normalization``.
    :param box: Modes are uniform on ``[-box, box]^2``; defaults to
        ``loc_scaling / normalization``.
    :param seed: Seed for the mode placement (reference uses ``0``).
    :param schedule: Affine schedule used to marginalize the modes in time.
    :return: A one-element expert list.
    """
    if box is None:
        box = _REFERENCE_LOC_SCALING / _REFERENCE_NORMALIZATION
    if std is None:
        std = _softplus(_REFERENCE_LOG_VAR_SCALING) / _REFERENCE_NORMALIZATION
    generator = torch.Generator().manual_seed(seed)
    means = (2.0 * torch.rand(num_modes, 2, generator=generator) - 1.0) * box
    covariances = std**2 * torch.eye(2).expand(num_modes, 2, 2).clone()
    weights = torch.ones(num_modes) / num_modes
    return [GaussianMixture2D(means, covariances, weights, schedule=schedule)]


def make_fkc_problem(
    case: str = 'gaussian40',
    schedule: AffineSchedule | None = None,
    seed: int = 0,
    std: float | None = None,
    box: float | None = None,
    beta: int = 3,
) -> FKCProblem:
    """Construct the 40-Gaussian FKC case.

    :param case: ``'gaussian40'`` (40 equal-std modes, default ``beta = 3``;
        FKC temperature annealing).
    :param schedule: Affine noise schedule; defaults to the linear schedule.
    :param seed: Mode-placement seed for the ``gaussian40`` case.
    :param std: Per-mode standard deviation for ``gaussian40``; ``None`` uses
        the reference notebook's ``softplus(2.0) / 50``.
    :param box: Mode half-width for ``gaussian40``; ``None`` uses the reference
        notebook's ``40 / 50``.
    :param beta: Positive integer power for ``gaussian40`` (1, 2, or 3).
        ``beta=1`` is the untempered 40-mode control.
    :return: Configured :class:`FKCProblem`.
    """
    if case not in FKC_CASES:
        raise ValueError(f'unknown FKC case {case!r}; expected one of {list(FKC_CASES)}.')
    if isinstance(beta, bool) or not isinstance(beta, int) or beta not in (1, 2, 3):
        raise ValueError('gaussian40 beta must be an integer in {1, 2, 3}.')
    return FKCProblem(
        forty_gaussian_experts(std=std, box=box, seed=seed, schedule=schedule), (beta,), schedule
    )
