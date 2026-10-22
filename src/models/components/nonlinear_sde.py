"""The 1D non-linear SDE toy of arXiv:2606.01086, Section 4.1.

The system is a non-linear drift, additive-noise SDE intended to mimic the
variance-preserving reverse diffusion SDE:

    ``dX_t = [X_t - X_t^3] dt + sqrt(beta_t) dW_t``,

where ``beta_t`` is the linear interpolation between ``beta_max`` at the
noise end ``t = 0`` and ``beta_min`` at the data end ``t = 1`` (paper:
``beta_min = 0.1``, ``beta_max = 20``). Early dynamics are noise-dominated
(the double-well drift ``x - x^3`` pulls toward the wells at ``±1``), and
the shrinking ``beta_t`` concentrates the process onto the wells — the
qualitative behaviour of the reverse VP diffusion.

Because the drift and diffusion coefficients are known in closed form, this
SDE provides ground truth for:

- *marginal states*: ``X_s ~ p_s`` approximated by a fine Euler-Maruyama
  rollout from ``X_0 ~ N(0, beta_max I)`` (Algorithm 1, line 3: "Simulate");
- *Euler-Maruyama steps*: ``X_s + f(s, X_s) h + sqrt(beta_s) (W_t - W_s)``
  over small ``h`` (Algorithm 1, line 6);
- *strong references*: a fine EM solve of the true SDE on a shared
  piecewise polynomial Brownian realization, for strong-error evaluation
  of a learned Itô map (paper, Figures 2-3).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .gaussian import randn_like


@dataclass(frozen=True)
class NonlinearSDE:
    """Non-linear drift, additive-noise 1D SDE with a linear ``beta_t`` schedule.

    :param beta_min: Noise level at the data end ``t = 1``.
    :param beta_max: Noise level at the noise end ``t = 0``.
    :param init_std: Standard deviation of the initial condition ``X_0``.
        The reverse-VP reading starts from *pure noise*, so the default is
        ``1.0`` — ``beta_max`` scales the *diffusion coefficient* over time,
        not the initial state. A large init (e.g. ``sqrt(beta_max)``) would
        seed tail states at ``|x| ~ 20`` where the double-well drift is
        ``~ -8000`` and no one-step Euler-Maruyama target is meaningful.
    """

    beta_min: float = 0.1
    beta_max: float = 20.0
    init_std: float = 1.0

    def __post_init__(self) -> None:
        """Validate that the noise levels and init spread are positive."""
        if self.beta_min <= 0.0 or self.beta_max <= 0.0:
            raise ValueError(
                f'beta levels must be positive; got ({self.beta_min}, {self.beta_max}).'
            )
        if self.init_std <= 0.0:
            raise ValueError(f'init_std must be positive; got {self.init_std}.')

    def beta(self, t: Tensor | float) -> Tensor:
        """Return ``beta_t`` interpolated linearly from ``beta_max`` to ``beta_min``."""
        t_t = torch.as_tensor(t)
        weight = (self.beta_max - self.beta_min) * (1.0 - t_t) + self.beta_min
        return weight if t_t.ndim else weight.reshape(())

    def drift(self, x: Tensor, t: Tensor | float) -> Tensor:
        """Return the double-well drift ``f(t, x) = x - x^3``."""
        del t
        return x - x.pow(3)

    def diffusion(self, t: Tensor | float) -> Tensor:
        """Return the additive diffusion coefficient ``g(t) = sqrt(beta_t)``."""
        return self.beta(t).clamp_min(0.0).sqrt()

    def tamed_drift(self, x: Tensor, t: Tensor | float, step: Tensor | float) -> Tensor:
        """Return the tamed EM drift ``f / (1 + step * |f|)`` at ``x``.

        The superlinear cubic term makes plain Euler-Maruyama unstable for
        tail states (``X_0 ~ N(0, beta_max)`` reaches ``|x| ~ 20`` where a
        plain step overshoots the restoring force and diverges). Tamed EM
        (the standard remedy for superlinear drifts) caps the step at one
        unit of drift and stays strongly consistent, so *every* solve of
        this SDE rides it: marginal simulation, strong references, and the
        one-step training targets (Algorithm 1, line 6). At regular states
        ``h |f| << 1`` and the taming is inert; at the rare tail states
        (``|x| ~ 20``, drift ``~ -8000``) it keeps every EM step bounded,
        so targets, simulation, and evaluation all describe one consistent
        process.
        """
        drift = self.drift(x, t)
        step_t = torch.as_tensor(step, device=drift.device, dtype=drift.dtype)
        scale = (1.0 + step_t * drift.abs()).reciprocal()
        return drift * scale

    def initial_condition(self, noise: Tensor) -> Tensor:
        """Return ``X_0 = init_std * noise`` for a standard-normal draw."""
        return float(self.init_std) * noise

    @torch.no_grad()
    def sample_marginals(
        self,
        noise: Tensor,
        sim_steps: int,
        *,
        max_index: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Simulate ``X_s ~ p_s`` on a uniform grid and return one draw per sample.

        Rolls a fine EM path from ``X_0 = sqrt(beta_max) noise`` over
        ``sim_steps`` uniform intervals of ``[0, 1]``, stores the state at
        every node, then gathers each sample's state at an independently
        drawn uniform grid index ``k ~ U{1, .., max_index}``.

        :param noise: Standard-normal draw of shape ``(B, ...)`` seeding ``X_0``.
        :param sim_steps: Number of uniform EM intervals over ``[0, 1]``.
        :param max_index: Largest grid index to draw (defaults to
            ``sim_steps``). Callers that need a step floor ``dt`` pass
            ``max_index = floor((1 - dt) * sim_steps)`` so ``s = k /
            sim_steps <= 1 - dt``.
        :return: ``(x_s, s, states)`` with ``x_s`` the gathered states
            ``(B, ...)``, ``s`` the gather times ``(B,)`` float32, and
            ``states`` the full ``(B, sim_steps + 1, ...)`` rollout.
        """
        if sim_steps < 1:
            raise ValueError(f'sim_steps must be >= 1; got {sim_steps}.')
        top = sim_steps if max_index is None else max_index
        if top < 1 or top > sim_steps:
            raise ValueError(f'max_index must lie in [1, sim_steps]; got {top}.')
        batch = noise.shape[0]
        state = self.initial_condition(noise)
        step = 1.0 / sim_steps
        nodes = torch.linspace(0.0, 1.0, sim_steps + 1, device=noise.device, dtype=noise.dtype)
        states = [state]
        for index in range(sim_steps):
            mid = nodes[index] + 0.5 * step
            dw = randn_like(state) * (step**0.5)
            state = state + self.tamed_drift(state, mid, step) * step + self.diffusion(mid) * dw
            states.append(state)
        stacked = torch.stack(states, dim=1)
        index = torch.randint(1, top + 1, (batch,), device=noise.device, generator=generator)
        s = index.to(torch.float32) * (1.0 / sim_steps)
        x_s = stacked[torch.arange(batch, device=noise.device), index]
        return x_s, s, stacked

    @torch.no_grad()
    def strong_reference(
        self,
        x_0: Tensor,
        realization: object,
        *,
        fine_steps: int,
    ) -> Tensor:
        """Return the fine EM solve of the true SDE on a shared realization.

        The reference rides the *exact* increments of the piecewise
        polynomial realization (its zeroth coefficients ``I^{(0)}`` are the
        true ``W`` increments), so a learned Itô map evaluated on
        restrictions of the same realization is pathwise-coupled to this
        reference — the strong-error metric of the paper's Figure 2.

        :param x_0: Initial states ``X_0`` of shape ``(B, ...)``.
        :param realization: A
            :class:`~src.models.components.strong_convergence.PiecewiseBrownianPath`
            with at least ``fine_steps`` uniform leaves on ``[0, 1]``.
        :param fine_steps: Number of fine EM steps over ``[0, 1]``.
        :return: The reference endpoint ``X_1`` of shape ``(B, ...)``.
        """
        return self.strong_reference_states(x_0, realization, fine_steps=fine_steps)[:, -1]

    @torch.no_grad()
    def strong_reference_states(
        self,
        x_0: Tensor,
        realization: object,
        *,
        fine_steps: int,
    ) -> Tensor:
        """Return the fine EM solve of the true SDE at every fine node.

        Same pathwise-coupled construction as :meth:`strong_reference`, but
        the state is recorded after every fine sub-step — the time-vs-value
        trajectory panel of the toy evaluation.

        :return: Reference states of shape ``(B, fine_steps + 1, ...)``.
        """
        nodes = realization.nodes[: fine_steps + 1]
        if float(nodes[0]) != 0.0 or abs(float(nodes[-1]) - 1.0) > 1e-6:
            raise ValueError('strong_reference requires a realization on [0, 1].')
        leaves = realization.leaves[:fine_steps]
        if any(leaf.degree < 1 for leaf in leaves):
            raise ValueError('strong_reference requires degree >= 1 leaves.')
        state = x_0
        states = [state]
        for index in range(fine_steps):
            mid = 0.5 * (nodes[index] + nodes[index + 1])
            step = nodes[index + 1] - nodes[index]
            state = (
                state
                + self.tamed_drift(state, mid, step) * step
                + self.diffusion(mid) * leaves[index].raw_increment().to(dtype=state.dtype)
            )
            states.append(state)
        return torch.stack(states, dim=1)
