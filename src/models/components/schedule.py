"""Affine (Gaussian-path) noise schedules for flow matching.

Follows the flow-matching convention of Lipman et al. (arXiv:2412.06264): the
interpolation between noise and data is

    X_t = alpha_t * X_1 + sigma_t * X_0,

with ``X_1 ~ p_data`` and ``X_0 ~ p_noise``. With the default linear schedule
(``alpha_t = t``, ``sigma_t = 1 - t``) this gives the project convention

    t = 0  ->  pure noise   (X_0)
    t = 1  ->  pure data    (X_1)

All time tensors are broadcast against image tensors of shape ``(B, C, H, W)``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor

#: Quantity a network may predict (or a loss may regress against). All three are
#: linearly interconvertible given the noisy point ``x_t`` and time ``t``:
#:
#: * ``"noise"``    -> the noise sample ``X_0``
#: * ``"data"``     -> the data sample ``X_1``
#: * ``"velocity"`` -> the flow-matching velocity ``u_t = alphadot X_1 + sigmadot X_0``
Parameterization = Literal['noise', 'data', 'velocity']


def expand_t(t: Tensor, x: Tensor) -> Tensor:
    """Broadcast a per-sample time tensor against an image tensor.

    :param t: Time tensor of shape ``(B,)`` or scalar.
    :param x: Reference tensor, typically ``(B, C, H, W)``.
    :return: ``t`` reshaped to ``(B, 1, 1, 1)`` (or matching ``x.ndim``).
    """
    t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    if t.ndim == 0:
        t = t.expand(x.shape[0])
    return t.reshape(-1, *([1] * (x.ndim - 1)))


class AffineSchedule(ABC):
    """Abstract affine schedule defined by ``alpha_t`` and ``sigma_t``.

    Subclasses provide the four scalar coefficient functions; everything else
    (interpolation, velocity target, SDE drift/diffusion) is derived here as
    pure functions.
    """

    @property
    def prior_std(self) -> float:
        """Return the standard deviation of the noise prior."""
        return 1.0

    @abstractmethod
    def alpha(self, t: Tensor) -> Tensor:
        """Return the data coefficient ``alpha_t``."""

    @abstractmethod
    def sigma(self, t: Tensor) -> Tensor:
        """Return the noise coefficient ``sigma_t``."""

    @abstractmethod
    def alpha_dot(self, t: Tensor) -> Tensor:
        """Return the time derivative ``d alpha_t / dt``."""

    @abstractmethod
    def sigma_dot(self, t: Tensor) -> Tensor:
        """Return the time derivative ``d sigma_t / dt``."""

    # -- derived quantities -------------------------------------------------

    def nu_squared(self, t: Tensor) -> Tensor:
        """Diffusion coefficient ``nu_t**2`` of the flow-as-SDE.

        ``nu_t**2 = 2 (alphadot_t / alpha_t) sigma_t**2 - 2 sigma_t sigmadot_t``.
        """
        a, s = self.alpha(t), self.sigma(t)
        return 2.0 * (self.alpha_dot(t) / a) * s**2 - 2.0 * s * self.sigma_dot(t)

    # -- weight-rate parameterization ---------------------------------------
    #
    # The FKC weight ODE is ``dw = g_beta dt`` with the potential
    # ``g_beta = (nu_t**2 / 2)(||S_beta||**2 - sum_i beta_i ||s_t^i||**2)``.
    # The network's scalar head may predict either the potential ``g``
    # (``weight_rate_is_potential = True``) or the clock integrand
    # ``psi = g / nu_t**2`` (``False``). The two are equivalent in exact
    # arithmetic. Their regression scales and finite-step quadratures differ.
    # Neither parameterization bounds the score or potential globally.

    @property
    def weight_rate_is_potential(self) -> bool:
        """Whether the head predicts the potential ``g`` (``True``) or ``g/nu^2``."""
        return False

    def weight_rate(self, g: Tensor, t: Tensor) -> Tensor:
        """Convert the analytic potential ``g`` into the head's rate target at ``t``."""
        if self.weight_rate_is_potential:
            return g
        return g / self.nu_squared(t)

    def weight_span(self, s: Tensor, t: Tensor, t0: float = 0.0) -> Tensor:
        """Span multiplying the head's rate to accumulate the log-weight over ``[s, t]``."""
        if self.weight_rate_is_potential:
            return torch.as_tensor(t) - torch.as_tensor(s)
        return self.martingale_clock(t, t0) - self.martingale_clock(s, t0)

    def weight_rate_jvp_span(self, s: Tensor, t: Tensor, t0: float = 0.0) -> Tensor:
        """Span multiplying the iMF JVP correction of the head's rate."""
        if self.weight_rate_is_potential:
            return torch.as_tensor(t) - torch.as_tensor(s)
        return self.weight_span(s, t, t0) / self.nu_squared(s)

    # -- Doss-Sussmann martingale clock ------------------------------------
    #
    # For the flow-as-SDE ``dX_t = b(t, X_t) dt + nu_t dW_t`` the diffusion
    # martingale ``M_t = int_{t0}^t nu_s dW_s`` is a Gaussian martingale whose
    # quadratic variation is the deterministic clock
    #
    #     <M>_t = I * clock(t),   clock(t) = int_{t0}^t nu_s**2 ds       (ς_t).
    #
    # By Dambis-Dubins-Schwarz ``M_t = Wtilde_{clock(t)}`` for a standard BM
    # ``Wtilde``, so the Doss-Sussmann transform samples ``M`` by running the
    # (unchanged) polynomial Brownian driver on the warped axis ``clock``. The
    # clock starts at ``t0 > 0`` because ``nu_s**2`` blows up like ``1 / s`` at
    # the origin for the linear schedule.

    def _martingale_clock_integrand(self, t: Tensor) -> Tensor:
        """Integrand ``nu_t**2 = d<M>_t / dt`` of the martingale clock.

        Overridable hook: subclasses with a closed-form clock override
        :meth:`martingale_clock` directly; this default routes the numerical
        quadrature in the base :meth:`martingale_clock` through ``nu_squared``.

        :param t: Time tensor.
        :return: ``nu_t**2`` evaluated at ``t``.
        """
        return self.nu_squared(t)

    def martingale_clock(self, t: Tensor, t0: float = 0.0, *, num_nodes: int = 256) -> Tensor:
        """Doss-Sussmann variance clock ``clock(t) = int_{t0}^t nu_s**2 ds`` (``ς_t``).

        This is the quadratic variation ``<M>_t`` of the diffusion martingale
        ``M_t = int_{t0}^t nu_s dW_s``. The default implementation evaluates the
        integral by the composite trapezoidal rule on a shared ``[t0, t]`` grid;
        schedules with an elementary antiderivative (e.g. :class:`LinearSchedule`)
        override this with the exact closed form.

        :param t: Upper limits ``(B,)`` (or scalar). Must satisfy ``t >= t0``.
        :param t0: Lower clock start ``t0 > 0`` (avoids the ``1 / s`` singularity).
        :param num_nodes: Number of trapezoid nodes for the numerical fallback.
        :return: Clock values with the same shape as ``t``.
        """
        t = torch.as_tensor(t)
        # Shared grid u in [0, 1]; per-sample nodes s = t0 + u * (t - t0).
        u = torch.linspace(0.0, 1.0, num_nodes, device=t.device, dtype=t.dtype)
        span = (t - t0).clamp_min(0.0)  # (B,)
        nodes = t0 + span.unsqueeze(-1) * u  # (B, num_nodes)
        integrand = self._martingale_clock_integrand(nodes)  # (B, num_nodes)
        integral = torch.trapezoid(integrand, nodes, dim=-1)  # (B,)
        return integral.reshape(t.shape)

    # -- exponential-integrator (EI) Euler coefficients --------------------
    #
    # Applying the integrating factor ``alpha_r**-1`` to the flow-as-SDE and
    # integrating over ``[s, t]`` gives the exact three-term map
    #
    #     X_t = (alpha_s / alpha_t) X_s
    #           + (A_st / alpha_t) u_st
    #           + (1 / alpha_t) int_s^t nu_r dW_r,        A_st = int_s^t alpha_r dr,
    #
    # where ``u_st`` is the ``alpha``-weighted average velocity over the
    # interval. Only the two deterministic coefficients live here; the sampler
    # owns the stochastic term. Unlike the x-pred map these coefficients never
    # touch ``sigma`` or the SNR clock, so they stay finite at ``t = 1``.

    def _ei_alpha_integral(self, s: Tensor, t: Tensor, *, num_nodes: int = 256) -> Tensor:
        """Return ``A_st = int_s^t alpha_r dr`` by composite trapezoidal quadrature.

        Numerical fallback mirroring the :meth:`martingale_clock` quadrature
        pattern: a shared normalized grid ``u in [0, 1]`` is mapped to per-sample
        nodes ``r = s + u (t - s)``. Schedules with an elementary antiderivative
        (:class:`LinearSchedule`) overrides this exactly.

        The span is deliberately **not** clamped, so the result is antisymmetric
        (``A_ts == -A_st``) and integrates correctly for ``t < s``.

        :param s: Start times, broadcastable against ``t``.
        :param t: Terminal times, broadcastable against ``s``.
        :param num_nodes: Number of trapezoid nodes.
        :return: ``A_st`` with the broadcast shape of ``s`` and ``t``.
        """
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        # Shared grid u in [0, 1]; per-sample nodes r = s + u * (t - s).
        u = torch.linspace(0.0, 1.0, num_nodes, device=t.device, dtype=t.dtype)
        nodes = s.unsqueeze(-1) + (t - s).unsqueeze(-1) * u  # (..., num_nodes)
        integral = torch.trapezoid(self.alpha(nodes), nodes, dim=-1)
        return integral.reshape(t.shape)

    def ei_alpha_integral(self, s: Tensor, t: Tensor, *, num_nodes: int = 256) -> Tensor:
        """Public accessor for ``A_st = int_s^t alpha_r dr``.

        Schedules with an elementary antiderivative override the underlying
        :meth:`_ei_alpha_integral` exactly; this exposes that same quantity to
        maps that integrate in the EI state ``Y = alpha X`` (where the ``1 /
        alpha_t`` of :meth:`ei_euler_coefficients` is absent) without
        re-deriving the quadrature.

        :param s: Start times, broadcastable against ``t``.
        :param t: Terminal times, broadcastable against ``s``.
        :param num_nodes: Trapezoid nodes used only by the numerical fallback.
        :return: ``A_st`` with the broadcast shape of ``s`` and ``t``.
        """
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        return self._ei_alpha_integral(s, t, num_nodes=num_nodes)

    def ei_euler_coefficients(
        self, s: Tensor, t: Tensor, *, num_nodes: int = 256
    ) -> tuple[Tensor, Tensor]:
        """Return the deterministic coefficients of the EI-Euler flow map.

        The map on ``[s, t]`` is

        ``X_t = state_coeff * X_s + velocity_coeff * u_st + (1 / alpha_t) int_s^t nu_r dW_r``,

        with

        ``state_coeff = alpha_s / alpha_t``   and   ``velocity_coeff = A_st / alpha_t``,

        where ``A_st = int_s^t alpha_r dr`` comes from :meth:`_ei_alpha_integral`.
        Both coefficients depend on ``alpha`` alone, hence remain finite for all
        ``0 < s <= t <= 1`` and satisfy the diagonal identity
        ``ei_euler_coefficients(t, t) == (1, 0)`` exactly (``A_tt = 0``).

        :param s: Start times, broadcastable against ``t``.
        :param t: Terminal times, broadcastable against ``s``.
        :param num_nodes: Trapezoid nodes used only by the numerical fallback.
        :return: ``(state_coeff, velocity_coeff)`` with the broadcast shape of times.
        """
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        alpha_t = self.alpha(t)
        state_coeff = self.alpha(s) / alpha_t
        velocity_coeff = 2.0 * self._ei_alpha_integral(s, t, num_nodes=num_nodes) / alpha_t
        return state_coeff, velocity_coeff

    def ei_psd_target_weights(
        self, s: Tensor, m: Tensor, t: Tensor, *, num_nodes: int = 256
    ) -> tuple[Tensor, Tensor]:
        """Return the EI-Euler PSD weights for a split at ``m``.

        Writing ``calA_ab = int_a^b alpha_r dr``, semigroup composition gives
        ``u_st = (calA_sm / calA_st) u_sm + (calA_mt / calA_st) u_mt``.
        See ``docs/ei_euler_objective.md`` for the derivation.

        :param s: Start times.
        :param m: Split times, strictly inside ``(s, t)``.
        :param t: Terminal times.
        :param num_nodes: Quadrature nodes used by schedules without a closed form.
        :return: The left and right PSD target weights.
        :raises ValueError: If any split does not lie strictly inside its interval.
        """
        s, m, t = torch.broadcast_tensors(
            torch.as_tensor(s), torch.as_tensor(m), torch.as_tensor(t)
        )
        if torch.any((m <= s) | (m >= t)):
            raise ValueError('EI-Euler PSD split times must satisfy s < m < t.')

        integral_sm = self._ei_alpha_integral(s, m, num_nodes=num_nodes)
        integral_mt = self._ei_alpha_integral(m, t, num_nodes=num_nodes)
        integral_st = self._ei_alpha_integral(s, t, num_nodes=num_nodes)
        denominator = integral_st.clamp_min(torch.finfo(integral_st.dtype).eps)
        return integral_sm / denominator, integral_mt / denominator

    # -- annealed (zeta) exponential-integrator coefficients ---------------
    #
    # The probability-flow ODE and the flow-as-SDE are the ``zeta = 0`` and
    # ``zeta = 1`` members of the annealed SDE family
    #
    #     dX_t = [(1 + zeta^2) u_t(X_t) - zeta^2 (alphadot_t/alpha_t) X_t] dt
    #            + zeta nu_t dW_t,
    #
    # which shares the marginals ``p_t`` (paper, "Annealing the Doss-Sussman
    # ODE"). Its exponentially lifted (``Y = alpha_t^{zeta^2} X``) integral form
    # is
    #
    #     Y_t = Y_s + (1 + zeta^2) A_zeta(s, t) u_st
    #           + zeta int_s^t alpha_r^{zeta^2} nu_r dW_r,
    #
    #     A_zeta(s, t) = int_s^t alpha_r^{zeta^2} dr,
    #
    # whose martingale carries the annealed clock
    # ``zeta^2 int alpha_r^{2 zeta^2} nu_r^2 dr``. At ``zeta = 0`` the
    # martingale vanishes and the skeleton is the Euler ODE map
    # ``x + (t - s) u``; at ``zeta = 1`` everything reduces exactly to the
    # un-annealed EI-Euler quantities above.

    def annealed_ei_alpha_integral(
        self, s: Tensor, t: Tensor, zeta: float = 1.0, *, num_nodes: int = 256
    ) -> Tensor:
        """Return ``A_zeta(s, t) = int_s^t alpha_r^{zeta^2} dr``.

        The default implementation evaluates the integral by composite
        trapezoidal quadrature on a shared ``[s, t]`` grid; schedules with an
        elementary antiderivative (:class:`LinearSchedule`) override this
        exactly. The endpoints ``zeta in {0, 1}`` are exact by construction:
        ``zeta = 0`` gives ``t - s`` and ``zeta = 1`` delegates to
        :meth:`ei_alpha_integral` (closed form where available).

        :param s: Start times, broadcastable against ``t``.
        :param t: Terminal times, broadcastable against ``s``.
        :param zeta: Annealing amplitude in ``[0, 1]``.
        :param num_nodes: Trapezoid nodes used only by the numerical fallback.
        :return: ``A_zeta(s, t)`` with the broadcast shape of ``s`` and ``t``.
        """
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        if zeta == 1.0:
            return self.ei_alpha_integral(s, t, num_nodes=num_nodes)
        if zeta == 0.0:
            return t - s
        u = torch.linspace(0.0, 1.0, num_nodes, device=t.device, dtype=t.dtype)
        nodes = s.unsqueeze(-1) + (t - s).unsqueeze(-1) * u
        integral = torch.trapezoid(self.alpha(nodes) ** (zeta * zeta), nodes, dim=-1)
        return integral.reshape(t.shape)

    def annealed_ei_weighted_clock(
        self, t: Tensor, t0: float = 0.0, zeta: float = 1.0, *, num_nodes: int = 256
    ) -> Tensor:
        """Annealed martingale clock ``zeta^2 int_{t0}^t alpha_r^{2 zeta^2} nu_r^2 dr``.

        This is the quadratic variation of the annealed lifted martingale
        ``M~^zeta_t = zeta int_{t0}^t alpha_r^{zeta^2} nu_r dW_r``, so the
        polynomial Brownian path of the annealed Doss-Sussmann object lives on
        this clock exactly as the un-annealed EI path lives on
        :meth:`ei_weighted_clock`. ``zeta = 0`` returns zero identically (the
        annealed ODE carries no noise) and ``zeta = 1`` delegates to
        :meth:`ei_weighted_clock`.

        :param t: Upper limits ``(B,)`` (or scalar), with ``t >= t0``.
        :param t0: Lower clock start ``t0 > 0``.
        :param zeta: Annealing amplitude in ``[0, 1]``.
        :param num_nodes: Trapezoid nodes used only by the numerical fallback.
        :return: Clock values with the same shape as ``t``.
        """
        t = torch.as_tensor(t)
        if zeta == 1.0:
            return self.ei_weighted_clock(t, t0, num_nodes=num_nodes)
        if zeta == 0.0:
            return torch.zeros_like(t)
        u = torch.linspace(0.0, 1.0, num_nodes, device=t.device, dtype=t.dtype)
        span = (t - t0).clamp_min(0.0)
        nodes = t0 + span.unsqueeze(-1) * u
        integrand = (
            (zeta * zeta) * self.alpha(nodes) ** (2.0 * zeta * zeta) * self.nu_squared(nodes)
        )
        integral = torch.trapezoid(integrand, nodes, dim=-1)
        return integral.reshape(t.shape)

    def annealed_ei_euler_coefficients(
        self, s: Tensor, t: Tensor, zeta: float = 1.0, *, num_nodes: int = 256
    ) -> tuple[Tensor, Tensor]:
        """Return the deterministic coefficients of the annealed EI map.

        The annealed skeleton on ``[s, t]`` in physical X-units is

        ``X_t = state_coeff * X_s + velocity_coeff * u_st
        + (zeta / alpha_t^{zeta^2}) int_s^t alpha_r^{zeta^2} nu_r dW_r``,

        with ``state_coeff = (alpha_s / alpha_t)^{zeta^2}`` and
        ``velocity_coeff = (1 + zeta^2) A_zeta(s, t) / alpha_t^{zeta^2}``.
        The diagonal identity ``coefficients(t, t) == (1, 0)`` holds for every
        zeta, and ``zeta = 1`` reproduces :meth:`ei_euler_coefficients`
        exactly while ``zeta = 0`` gives the plain Euler map
        ``(1, t - s)``.

        :param s: Start times, broadcastable against ``t``.
        :param t: Terminal times, broadcastable against ``s``.
        :param zeta: Annealing amplitude in ``[0, 1]``.
        :param num_nodes: Trapezoid nodes used only by the numerical fallback.
        :return: ``(state_coeff, velocity_coeff)`` with the broadcast shape.
        """
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        alpha_s, alpha_t = self.alpha(s), self.alpha(t)
        power = zeta * zeta
        state_coeff = (alpha_s / alpha_t) ** power
        velocity_coeff = (
            (1.0 + power)
            * self.annealed_ei_alpha_integral(s, t, zeta, num_nodes=num_nodes)
            / alpha_t**power
        )
        return state_coeff, velocity_coeff

    def annealed_ei_martingale_variance(
        self, s: Tensor, t: Tensor, t0: float = 0.0, zeta: float = 1.0, *, num_nodes: int = 256
    ) -> Tensor:
        """Variance of the annealed martingale increment on ``[s, t]``.

        Equals the annealed clock difference; antisymmetric in ``(s, t)``
        exactly like :meth:`ei_martingale_variance`.

        :param s: Start times, broadcastable against ``t``.
        :param t: Terminal times, broadcastable against ``s``.
        :param t0: Shared clock anchor (cancels in the difference).
        :param zeta: Annealing amplitude in ``[0, 1]``.
        :return: Clock difference with the broadcast shape of ``s`` and ``t``.
        """
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        return self.annealed_ei_weighted_clock(
            t, t0, zeta, num_nodes=num_nodes
        ) - self.annealed_ei_weighted_clock(s, t0, zeta, num_nodes=num_nodes)

    def annealed_ei_driver_variance(
        self, s: Tensor, t: Tensor, t0: float = 0.0, zeta: float = 1.0, *, num_nodes: int = 256
    ) -> Tensor:
        """Variance of the scaled annealed driver at the terminal time ``t``.

        ``annealed_ei_driver_variance(s, t) = annealed_ei_martingale_variance(s, t)
        / alpha_t^{2 zeta^2}``, clamped at zero, mirroring
        :meth:`ei_driver_variance` for the annealed map.

        :param s: Start times, broadcastable against ``t``.
        :param t: Terminal times, broadcastable against ``s``, with ``t >= s``.
        :param t0: Shared clock anchor forwarded to the clock difference.
        :param zeta: Annealing amplitude in ``[0, 1]``.
        :return: Non-negative variance with the broadcast shape.
        """
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        variance = self.annealed_ei_martingale_variance(
            s, t, t0=t0, zeta=zeta, num_nodes=num_nodes
        ) / self.alpha(t) ** (2.0 * zeta * zeta)
        return variance.clamp_min(0.0)

    # -- EI martingale term: scaling and variance --------------------------
    #
    # The stochastic term of the EI-Euler map is the *scaled* diffusion
    # martingale increment
    #
    #     (1 / alpha_t) int_s^t nu_r dW_r,
    #
    # built from the flow-as-SDE diffusion ``nu_r`` (:meth:`nu_squared`). Only
    # this EI form pairs with :meth:`ei_euler_coefficients`.
    #
    # Because ``int_s^t nu_r dW_r`` is a Gaussian martingale increment with
    # deterministic quadratic variation, its variance is exactly the
    # martingale-clock difference; the existing :meth:`martingale_clock` is
    # reused, so no second quadrature path is introduced.

    def ei_martingale_variance(self, s: Tensor, t: Tensor, *, t0: float = 0.0) -> Tensor:
        """Variance of the EI martingale increment ``int_s^t nu_r dW_r``.

        The increment is Gaussian with deterministic quadratic variation, so

            Var(int_s^t nu_r dW_r) = int_s^t nu_r**2 dr = clock(t) - clock(s),

        evaluated through the existing :meth:`martingale_clock`. The shared
        anchor ``t0`` cancels in the difference, hence every schedule reuses its
        own clock (closed form or trapezoid) and no new quadrature is added.

        This measures the flow-as-SDE diffusion ``nu_r``.

        :param s: Start times, broadcastable against ``t``.
        :param t: Terminal times, broadcastable against ``s``, with ``t >= s``.
        :param t0: Shared clock anchor forwarded to :meth:`martingale_clock`. It
            cancels in the difference, so it only needs to be a legal anchor for
            the schedule (:class:`LinearSchedule` requires a positive origin).
        :return: Clock difference with the broadcast shape of ``s`` and ``t``.
            Left unclamped, so it is antisymmetric and negative when ``t < s``;
            use :meth:`ei_driver_variance` for the clamped, scaled form.
        """
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        return self.ei_weighted_clock(t, t0) - self.ei_weighted_clock(s, t0)

    def ei_martingale_scale(self, t: Tensor) -> Tensor:
        """Exponential-integrator scaling ``1 / alpha_t`` of the martingale term.

        Applying the integrating factor ``alpha_r**-1`` to the flow-as-SDE and
        undoing it at the terminal time leaves ``1 / alpha_t`` multiplying the
        accumulated martingale in the flow map

            Phi_st(x) = (alpha_s / alpha_t) x + (A_st / alpha_t) u_st
                        + (1 / alpha_t) int_s^t nu_r dW_r.

        Like :meth:`ei_euler_coefficients` it depends on ``alpha`` alone, so it
        never touches ``sigma`` or the SNR clock and stays finite on ``(0, 1]``.

        :param t: Terminal times.
        :return: ``1 / alpha_t`` with the shape of ``t``.
        """
        return 1.0 / self.alpha(t)

    def ei_driver_variance(self, s: Tensor, t: Tensor, *, t0: float = 0.0) -> Tensor:
        """Variance of the scaled EI driver ``(1 / alpha_t) int_s^t nu_r dW_r``.

        Composes the two pieces above into the terminal variance the sampler
        actually injects at ``t``:

            ei_driver_variance(s, t) = ei_martingale_variance(s, t) / alpha_t**2.

        The result is clamped at zero so round-off (or an accidentally reversed
        interval) can never feed a negative variance into a ``sqrt``.

        :param s: Start times, broadcastable against ``t``.
        :param t: Terminal times, broadcastable against ``s``, with ``t >= s``.
        :param t0: Shared clock anchor forwarded to :meth:`ei_martingale_variance`.
        :return: Non-negative variance with the broadcast shape of ``s`` and ``t``.
        """
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        variance = self.ei_martingale_variance(s, t, t0=t0) / self.alpha(t) ** 2
        return variance.clamp_min(0.0)

    # -- sampling / targets (pure) -----------------------------------------

    def sample_x_t(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Form the interpolated point ``X_t = alpha_t X_1 + sigma_t X_0``.

        :param x_0: Noise sample ``X_0``, shape ``(B, C, H, W)``.
        :param x_1: Data sample ``X_1``, shape ``(B, C, H, W)``.
        :param t: Time tensor, shape ``(B,)`` or scalar.
        :return: Interpolated tensor ``X_t``.
        """
        a, s = expand_t(self.alpha(t), x_1), expand_t(self.sigma(t), x_1)
        return a * x_1 + s * x_0

    def velocity(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Marginal flow-matching velocity ``u_t = alphadot_t X_1 + sigmadot_t X_0``.

        :return: Conditional velocity target for plain flow matching.
        """
        a_dot = expand_t(self.alpha_dot(t), x_1)
        s_dot = expand_t(self.sigma_dot(t), x_1)
        return a_dot * x_1 + s_dot * x_0

    def drift(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """SDE drift target ``b_t(x) = 2 u_t(x) - (alphadot_t / alpha_t) x``.

        This is the stochastic-flow drift the SSFM model regresses against
        (Holderrieth flow-as-SDE form), evaluated at ``x = X_t``.
        """
        x_t = self.sample_x_t(x_0, x_1, t)
        u_t = self.velocity(x_0, x_1, t)
        ratio = expand_t(self.alpha_dot(t) / self.alpha(t), x_t)
        return 2.0 * u_t - ratio * x_t

    # -- parameterization conversions (pure) -------------------------------
    #
    # The network output and the loss target may each be expressed as noise,
    # data, or velocity. Given the noisy point ``x_t`` and time ``t``, any one
    # of the three determines the other two via the linear system
    #
    #     x_t = alpha_t X_1 + sigma_t X_0,
    #     u_t = alphadot_t X_1 + sigmadot_t X_0.

    def to_data(self, value: Tensor, kind: Parameterization, x_t: Tensor, t: Tensor) -> Tensor:
        """Convert ``value`` (interpreted as ``kind``) into the data point ``X_1``.

        :param value: Tensor holding the quantity, shape ``(B, C, H, W)``.
        :param kind: What ``value`` represents (``"noise"``/``"data"``/``"velocity"``).
        :param x_t: The noisy interpolation point ``X_t``.
        :param t: Time tensor, shape ``(B,)`` or scalar.
        :return: The data sample ``X_1``.
        """
        if kind == 'data':
            return value
        a, s = expand_t(self.alpha(t), x_t), expand_t(self.sigma(t), x_t)
        if kind == 'noise':
            return (x_t - s * value) / a
        # velocity: x_1 = (sigmadot x_t - sigma u) / (alpha sigmadot - sigma alphadot)
        a_dot, s_dot = expand_t(self.alpha_dot(t), x_t), expand_t(self.sigma_dot(t), x_t)
        det = a * s_dot - s * a_dot
        return (s_dot * x_t - s * value) / det

    def to_noise(self, value: Tensor, kind: Parameterization, x_t: Tensor, t: Tensor) -> Tensor:
        """Convert ``value`` (interpreted as ``kind``) into the noise point ``X_0``.

        :param value: Tensor holding the quantity, shape ``(B, C, H, W)``.
        :param kind: What ``value`` represents.
        :param x_t: The noisy interpolation point ``X_t``.
        :param t: Time tensor, shape ``(B,)`` or scalar.
        :return: The noise sample ``X_0``.
        """
        if kind == 'noise':
            return value
        a, s = expand_t(self.alpha(t), x_t), expand_t(self.sigma(t), x_t)
        if kind == 'data':
            return (x_t - a * value) / s
        # velocity: x_0 = (alpha u - alphadot x_t) / (alpha sigmadot - sigma alphadot)
        a_dot, s_dot = expand_t(self.alpha_dot(t), x_t), expand_t(self.sigma_dot(t), x_t)
        det = a * s_dot - s * a_dot
        return (a * value - a_dot * x_t) / det

    def to_velocity(self, value: Tensor, kind: Parameterization, x_t: Tensor, t: Tensor) -> Tensor:
        """Convert ``value`` (interpreted as ``kind``) into the velocity ``u_t``.

        :param value: Tensor holding the quantity, shape ``(B, C, H, W)``.
        :param kind: What ``value`` represents.
        :param x_t: The noisy interpolation point ``X_t``.
        :param t: Time tensor, shape ``(B,)`` or scalar.
        :return: The flow-matching velocity ``u_t``.
        """
        if kind == 'velocity':
            return value
        x_0 = self.to_noise(value, kind, x_t, t)
        x_1 = self.to_data(value, kind, x_t, t)
        return self.velocity(x_0, x_1, t)

    def convert(
        self,
        value: Tensor,
        kind: Parameterization,
        target: Parameterization,
        x_t: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Convert ``value`` from parameterization ``kind`` to ``target``.

        Lets the network-output parameterization and the loss-target
        parameterization be chosen independently.

        :param value: Tensor holding the quantity, shape ``(B, C, H, W)``.
        :param kind: What ``value`` currently represents.
        :param target: Desired parameterization of the returned tensor.
        :param x_t: The noisy interpolation point ``X_t``.
        :param t: Time tensor, shape ``(B,)`` or scalar.
        :return: ``value`` expressed in the ``target`` parameterization.
        """
        if kind == target:
            return value
        if target == 'data':
            return self.to_data(value, kind, x_t, t)
        if target == 'noise':
            return self.to_noise(value, kind, x_t, t)
        return self.to_velocity(value, kind, x_t, t)


class LinearSchedule(AffineSchedule):
    """Linear (rectified-flow) schedule: ``alpha_t = t``, ``sigma_t = 1 - t``.

    Gives ``t = 0 -> noise`` and ``t = 1 -> data`` with constant velocity
    ``u_t = X_1 - X_0``.

    .. note::
        The SDE drift ratio ``alpha_dot / alpha = 1 / t`` and diffusion
        ``nu = sqrt(2 (1 - t) / t)`` are **singular at t = 0``. Use an epsilon
            """

    def __init__(self, noise_scale: float = 1.0) -> None:
        if noise_scale <= 0.0:
            raise ValueError(f'noise_scale must be positive; got {noise_scale}.')
        self.noise_scale = noise_scale

    @property
    def prior_std(self) -> float:
        """Return the linear path's noise standard deviation."""
        return self.noise_scale

    def alpha(self, t: Tensor) -> Tensor:  # noqa: D102
        return t

    def sigma(self, t: Tensor) -> Tensor:  # noqa: D102
        return self.noise_scale * (1.0 - t)

    def alpha_dot(self, t: Tensor) -> Tensor:  # noqa: D102
        return torch.ones_like(t)

    def sigma_dot(self, t: Tensor) -> Tensor:  # noqa: D102
        return -self.noise_scale * torch.ones_like(t)

    def martingale_clock(self, t: Tensor, t0: float = 0.0, *, num_nodes: int = 256) -> Tensor:
        """Closed-form clock ``clock(t) = 2 log(t / t0) - 2 (t - t0)``.

        For the linear schedule ``nu_s**2 = 2 (1 - s) / s``, so

            clock(t) = int_{t0}^t 2 (1 - s) / s ds = 2 log(t / t0) - 2 (t - t0),

        which diverges as ``t0 -> 0+`` through the ``log`` term (the ``1 / s``
        blow-up of ``nu_s**2`` at the origin); hence ``t0 > 0`` is required.

        :param t: Upper limits ``(B,)`` (or scalar), with ``t >= t0 > 0``.
        :param t0: Lower clock start ``t0 > 0``.
        :param num_nodes: Ignored (kept for signature compatibility).
        :return: Clock values with the same shape as ``t``.
        """
        if t0 <= 0.0:
            raise ValueError(f'LinearSchedule martingale clock needs t0 > 0; got {t0}.')
        t = torch.as_tensor(t)
        return self.noise_scale**2 * (2.0 * torch.log(t / t0) - 2.0 * (t - t0))

    def _ei_alpha_integral(self, s: Tensor, t: Tensor, *, num_nodes: int = 256) -> Tensor:
        """Closed-form ``A_st = (t**2 - s**2) / 2``.

        With ``alpha_r = r`` the integral is elementary:

            A_st = int_s^t r dr = (t**2 - s**2) / 2.

        Note ``noise_scale`` rescales ``sigma`` only, so it does not enter here;
        the EI-Euler state ratio is likewise ``alpha_s / alpha_t = s / t``.

        :param s: Start times, broadcastable against ``t``.
        :param t: Terminal times, broadcastable against ``s``.
        :param num_nodes: Ignored (kept for signature compatibility).
        :return: ``A_st`` with the broadcast shape of ``s`` and ``t``.
        """
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        return 0.5 * (t**2 - s**2)

    def ei_weighted_clock(self, t: Tensor, t0: float = 0.0, *, num_nodes: int = 256) -> Tensor:
        """Closed-form ``int alpha_r**2 nu_r**2 dr`` for the linear schedule."""
        del num_nodes
        t = torch.as_tensor(t)
        lower = torch.as_tensor(t0, device=t.device, dtype=t.dtype)

        def antiderivative(r: Tensor) -> Tensor:
            return self.noise_scale**2 * (r**2 - (2.0 / 3.0) * r**3)

        return antiderivative(t) - antiderivative(lower)

    def annealed_ei_alpha_integral(
        self, s: Tensor, t: Tensor, zeta: float = 1.0, *, num_nodes: int = 256
    ) -> Tensor:
        """Closed-form ``A_zeta(s, t) = (t^{1+zeta^2} - s^{1+zeta^2}) / (1 + zeta^2)``.

        With ``alpha_r = r`` the integral is elementary for every zeta:

            ``A_zeta = int_s^t r^{zeta^2} dr = (t^{1+zeta^2} - s^{1+zeta^2}) / (1 + zeta^2)``.

        ``zeta = 1`` reproduces ``0.5 * (t**2 - s**2)`` and ``zeta = 0`` gives
        ``t - s`` (the plain Euler skeleton). ``noise_scale`` rescales ``sigma``
        only, so it does not enter.

        :param s: Start times, broadcastable against ``t``.
        :param t: Terminal times, broadcastable against ``s``.
        :param zeta: Annealing amplitude in ``[0, 1]``.
        :param num_nodes: Ignored (kept for signature compatibility).
        :return: ``A_zeta(s, t)`` with the broadcast shape of ``s`` and ``t``.
        """
        del num_nodes
        s, t = torch.broadcast_tensors(torch.as_tensor(s), torch.as_tensor(t))
        power = zeta * zeta
        return (t ** (power + 1.0) - s ** (power + 1.0)) / (power + 1.0)

    def annealed_ei_weighted_clock(
        self, t: Tensor, t0: float = 0.0, zeta: float = 1.0, *, num_nodes: int = 256
    ) -> Tensor:
        """Closed-form annealed clock for the linear schedule.

        With ``nu_r**2 = 2 (1 - r) / r`` the annealed clock integrand is
        ``zeta^2 r^{2 zeta^2} 2 (1 - r) / r = 2 zeta^2 (1 - r) r^{2 zeta^2 - 1}``,
        whose antiderivative is

            ``f_zeta(r) = noise_scale^2 (r^{2 zeta^2} - (2 zeta^2 / (2 zeta^2 + 1)) r^{2 zeta^2 + 1})``,

        so ``clock_zeta(t) = f_zeta(t) - f_zeta(t0)``. At ``zeta = 1`` this is
        exactly the un-annealed ``r**2 - (2/3) r**3`` antiderivative; at
        ``zeta = 0`` it is identically zero (the annealed ODE carries no
        noise). Both endpoints are therefore exact without delegation.

        :param t: Upper limits ``(B,)`` (or scalar), with ``t >= t0 > 0``.
        :param t0: Lower clock start ``t0 > 0``.
        :param zeta: Annealing amplitude in ``[0, 1]``.
        :param num_nodes: Ignored (kept for signature compatibility).
        :return: Clock values with the same shape as ``t``.
        """
        del num_nodes
        t = torch.as_tensor(t)
        lower = torch.as_tensor(t0, device=t.device, dtype=t.dtype)
        power = 2.0 * zeta * zeta

        def antiderivative(r: Tensor) -> Tensor:
            return self.noise_scale**2 * (r**power - (power / (power + 1.0)) * r ** (power + 1.0))

        return antiderivative(t) - antiderivative(lower)


class GeometricSchedule(AffineSchedule):
    """Variance-exploding (VE) geometric noise schedule from the FKC GMM notebook.

    The reference for the 40-Gaussian FKC experiment
    (``applications/temperature_annealing/runner/notebooks/
    gmm_temp_annealed_birth_death.ipynb``) uses ``GeometricNoiseSchedule`` on a
    physical time ``tau`` running data (``tau = 0``) to noise (``tau = 1``)::

        sigma_ref(tau) = sigma_min * sqrt(sigma_diff**(2 tau) - 1),
        g_ref(tau)     = sigma_min * sigma_diff**tau * sqrt(2 log sigma_diff),
        h_ref(tau)     = int_0^tau g_ref**2 = sigma_ref(tau)**2,

    with ``sigma_diff = sigma_max / sigma_min`` (the geometric VE schedule of
    Karras et al. 2022, Eqs. 192/199). It is a *pure* variance-exploding path:
    there is no signal scaling, so ``alpha_tau = 1``. The GMM notebook uses
    ``sigma_min = 0.01`` and ``sigma_max = 10 * sqrt(temperature)``.

    We express it in this repo's convention (``t = 0`` noise, ``t = 1`` data)
    via ``tau = 1 - t``::

        alpha(t) = 1,
        sigma(t) = sigma_min * sqrt(sigma_diff**(2 (1 - t)) - 1),

    so ``sigma(1) = 0`` (data) and ``sigma(0) = sqrt(sigma_max**2 - sigma_min**2)
    ~= sigma_max`` (noise). The flow-as-SDE diffusion is the reference's
    ``g_ref(1 - t)**2``::

        nu_t**2 = 2 sigma_min**2 log(sigma_diff) sigma_diff**(2 (1 - t)),

    which is finite at both endpoints (``nu_0**2 = 2 sigma_min**2 log(sigma_diff)
    sigma_diff**2``), so ``t0 = 0`` is admissible.

    Because ``alpha_t = 1`` the drift term ``a_t = (alpha_dot_t / alpha_t) x``
    vanishes, so the composed FKC velocity and weight reduce *exactly* to the
    reference's vector-field form, ``u_beta = sum_i beta_i u^i + (1 - B) a_t``
    and ``g_beta = 0.5 nu_t**2 (||s_beta||**2 - sum_i beta_i ||s_i||**2)``.
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 10.0) -> None:
        """Initialize the schedule.

        :param sigma_min: Noise floor scale (reference ``0.01``).
        :param sigma_max: Noise ceiling (reference ``10 * sqrt(temperature)``).
        """
        if sigma_min <= 0.0:
            raise ValueError(f'sigma_min must be positive; got {sigma_min}.')
        if sigma_max <= sigma_min:
            raise ValueError(f'need 0 < sigma_min < sigma_max; got {sigma_min}, {sigma_max}.')
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_diff = self.sigma_max / self.sigma_min
        self._log_diff = math.log(self.sigma_diff)

    @property
    def prior_std(self) -> float:
        """Noise-end standard deviation ``sigma(0) = sqrt(sigma_max**2 - sigma_min**2)``."""
        return math.sqrt(self.sigma_max**2 - self.sigma_min**2)

    def _sigma_squared(self, t: Tensor) -> Tensor:
        """Return ``sigma(t)**2 = sigma_min**2 (sigma_diff**(2 (1 - t)) - 1)``."""
        t = torch.as_tensor(t)
        return self.sigma_min**2 * (self.sigma_diff ** (2.0 * (1.0 - t)) - 1.0)

    def alpha(self, t: Tensor) -> Tensor:  # noqa: D102
        return torch.ones_like(torch.as_tensor(t))

    def sigma(self, t: Tensor) -> Tensor:  # noqa: D102
        return self._sigma_squared(t).clamp_min(0.0).sqrt()

    def alpha_dot(self, t: Tensor) -> Tensor:  # noqa: D102
        return torch.zeros_like(torch.as_tensor(t))

    def sigma_dot(self, t: Tensor) -> Tensor:  # noqa: D102
        """Return ``d sigma_t / dt`` (finite for ``t < 1``, ``-inf`` at ``t = 1``).

        Diverges at the data endpoint, but the finite product ``sigma_t *
        sigma_dot_t`` used by :meth:`nu_squared` is overridden below, so nothing
        in the FKC machinery evaluates this singular factor directly.
        """
        t = torch.as_tensor(t)
        numerator = self.sigma_min**2 * self._log_diff * self.sigma_diff ** (2.0 * (1.0 - t))
        sigma = self.sigma(t)
        tiny = torch.finfo(sigma.dtype).tiny
        return -numerator / sigma.clamp_min(tiny)

    def nu_squared(self, t: Tensor) -> Tensor:
        """Closed-form ``nu_t**2 = 2 sigma_min**2 log(sigma_diff) sigma_diff**(2 (1 - t))``.

        Equals the reference's ``g_ref(1 - t)**2`` (the VE diffusion at the
        mirrored time) and stays finite at both endpoints.
        """
        t = torch.as_tensor(t)
        return 2.0 * self.sigma_min**2 * self._log_diff * self.sigma_diff ** (2.0 * (1.0 - t))

    @property
    def weight_rate_is_potential(self) -> bool:
        """Predict the physical-time potential rate, with no extra division by nu².

        For the default parameters nu²(1) is positive (~0.00138), not zero.
        Finite-width Gaussian experts have no time-endpoint score singularity;
        neither their scores nor the potential are globally bounded in x.
        """
        return True

    def martingale_clock(self, t: Tensor, t0: float = 0.0, *, num_nodes: int = 256) -> Tensor:
        """Closed-form clock ``int_{t0}^t nu_r**2 dr = sigma(t0)**2 - sigma(t)**2``.

        Since ``nu_t**2 = -d/dt sigma_t**2``, the VE clock is just the drop in
        noise variance between ``t0`` and ``t``. Finite at ``t0 = 0``.

        :param t: Upper limits ``(B,)`` (or scalar), with ``t >= t0``.
        :param t0: Lower clock start.
        :param num_nodes: Ignored (kept for signature compatibility).
        """
        del num_nodes
        t = torch.as_tensor(t)
        t0_t = torch.as_tensor(t0, device=t.device, dtype=t.dtype)
        return self._sigma_squared(t0_t) - self._sigma_squared(t)

    def ei_weighted_clock(self, t: Tensor, t0: float = 0.0, *, num_nodes: int = 256) -> Tensor:
        """Closed-form ``int_{t0}^t alpha_r**2 nu_r**2 dr``.

        With ``alpha_t = 1`` this coincides with :meth:`martingale_clock`.

        :param t: Upper limits ``(B,)`` (or scalar), with ``t >= t0``.
        :param t0: Lower clock start.
        :param num_nodes: Ignored (kept for signature compatibility).
        """
        return self.martingale_clock(t, t0, num_nodes=num_nodes)
