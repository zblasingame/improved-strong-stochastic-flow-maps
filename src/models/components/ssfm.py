"""Abstract stochastic-flow-map network (SSFMNet).

An SSFMNet parameterizes the flow-map operator ``Phi_{s,t}`` of an additive-noise
SDE. Unlike a plain flow-matching network (which conditions on a single time
``t``), it conditions on a *start* time ``s`` and a *terminal* time ``t`` and
ingests the polynomial Brownian *coefficients* that drive the stochastic part of
the map (cf. :mod:`src.models.components.brownian`).

This module defines only the abstract contract; concrete backbones (e.g. a DiT
two-time variant) subclass it by implementing :meth:`SSFMNet._field`, which
returns the raw drift field (and, under the Euler-Maruyama parameterization, a
diffusion field). The base class assembles these into the flow-map step ``Psi``:

- ``'euler'`` (default): ``Psi_{s,t}(x) = x + (t - s) drift``.
- ``'euler_maruyama'``: ``Psi_{s,t}(x) = x + (t - s) drift + W_{s,t} diffusion``,

where ``W_{s,t}`` is the *polynomial* Brownian increment evaluated from a
:class:`~src.models.components.brownian.BrownianPolynomial` the caller assigns
via :meth:`SSFMNet.set_brownian` before each step. The model is
**unconditional**: no class labels are supplied.

Notation
--------
- ``s`` : start time (per-sample, shape ``(B,)`` or broadcastable).
- ``t`` : terminal time.
- ``coefficients`` : polynomial Brownian coefficient features, typically
  ``(B, N, C, H, W)`` from
  :meth:`src.models.components.brownian.BrownianPolynomial.coefficient_features`,
  where ``N`` is the polynomial degree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch.nn as nn
from torch import Tensor

from .brownian import BrownianPolynomial

Parameterization = Literal['euler', 'euler_maruyama']


def _diagonal_velocity(net: SSFMNet, x: Tensor, t: Tensor, degree: int) -> Tensor:
    """Evaluate a backbone's Euler velocity head on the diagonal ``s = t``.

    The diagonal student of the EI-Euler objective
    (:func:`src.models.components.losses.ei_euler.ei_euler_diagonal_loss`) is
    Brownian-free: it must reduce to a plain single-time velocity field. This
    helper enforces that by feeding ``degree`` *zero* martingale coefficient
    fields and by clearing the transient Brownian state for the duration of the
    call, so no padded-degree key/value mask is derived from a stale path whose
    degree width would disagree with the silenced features.

    :param net: The flow-map backbone to evaluate.
    :param x: Inputs of shape ``(B, C, H, W)``.
    :param t: Times broadcastable to ``(B,)``; used for both endpoints.
    :param degree: Number of zero coefficient fields the backbone expects.
    :return: The velocity field of shape ``(B, C, H, W)``.
    """
    times = t.reshape(-1)
    coefficients = x.new_zeros(x.shape[0], degree, *x.shape[1:])
    net.set_brownian(None)
    try:
        return net.drift(x, times, times, coefficients)
    finally:
        net.set_brownian(None)


def analytic_euler_map(
    net: SSFMNet,
    schedule: object,
    x: Tensor,
    s: Tensor,
    t: Tensor,
    brownian: BrownianPolynomial,
    coefficients: Tensor,
) -> Tensor:
    """Apply a drift-only Euler map with the schedule's closed-form diffusion.

    The network predicts only the physical drift. Additive noise is supplied
    analytically as ``sqrt(nu_squared(s)) * (W_t - W_s)`` from the same path
    whose polynomial coefficients condition the network.
    """
    if net.parameterization != 'euler':
        raise ValueError('analytic_euler_map requires net.parameterization="euler".')
    drift = net.drift(x, s, t, coefficients)
    h = (t - s).view(t.shape[0], *([1] * (x.ndim - 1)))
    nu = schedule.nu_squared(s).clamp_min(0.0).sqrt()
    nu = nu.view(s.shape[0], *([1] * (x.ndim - 1)))
    return x + h * drift + nu * brownian.raw_increment()


class SSFMNet(nn.Module, ABC):
    """Abstract base class for stochastic-flow-map networks.

    Subclasses implement :meth:`_field`, returning the raw drift field (and,
    under ``'euler_maruyama'``, a diffusion field of the same shape as ``x``).
    The base assembles the flow-map step ``Psi`` in :meth:`forward`.

    The polynomial Brownian increment ``W_{s,t}`` used by the Euler-Maruyama
    term is read from a :class:`BrownianPolynomial` the caller assigns via
    :meth:`set_brownian` before each step; it is kept as transient state (not a
    buffer/parameter, so it does not move with ``.to()`` or enter the
    ``state_dict``) and re-sampled fresh per step.

    Parameters
    ----------
    parameterization : {'euler', 'euler_maruyama'}, optional
        ``'euler'`` (default) gives ``Psi = x + (t - s) drift``.
        ``'euler_maruyama'`` gives ``Psi = x + (t - s) drift + W_{s,t} diffusion``,
        with the diffusion field also produced by the backbone.
    """

    def __init__(
        self, *, parameterization: Parameterization = 'euler', normalize_coefficients: bool = True
    ) -> None:
        super().__init__()
        if parameterization not in ('euler', 'euler_maruyama'):
            raise ValueError(
                f"parameterization must be 'euler' or 'euler_maruyama'; got {parameterization!r}."
            )
        self.parameterization = parameterization
        self.normalize_coefficients = normalize_coefficients
        self._brownian: BrownianPolynomial | None = None

    def set_brownian(self, brownian: BrownianPolynomial | None) -> None:
        """Assign the current step's polynomial Brownian path (transient state).

        :param brownian: The :class:`BrownianPolynomial` for the active ``[s, t]``
            step, or ``None`` to clear it.
        """
        self._brownian = brownian

    def get_increment(self, s: Tensor, t: Tensor) -> Tensor:
        """Return the polynomial increment ``W_{s,t}`` from the assigned path.

        The interval endpoints are detached (cf. the reference ``stop_gradient``
        on the step endpoints) so autograd through ``s``/``t`` does not cancel the
        ``Wdot g`` correction that the diagonal losses rely on.

        :param s: Start times broadcastable to the sample shape.
        :param t: Terminal times.
        :return: The increment ``W_{s,t}`` with the sample shape ``(B, C, H, W)``.
        :raises RuntimeError: If no Brownian path has been assigned.
        """
        if self._brownian is None:
            raise RuntimeError(
                'No Brownian path assigned; call set_brownian(...) before forward '
                'when using the euler_maruyama parameterization.'
            )
        # Reshape per-sample times (B,) to broadcast against the (B, C, H, W) field.
        field_ndim = self._brownian.coeffs[0].ndim
        u = s.detach().view(s.shape[0], *([1] * (field_ndim - 1)))
        v = t.detach().view(t.shape[0], *([1] * (field_ndim - 1)))
        return self._brownian.increment(u, v)

    @abstractmethod
    def _field(
        self, x: Tensor, s: Tensor, t: Tensor, coefficients: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Return the raw drift field and (for euler_maruyama) the diffusion field.

        :param x: Inputs of shape ``(B, C, H, W)``.
        :param s: Start times of shape ``(B,)``.
        :param t: Terminal times of shape ``(B,)``.
        :param coefficients: Brownian coefficient features ``(B, N, C, H, W)``.
        :return: ``(drift, diffusion)`` where ``drift`` has shape ``(B, C, H, W)``
            and ``diffusion`` is ``None`` under ``'euler'`` or a ``(B, C, H, W)``
            field otherwise.
        """
        raise NotImplementedError

    def _apply_parameterization(
        self, x: Tensor, drift: Tensor, diffusion: Tensor | None, s: Tensor, t: Tensor
    ) -> Tensor:
        """Assemble the flow-map step ``Psi`` from the raw fields.

        :param x: Inputs ``(B, C, H, W)``.
        :param drift: Drift field ``(B, C, H, W)``.
        :param diffusion: Diffusion field ``(B, C, H, W)`` (euler_maruyama) or ``None``.
        :param s: Start times ``(B,)``.
        :param t: Terminal times ``(B,)``.
        :return: The flow-map output ``Psi`` of shape ``(B, C, H, W)``.
        """
        h = (t - s).view(t.shape[0], *([1] * (x.ndim - 1)))
        psi = x + h * drift
        if self.parameterization == 'euler_maruyama':
            psi = psi + self.get_increment(s, t) * diffusion
        return psi

    def drift(self, x: Tensor, s: Tensor, t: Tensor, coefficients: Tensor) -> Tensor:
        """Return only the raw drift field (for instantaneous loss).

        :param x: Inputs ``(B, C, H, W)``.
        :param s: Start times ``(B,)``.
        :param t: Terminal times ``(B,)``.
        :param coefficients: Brownian coefficient features ``(B, N, C, H, W)``.
        :return: The drift field of shape ``(B, C, H, W)``.
        """
        drift, _ = self._field(x, s.reshape(-1), t.reshape(-1), coefficients)
        return drift

    def fields(
        self, x: Tensor, s: Tensor, t: Tensor, coefficients: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Return the raw ``(drift, diffusion)`` fields.

        ``diffusion`` is ``None`` under the ``'euler'`` parameterization and a
        ``(B, C, H, W)`` field under ``'euler_maruyama'``. Exposes both heads for
        losses that regress the diffusion field (e.g. the euler_maruyama
        instantaneous loss).

        :param x: Inputs ``(B, C, H, W)``.
        :param s: Start times ``(B,)``.
        :param t: Terminal times ``(B,)``.
        :param coefficients: Brownian coefficient features ``(B, N, C, H, W)``.
        :return: ``(drift, diffusion)``.
        """
        return self._field(x, s.reshape(-1), t.reshape(-1), coefficients)

    def forward(self, x: Tensor, s: Tensor, t: Tensor, coefficients: Tensor) -> Tensor:
        """Evaluate the flow-map step ``Psi_{s,t}`` at ``x``.

        :param x: Inputs of shape ``(B, C, H, W)``.
        :param s: Start times of shape ``(B,)`` (or broadcastable to it).
        :param t: Terminal times of shape ``(B,)``.
        :param coefficients: Polynomial Brownian coefficient features, typically
            ``(B, N, C, H, W)`` where ``N`` is the degree.
        :return: The flow-map output ``(B, C, H, W)``.
        """
        s, t = s.reshape(-1), t.reshape(-1)
        drift, diffusion = self._field(x, s, t, coefficients)
        return self._apply_parameterization(x, drift, diffusion, s, t)


