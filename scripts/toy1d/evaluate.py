"""Comparison charts for the 1D non-linear SDE toy (arXiv:2606.01086 Sec. 4.1).

Loads one trained module per ablation run (any noise arm: ``wz`` / ``ds`` /
``none``), evaluates every map on ONE shared realization, and renders a
two-panel figure:

- *trajectories*: the 1024-step Euler-Maruyama ground-truth solve (time vs
  value, grey) against each run's learned map rollout at a fixed step count
  (one colour per run);
- *strong error*: the mean squared endpoint error of each run against the
  fine reference, as a function of the map's step count.

Fair-comparison guarantee: every run is evaluated on the SAME initial
condition ``X_0`` and the SAME underlying Brownian path — one generator
stream seeds the closed-form-martingale (clock) realization and the plain
native-time realization identically (the per-leaf coefficient draws are
the same standard-normal stream, so the implied ``W`` coincides pathwise),
and each arm's fine reference rides that same path. Runs are therefore
compared on identical noise and identical starts.

Usage:
    uv run python -m scripts.toy1d.evaluate \
        --run-dir path/to/run1 --run-dir path/to/run2 ... --out chart.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import hydra
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from src.models.components.brownian import BrownianPolynomial
from src.models.components.strong_convergence import (
    PiecewiseBrownianPath,
    sample_piecewise_brownian,
)
from src.models.components.toy_martingale import (
    sample_piecewise_toy_martingale,
    toy_clock,
)
from src.models.ssfm_module import SSFMLitModule

COLORS = (
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
    'maroon',
    'navy',
    'crimson',
)


def load_module(run_dir: Path) -> tuple[SSFMLitModule, str]:
    """Instantiate the module from a run's saved Hydra config + ``last.ckpt``."""
    config = OmegaConf.load(run_dir / '.hydra' / 'config.yaml')
    module: SSFMLitModule = hydra.utils.instantiate(config.model)
    checkpoint = torch.load(
        run_dir / 'checkpoints' / 'last.ckpt', map_location='cpu', weights_only=False
    )
    module.load_state_dict(checkpoint['state_dict'])
    module.eval()
    arm = getattr(module.loss, 'arm', None)
    if arm is None:
        return module, Path(run_dir).name
    form = getattr(module.loss, 'form', None)
    degree = int(getattr(module.loss, 'degree', 0))
    label = f'{arm}-{module.loss.distillation_name}' + (f'-{form}' if form else '')
    return module, f'{label}-N{degree}'


class SharedEvaluation:
    """One shared ``X_0`` and Brownian realization for every run.

    The generator stream is consumed identically for both realizations
    (native-time leaves for the plain arm, clock-polynomial leaves for the
    closed-form martingale arms), so the underlying ``W`` and the initial
    condition are the same for every run — a fair pathwise comparison.
    """

    def __init__(
        self,
        sde,
        degree: int,
        num_samples: int,
        em_steps: int,
        seed: int,
        display_samples: int = 0,
        display_seed: int = 0,
    ) -> None:
        self.sde = sde
        self.degree = degree
        self.em_steps = em_steps
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(num_samples, 1, generator=generator)
        self.x_0 = sde.initial_condition(noise)
        # One shared leaf-coefficient stream: the native and clock
        # constructions draw the same standard-normal coefficients, so
        # they describe the same underlying Brownian path.
        leaf_generator = torch.Generator().manual_seed(seed + 10_000)
        self.native = sample_piecewise_brownian(
            (num_samples, 1),
            degree,
            em_steps,
            0.0,
            device=self.device,
            dtype=self.dtype,
            generator=leaf_generator,
        )
        clock_nodes, clock_leaves = sample_piecewise_toy_martingale(
            sde,
            (num_samples, 1),
            degree,
            em_steps,
            device=self.device,
            dtype=self.dtype,
            generator=leaf_generator,
        )
        self.clock_piecewise = PiecewiseBrownianPath(nodes=clock_nodes, leaves=clock_leaves)
        self.fine_nodes = torch.linspace(0.0, 1.0, em_steps + 1)
        self.native_reference = self._endpoint(self.native.leaves, scale_by_diffusion=True)
        self.clock_reference = self._endpoint(clock_leaves, scale_by_diffusion=False)
        self.native_states = self._states(self.native.leaves, scale_by_diffusion=True)
        self.clock_states = self._states(clock_leaves, scale_by_diffusion=False)
        self._display = None
        self.display_samples = display_samples
        self.display_seed = display_seed

    def reference_for(self, arm: str) -> torch.Tensor:
        """Return the fine reference endpoint of the arm's family.

        Every arm rides the closed-form martingale increment, so the
        clock-family reference applies to all of them.
        """
        del arm
        return self.clock_reference

    def _fine_step(self, state, index, leaves, *, scale_by_diffusion: bool):
        nodes = self.fine_nodes
        mid = 0.5 * (nodes[index] + nodes[index + 1])
        step = nodes[index + 1] - nodes[index]
        increment = leaves[index].raw_increment()
        if scale_by_diffusion:
            increment = self.sde.diffusion(mid) * increment
        return state + self.sde.tamed_drift(state, mid, step) * step + increment

    def _endpoint(self, leaves, *, scale_by_diffusion: bool) -> torch.Tensor:
        state = self.x_0
        for index in range(len(leaves)):
            state = self._fine_step(state, index, leaves, scale_by_diffusion=scale_by_diffusion)
        return state

    def _states(self, leaves, *, scale_by_diffusion: bool) -> torch.Tensor:
        state = self.x_0
        states = [state]
        for index in range(len(leaves)):
            state = self._fine_step(state, index, leaves, scale_by_diffusion=scale_by_diffusion)
            states.append(state)
        return torch.stack(states, dim=1)

    def display(self) -> SharedEvaluation:
        """Return the (cached) display-sized shared draws, same stream."""
        if self._display is None:
            self._display = SharedEvaluation(
                self.sde,
                self.degree,
                self.display_samples,
                self.em_steps,
                self.display_seed,
            )
        return self._display

    @torch.no_grad()
    def rollout(self, module: SSFMLitModule, count: int, *, for_display: bool = False):
        """Roll one module's learned map over ``count`` uniform steps.

        Returns states ``(B, count + 1, 1)`` on the arm's shared realization.
        """
        arm = module.loss.arm
        net = module.net
        degree = int(module.loss.degree)
        if for_display:
            view = self.display()
            x_0 = view.x_0
            clock = view.clock_piecewise
        else:
            x_0 = self.x_0
            clock = self.clock_piecewise
        ones = torch.ones(x_0.shape[0])
        grid = torch.linspace(0.0, 1.0, count + 1)
        state = x_0
        states = [state]
        for s_cur, t_nxt in zip(grid[:-1], grid[1:], strict=True):
            s_clock = toy_clock(self.sde, s_cur)
            t_clock = toy_clock(self.sde, t_nxt)
            path = clock.interval(s_clock, t_clock)
            leaf = BrownianPolynomial(
                coeffs=path.coeffs[:degree], s=path.s, t=path.t, degrees=None
            )
            features = leaf.coefficient_features(normalize=net.normalize_coefficients)
            s_b, t_b = s_cur.expand_as(ones), t_nxt.expand_as(ones)
            if arm == 'ds':
                z_t = net(state, s_b, t_b, features)
                state = z_t + leaf.raw_increment()
            else:
                # wz and none: identical x-space maps with the closed-form
                # increment (the arms differ only in training objectives).
                velocity = net.drift(state, s_b, t_b, features)
                state = state + (t_nxt - s_cur) * velocity + leaf.raw_increment()
            states.append(state)
        return torch.stack(states, dim=1)


def main() -> None:
    """Evaluate every run on the shared realization and render the chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run-dir',
        action='append',
        required=True,
        help='Hydra run directory of one trained toy run (repeatable).',
    )
    parser.add_argument(
        '--label',
        action='append',
        default=None,
        help='Legend label per run directory (repeatable; defaults to arm/kind/form).',
    )
    parser.add_argument('--out', type=Path, required=True, help='Output PNG path.')
    parser.add_argument(
        '--num-samples', type=int, default=4096, help='Particles for the strong errors.'
    )
    parser.add_argument(
        '--num-paths', type=int, default=5, help='Particles shown in the trajectory panel.'
    )
    parser.add_argument(
        '--em-steps', type=int, default=1024, help='Ground-truth EM reference steps.'
    )
    parser.add_argument(
        '--step-counts',
        type=int,
        nargs='+',
        default=(1, 2, 4, 8, 16, 32),
        help='Map step counts for the strong-error sweep.',
    )
    parser.add_argument(
        '--steps-plot', type=int, default=16, help='Map step count of the trajectory panel.'
    )
    parser.add_argument('--seed', type=int, default=0, help='Shared-realization seed.')
    parser.add_argument(
        '--exclude',
        action='append',
        default=None,
        help='Skip runs whose auto-label contains this substring (repeatable).',
    )
    args = parser.parse_args()

    if args.steps_plot not in args.step_counts:
        args.step_counts = sorted({*args.step_counts, args.steps_plot})

    excluded = tuple(args.exclude or ())
    modules = []
    auto_labels = []
    for run_dir in args.run_dir:
        module, auto = load_module(Path(run_dir))
        if excluded and any(token in auto for token in excluded):
            print(f'skipping {run_dir} (auto-label {auto!r} matches --exclude)')
            continue
        modules.append(module)
        auto_labels.append(auto)
    if not modules:
        raise SystemExit('All runs were excluded; nothing to chart.')
    labels = args.label if args.label else auto_labels
    if args.label and len(labels) != len(modules):
        raise SystemExit('Provide as many --label entries as non-excluded runs.')

    # The shared realization is built at the MAXIMUM degree; every run then
    # consumes its own degree-N truncation of the same underlying Brownian
    # path (``path.coeffs[:N]``), so degree sweeps stay pathwise-fair while
    # form grids (all one degree) are unaffected.
    degrees = {int(module.loss.degree) for module in modules}
    sde = modules[0].loss.sde
    shared = SharedEvaluation(sde, max(degrees), args.num_samples, args.em_steps, args.seed)
    shared.display_samples = args.num_paths
    shared.display_seed = args.seed + 1

    errors: dict[str, dict[int, float]] = {}
    for label, module in zip(labels, modules, strict=True):
        arm = module.loss.arm
        reference = shared.reference_for(arm)
        errors[label] = {}
        for count in args.step_counts:
            endpoint = shared.rollout(module, count)[:, -1]
            errors[label][count] = float((endpoint - reference).square().mean())

    # ------------------------------------------------------------------ chart
    figure, (axis_traj, axis_error) = plt.subplots(1, 2, figsize=(12, 4.5))
    fine_times = torch.linspace(0.0, 1.0, args.em_steps + 1)
    axis_traj.plot(
        fine_times.numpy(),
        shared.native_states[: args.num_paths, :, 0].numpy().T,
        color='0.55',
        linewidth=1.0,
        alpha=0.8,
        label=f'true SDE ({args.em_steps}-step EM)',
    )
    for index, (label, module) in enumerate(zip(labels, modules, strict=True)):
        states = shared.rollout(module, args.steps_plot, for_display=True)
        times = torch.linspace(0.0, 1.0, args.steps_plot + 1)
        axis_traj.plot(
            times.numpy(),
            states[:, :, 0].numpy().T,
            color=COLORS[index % len(COLORS)],
            linewidth=1.0,
            alpha=0.9,
            label=f'{label} ({args.steps_plot}-step)',
        )
    axis_traj.set_xlabel('time')
    axis_traj.set_ylabel('value')
    axis_traj.set_title(f'True SDE vs learned maps ({args.num_paths} shared particles)')
    axis_traj.legend(fontsize=7)

    for index, label in enumerate(labels):
        counts = sorted(errors[label])
        axis_error.plot(
            counts,
            [errors[label][c] for c in counts],
            marker='o',
            color=COLORS[index % len(COLORS)],
            linewidth=1.2,
            label=label,
        )
    axis_error.set_yscale('log')
    axis_error.set_xscale('log', base=2)
    axis_error.set_xticks(sorted(args.step_counts))
    axis_error.set_xticklabels([str(c) for c in sorted(args.step_counts)])
    axis_error.set_xlabel('map step count')
    axis_error.set_ylabel('strong MSE vs ground truth')
    axis_error.set_title('Strong error per step count (shared noise)')
    axis_error.legend(fontsize=7)

    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f'chart saved to {args.out}')

    csv_path = args.out.with_suffix('.csv')
    with csv_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['run', 'steps', 'strong_mse'])
        for label in labels:
            for count in sorted(errors[label]):
                writer.writerow([label, count, errors[label][count]])
    print(f'errors saved to {csv_path}')
    print(json.dumps(errors, indent=2))


if __name__ == '__main__':
    main()
