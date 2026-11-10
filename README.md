# Rigid Body Path Planning using Mixed-Integer Linear Programming

Reference implementation of

> **Rigid Body Path Planning Using Mixed-Integer Linear Programming**
> Mingxin Yu and Chuchu Fan
> *IEEE Robotics and Automation Letters*, vol. 9, no. 11, pp. 10026–10033, Nov. 2024
> [IEEE Xplore](https://ieeexplore.ieee.org/document/10694695) · [doi:10.1109/LRA.2024.3468165](https://doi.org/10.1109/LRA.2024.3468165) · [arXiv:2409.11520](https://arxiv.org/abs/2409.11520)

Moving a rigid body through a cluttered space with narrow passages is hard for
both mixed-integer and sampling-based planners: the former because a monolithic
MILP over the whole workspace has too many binary variables, the latter because
narrow passages are sampled rarely. This repository takes a middle path in three
stages.

1. **Decompose.** Grow a graph of collision-free convex polytopes covering the
   free workspace, seeded from a visibility graph (`algorithm/pre_processing.py`,
   `algorithm/constrained_iris.py`).
2. **Certify, offline.** For every pair of neighbouring polytopes, solve a
   *small* MILP that is restricted to two or three regions at a time, and record
   the feasible transitions as edges of a dense graph
   (`algorithm/verify_edges.py`, `milp_model/`).
3. **Query, online.** Answer a start/goal pair by connecting both endpoints to
   the dense graph and running a shortest-path search, so no MILP is solved at
   query time (`algorithm/grow_roadmap.py:inference_with_rdmap`).

Because each MILP sees at most three polytopes, the number of binary variables
per problem stays constant as the environment grows.

## Demos

| | |
|---|---|
| [`media/3d-demo.mp4`](media/3d-demo.mp4) | 3D rigid body through a room |
| [`media/dense.mp4`](media/dense.mp4) | the `dense-3D` cluttered-room environment |
| [`media/peg3d.mp4`](media/peg3d.mp4) | the `peg-3D` peg-in-hole environment |

## Repository layout

```
algorithm/        the method itself
  pre_processing.py     visibility graph, coarse polytope roadmap, free patches
  constrained_iris.py   IRIS variant that grows polytopes from seed ellipsoids
  verify_edges.py       builds and solves the batch of small offline MILPs
  grow_roadmap.py       offline roadmap construction + online query
milp_model/       Gurobi MILP formulations
  basic_wp_milp.py          abstract waypoint MILP
  inside_sampleEdge_milp.py the formulation used in the paper
environment/      environment back-ends (see "Optional back-ends" below)
  basic_env.py              geometry, obstacles, H/V representations
  clean_env.py              the only back-end the paper's method needs
  predefined_2Denv.py       2D benchmark environments
  predefined_3Denv.py       3D benchmark environments
baseline/         baseline planners: GCS, OMPL, Klampt, a plain PRM
utils/            geometry helpers, plotting, discretisation constants
src/              YAML environment definitions, Drake URDF generators
figures/          scripts that produce the figures in the paper
media/            demo videos
result/           output directory (figures and cached regions are written here)
eval_2D.py        entry point for the 2D experiments
eval_3D.py        entry point for the 3D experiments
```

## Installation

Python 3.9 is what this was developed against.

```bash
git clone https://github.com/MIT-REALM/rigid-body-milp-planning.git
cd rigid-body-milp-planning
conda env create -f environment.yml && conda activate mip-mp
```

or, into an environment you already have:

```bash
pip install -r requirements.txt
```

> **macOS note.** With pip-installed wheels, `torch` (pulled in by `lightning`)
> and `open3d` each bundle their own copy of `libomp.dylib`, and loading
> `torch`'s first aborts the process with `OMP: Error #179`. `eval_2D.py` and
> `figures/teaser.py` therefore import `open3d` on line 10, before
> `pytorch_lightning`; do not move or remove that line. See *Known rough edges*.

### Solver licences

Two solvers are required by the core method and both are commercial, with free
academic licences:

- **Gurobi** solves every MILP — [academic licence](https://www.gurobi.com/academia/academic-program-and-licenses/)
- **MOSEK** is the SOCP backend cvxpy uses for the inscribed-ellipsoid step in
  `constrained_iris.py` — [academic licence](https://www.mosek.com/products/academic-licenses/)

Point each at its licence file the usual way (`GRB_LICENSE_FILE`,
`MOSEKLM_LICENSE_FILE`). Note that `environment/drake_env.py:25` overwrites
`MOSEKLM_LICENSE_FILE` with a hard-coded relative path — see *Known rough
edges*.

MOSEK is swappable. `algorithm/constrained_iris.py:173` hard-codes
`prob.solve(solver=cp.MOSEK)`, and line 172 is a commented-out SCS call.
Exchange the two lines to run the inscribed-ellipsoid SOCP on SCS instead, which
cvxpy pulls in as a dependency and which needs no licence. The timings in the
paper were measured with MOSEK. Gurobi has no equivalent fallback: the MILPs are
written directly against `gurobipy`.

### Optional back-ends

Nothing below is needed to run the paper's method. Install only the baselines
you want to compare against:

```bash
pip install -r requirements-baselines.txt
```

`environment/__init__.py` and `baseline/__init__.py` import every optional
back-end defensively, so a partial install is fine. A back-end whose
dependencies are missing is still bound to its name — `from environment import
DrakeEnvironment` keeps working — and raises `ImportError` with an install hint
only if you actually select that method. To see what is available in your
interpreter:

```bash
python -c "import environment, baseline; print(environment.AVAILABLE); print(baseline.AVAILABLE)"
```

Two back-ends are not installable from PyPI:

- **OMPL** must be built with its Python bindings —
  [instructions](https://ompl.kavrakilab.org/installation.html).
- **PythonCDT** is needed by the Klampt WCO baseline and must be built from
  source. It is included as a submodule:
  ```bash
  git submodule update --init --recursive
  pip install ./dependencies/PythonCDT
  ```

## Running the experiments

Both entry points are configured by editing the `__main__` block at the bottom
of the script rather than by command-line flags. The lines to change:

### 2D — `eval_2D.py`

| Line | Setting | Values |
|---|---|---|
| 76 | `seed` | any int |
| 78–80 | `mode` | `"timing"` (report offline/online times), `"vis_env"` (save environment figure), `"teaser"` |
| 88 | `env_name` | indexes `env_list` on lines 82–87 |
| 100 | `method_list` | one or more method names, see below |
| 103–126 | agent | uncomment one `agent_name` / `agent` block (`"stick"` or `"L"`) |

```bash
python eval_2D.py
```

2D environments: `corner2D`, `bugtrap2D`, `chull-2D`, `infeas1-2D`, `STL2-2D`,
`STL3-2D`, plus the YAML scenes `src/simple10.yaml` and
`src/challenging600.yaml`.

Methods: `ciris_ours` (this paper), `single_milp` (monolithic MILP),
`gcs`, `ompl-rrt`, `ompl-rrt_star`, `ompl-prm`, `klampt-prm`, `klampt-wco`,
`custom-prm`.

> **Note.** Line 100 is set to `method_list = ['ciris_ours']`, the method from
> this paper, which needs only the core dependencies. The baseline alternatives
> are on the commented lines immediately around it; each needs its own optional
> back-end installed (see *Optional back-ends*). More than one entry in the list
> runs them side by side in a shared figure.

### 3D — `eval_3D.py`

| Line | Setting | Values |
|---|---|---|
| 34 | `seed` | any int |
| 37–38 | `mode` | `"timing"` or `"vis_env"` |
| 41 | `env_name` | indexes `env_list` on line 40 |
| 47–51 | `method` | a single method name |
| 53–57 | agent | uncomment one `agent_name` / `agent` block (`'stick'` or `'L'`) |

```bash
python eval_3D.py
```

3D environments: `easy-3D`, `dense-3D`, `peg-3D`, `demo-3D`.

`eval_3D.py` renders offscreen with Open3D and writes a video of the resulting
path to `peg3d.mp4` in the working directory.

### Figures

The scripts in `figures/` reproduce figures from the paper. They differ in the
directory they expect to be run from: `teaser.py` and `prm_scaling.py` import
from the repository root, while `vis_2d_cfree.py` reads `../result/...` and its
cached `sample_mask_*.npy` files from the current directory.

```bash
python figures/teaser.py            # from the repository root
cd figures && python vis_2d_cfree.py   # from inside figures/
```

## Known rough edges

This is research code released as it was used for the paper. The algorithm files
are unmodified; the following are real sharp edges worth knowing about before
you build on it.

- **Two OpenMP runtimes on macOS.** `algorithm/grow_roadmap.py` imports
  `lightning` (line 8, hence torch) and then `utils.color_config` (line 18,
  hence open3d). Both torch and open3d ship their own `libomp.dylib` inside
  their wheels, and loaded in that order the process dies with
  `OMP: Error #179: Function pthread_mutex_init failed`. Nothing in the planner
  is at fault and `KMP_DUPLICATE_LIB_OK` does not help. The entry points work
  around it by importing `open3d` first (line 10 of `eval_2D.py` and of
  `figures/teaser.py`; `eval_3D.py` already imported it early). Conda builds of
  torch link the environment's shared `libomp` and are unaffected, so the
  workaround is a no-op there.
- **Discretisation constants are shadowed.** `algorithm/pre_processing.py:13`
  does `from utils.config import *`, which sets `NUM_THETA_DIV = 60` and
  `NUM_BOUNDARY_LINE_DIV = 161`, and then lines 17–18 immediately redefine them
  to `40` and `180`. Inside `pre_processing.py` the local values win, so editing
  `utils/config.py` does **not** change boundary discretisation there.
- **`ciris_ours` caches its roadmap between runs.** The first run for a given
  agent and environment builds the polytope roadmap and pickles it to
  `result/{agent}-env_vis/{env}.pkl`; later runs reuse it and report an offline
  time of 0, because no offline work happens. Delete that file to force a
  rebuild and get a real offline timing.
- **`single_milp` does not run.** The monolithic-MILP comparison in `eval_2D.py`
  (lines 163–172) calls `environment.sample_free(...)` and
  `environment.agent_dense`, neither of which exists on any environment class;
  it also reads an undefined `start_time` and calls `ax.title(...)` as if it were
  a method. Use the numbers reported in the paper for this baseline.
- **Offline MILP solving needs more than 8 cores.**
  `algorithm/verify_edges.py:117` sizes its pool as
  `min((mp.cpu_count() - 8) // 2, len(MILP_inputs))`, which is zero or negative
  on machines with ≤ 8 cores and will fail to start the pool. Adjust for your
  machine.
- **The GCS baseline runs in two passes.** `baseline/gcs/gcs_soln.py` computes
  the C-IRIS regions, pickles them to `result/{agent}-gcs/{env}.pkl`, and then
  calls `exit()` on line 27. The second invocation loads the cache and plans. The
  cached `.pkl` files are not shipped in this repository, so the first run of any
  GCS configuration recomputes them (Drake plus a MOSEK licence required).
- **Hard-coded MOSEK path.** `environment/drake_env.py:25` sets
  `MOSEKLM_LICENSE_FILE` to `environment/../../mosek/mosek.lic`, i.e. a `mosek/`
  directory one level *above* the repository. Edit that line or place your
  licence accordingly.
- **`grow_roadmap.py` exits on failure.** When no path exists in the dense graph,
  `inference_with_rdmap` plots the graph and calls `exit()` (line 158) instead of
  raising, which makes it awkward to use as a library.
- **`solve_growing_MILP` never grows.** In `algorithm/verify_edges.py` the
  function returns or returns `None` before reaching `add_waypoint()` (lines
  137–139), and `MAX_NUM_WAYPOINT` is `1`. Each small MILP is attempted once with
  zero intermediate waypoints, which is what the paper reports; the
  waypoint-growing loop is vestigial.
- **`cdd` fast path is dead.** `utils/geometry_operation.py:get_H_rep_from_V`
  wraps a `cdd`-based conversion in `try`, but the `cdd` import is commented out
  at line 2, so the call always raises `NameError` and falls through to the
  `scipy.spatial.ConvexHull` path. Functionally fine, but the `try/except` is
  misleading.
- **`tri_ours` is not implemented.** It appears in `all_method_list` in
  `eval_2D.py` but trips an assertion; only `ciris_ours` is wired up.
- **Both `lightning` and `pytorch-lightning` are required**, for
  `SimpleProfiler` and `seed_everything` respectively. Neither deep-learning
  framework is otherwise used.
- `baseline/deprecated/` is still imported (`klampt_soln` by
  `baseline/__init__.py`, `milp_sep_dim` by `eval_2D.py`) despite the directory
  name.

## Citation

```bibtex
@article{yu2024rigid,
  author  = {Yu, Mingxin and Fan, Chuchu},
  title   = {Rigid Body Path Planning Using Mixed-Integer Linear Programming},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {9},
  number  = {11},
  pages   = {10026--10033},
  year    = {2024},
  doi     = {10.1109/LRA.2024.3468165}
}
```

## License

MIT — see [LICENSE](LICENSE).

Parts of `baseline/` are derived from other open-source projects and remain under
their original licences; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
