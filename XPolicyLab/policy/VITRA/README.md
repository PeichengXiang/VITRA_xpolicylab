# VITRA

**Contributor:** xiangpc | **Paper:** VITRA | **arXiv:** [2510.21571](https://arxiv.org/abs/2510.21571) | **Original code:** [microsoft/VITRA](https://github.com/microsoft/VITRA)

This adapter integrates the official VITRA implementation into XPolicyLab while keeping the upstream model architecture and prediction semantics intact. The nested `VITRA/` directory is the upstream checkout. Local upstream changes are intentionally small and auditable: the Spark0 dataset bridge, local-config PaliGemma construction, non-persistent W&B authentication, complete/atomic FSDP checkpoint saving, an equivalent pinned headless OpenCV dependency compatible with VITRA's NumPy constraint, and moving the unsupported `decord` human-video dependency out of the robot-training base environment.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Action semantics

Invoke XPolicyLab with `action_type=ee`. At the XPolicyLab boundary every action step is absolute and has exactly these keys:

- `left_ee_pose`, `right_ee_pose`: `[x, y, z, qw, qx, qy, qz]`
- `left_ee_joint_state`, `right_ee_joint_state`: 20 absolute Wuji joint targets

Inside VITRA, each hand uses an absolute camera-frame current wrist pose and absolute hand state. Its predicted wrist action is a one-step translation/rotation delta, while its hand-joint prediction is the next absolute target. The shared Spark0 bridge accumulates the wrist deltas and converts the resulting poses back to XPolicyLab's environment frame. Rotation uses matrix composition (`R_next = R_delta @ R_current`), not Euler-angle subtraction/addition.

For EgoVLA `inspire12`, robot fine-tuning follows the hybrid contract used by
VITRA's robot path: the wrist label is the realized transition from recorded
`EEF[t]` to recorded `EEF[t+1]`, and the hand label is the direct future
execution command in `action[t]`. Controller `target_ee_pose` fields do not
define wrist labels. The final row of every episode is masked because it has no
recorded `EEF[t+1]`. The manifest and statistics both record
`egovla_observed_ee_step_future_hand_command_v3`, and training refuses older or
mismatched EgoVLA statistics.

The Spark0 HDF5 EE pose is the environment-frame pose of `Link7_L`/`Link7_R`, not the physical hand wrist. The 20 hand values are absolute Wuji Hand2 revolute-joint coordinates in radians, in the dataset's Isaac stage-major order: `[index,middle,pinky,ring,thumb]` for flex, then abduction, PIP/thumb-MCP, and DIP/thumb-IP.

Two data representations are supported. `wuji20` is the legacy 52-D bridge and retains the sparse, uncalibrated 20-to-MANO45 injection in `mapping_wuji20_mano45.json`. `mano45` consumes the dense MANO labels produced by Spark-0 `add_mano.py`: each hand has camera-frame root pose plus 15 local MANO rotations. At runtime it runs the same Wuji FK/MANO fitter for observations and uses an explicitly selected inverse for actions; it does not use the sparse mapping or old Wuji statistics. Historical checkpoints default to the bounded geometric inverse. An optional hash-locked, side-specific affine inverse maps MANO local rotvec45 (encoded as per-joint XYZ Euler angles) directly to Wuji q20 in stage-major order. Its accepted output is never IK-refined; a safety rejection retries the historical geometric inverse, and only failure of that fallback holds both hands at the last complete safe Link7/q command for the remainder of the chunk.

## Installation

Use a Python 3.10/3.11 CUDA environment. The official VITRA package pins its tested Torch/Transformers versions:

```bash
cd XPolicyLab/policy/VITRA
bash install.sh
```

Place the released full VITRA checkpoint under the workspace's `pretrain_model/VITRA-VLA-3B/`. The integration constructs the PaliGemma 2 architecture from local processor/config assets in `pretrain_model/paligemma2-3b-mix-224-local/`, then strictly loads the full VITRA state dict; it does not download or initialize from a second 3B base-weight file.

## Data processing

The converter creates a local thin view of the immutable Spark0 source episodes and computes representation-specific normalization statistics. The legacy form is:

```bash
bash process_data.sh spark0_bench cotrain tianji_marvin_wuji ee
```

For dense MANO labels, explicitly select `mano45` and point at the HDF5 tree containing `mano/state`, `mano/action`, and validity fields:

```bash
VITRA_DATA_REPRESENTATION=mano45 \
SPARK0_SOURCE_DATA=/absolute/path/to/spark0_bench_mano \
bash process_data.sh spark0_bench_mano cotrain tianji_marvin_wuji ee
```

The MANO statistics are per hand state61/action51 and are not interchangeable with the legacy Wuji state26/action26 statistics.

From the workspace root (the directory containing `XPolicyLab/` and
`data_scripts/`), prepare the EgoVLA H1 Inspire12 view and its matching
statistics with:

```bash
bash data_scripts/prepare_egovla_inspire12_robot_command_v3.sh \
  /absolute/path/to/EgoVLA_raw_remove_deprecated
```

Images are decoded only by the offline converter/loader and remain RGB. During evaluation, XPolicyLab's policy server supplies already-decoded RGB arrays; `model.py` deliberately rejects encoded bytes rather than decoding them again.

### EgoVLA H1 camera calibration

The common EgoVLA simulator bridge currently emits the decoded `cam_head` RGB
image but not its calibration. For the exact `ego_h1_inspire` simulator,
`model.py` therefore loads the hash-locked
`artifacts/ego_h1_inspire_main_camera_v1.json` profile. It is used only when
both `cam_head` intrinsics and extrinsics are absent; supplied metadata is
validated and preserved, while partial or invalid metadata fails closed.

The profile records the official 1280×720 main-camera pose and pinhole matrix,
with `K_out = diag(width/1280, height/720, 1) @ K_source` for the explicitly
allowed 1280×720 and 384×384 runtime images. The pose is `camera_to_env` in
Spark's `x_right_y_up_z_back` axes and `[x,y,z,qw,qx,qy,qz]` order. No identity
extrinsic or guessed FOV is synthesized. Override the artifact path only with
the matching `h1_camera_calibration_sha256` value. The checked-in
`h1_camera_calibration_scope: simulator` is intentionally guarded by
`EVAL_ENV_TYPE`: unset/`sim`/`debug` permit the official simulator fallback,
whereas `real`/`real_world` always require both runtime intrinsics and
extrinsics and fail closed. Set the scope to `runtime` to require metadata in
any environment; a real robot must never reuse this simulator profile.

## Training

Training follows the standard XPolicyLab six-argument interface. For the requested 8-GPU run, pass all visible devices and export the W&B credential only in the process environment:

```bash
export WANDB_API_KEY='<key>'
bash train.sh spark0_bench cotrain tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
```

For MANO training, set `VITRA_DATA_REPRESENTATION=mano45` and the prepared MANO data root; do not resume a Wuji optimizer/run as if it were the same representation.

If a long run is interrupted after a complete interval checkpoint, resume the latest complete checkpoint explicitly:

```bash
VITRA_RESUME=1 bash train.sh spark0_bench cotrain tianji_marvin_wuji ee 42 0,1,2,3,4,5,6,7
```

Without `VITRA_RESUME=1`, the launcher requires a completely empty run root. Resume mode accepts only a contiguous sequence of complete 10k checkpoints and refuses partial, epoch-end, duplicated, or unexpected checkpoint paths; if step 80k already exists it verifies the run instead of training again.

The integration config uses batch size 8 per GPU on eight GPUs (global batch size 64, gradient accumulation 1), runs 80,000 optimizer/global steps, and saves at 10,000-step intervals. Successful completion therefore yields exactly these eight checkpoint directories:

```text
epoch=E-step=10000.ckpt
epoch=E-step=20000.ckpt
...
epoch=E-step=80000.ckpt
```

Each checkpoint contains `weights.pt`, `optimizer.pt`, and `meta.json`. Checkpoints live beneath the standard run root. The concrete run directory always starts with the ISO launch date; a resumed run reuses its original date:

```text
policy/VITRA/checkpoints/<bench>-<ckpt>-<env>-ee-<seed>/<YYYY-MM-DD>-<bench>-<ckpt>_TB64_B8_bf16True/checkpoints/
```

## Evaluation

`eval.sh` uses the standard ten positional arguments:

```bash
bash eval.sh \
  <bench_name> <task_name> <ckpt_name> tianji_marvin_wuji ee <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```

The adapter resolves the standard checkpoint run root with XPolicyLab's shared resolver, then recursively selects the checkpoint with the greatest numeric `step`. For a released checkpoint or a non-standard location, set paths without modifying `deploy.yml`:

```bash
VITRA_MODEL_PATH=/absolute/path/to/weights.pt \
VITRA_CONFIG_PATH=/absolute/path/to/config.json \
VITRA_STATISTICS_PATH=/absolute/path/to/teledata_statistics.json \
bash eval.sh spark0_bench stack_bowls released tianji_marvin_wuji ee 0 0 0 vitra eval-env
```

The checkpoint config owns `data_representation`, statistics path, and statistics SHA. A MANO checkpoint additionally requires `VITRA_MANO_TOOLS_ROOT` pointing at the Spark-0 checkout used by `add_mano.py`; supplying `VITRA_MAPPING_PATH` for a MANO checkpoint is rejected. The historical geometric inverse and the linear mode's geometric safety fallback run with one BLAS thread, at most 12 numerical iterations per hand, and a post-solve latency gate so a bad prediction cannot recreate the previous unbounded evaluator stall.

### MANO inverse A/B mode

`mano_inverse_mode` is deliberately explicit. A run with no saved field remains `geometric`, preserving the original `VITRA_0823_mano` contract. Non-empty deployment fields override saved run fields; otherwise the saved config is used. `linear` requires an artifact path and its exact SHA-256, so merely checking the bundled artifact into the tree cannot change an old evaluation.

For a deployment-only A/B variant, without editing the historical checkpoint config:

```bash
VITRA_MANO_INVERSE_MODE=linear \
VITRA_MANO_LINEAR_INVERSE_PATH=artifacts/mano45_to_wuji20_linear_v1.json \
VITRA_MANO_LINEAR_INVERSE_SHA256=f196701addfcddc7bfac70a6caf722b237a398af104aa6a15993a60ccea93940 \
bash eval.sh spark0_bench stack_bowls cotrain tianji_marvin_wuji ee 42 0 0 vitra eval-env
```

A standalone saved-config variant uses the same three keys. Use an absolute artifact path, or make a relative path relative to that saved `config.json` directory. Startup provenance prints `inverse_mode`, the artifact's exact path/SHA/representation/split, and cumulative `accepted`/`fallback` counters by side. To avoid evaluator I/O overhead, accepted-frame OOD/cycle/velocity/URDF/geometry details and cumulative counters are logged on the first acceptance and every 1,000 acceptances per side; every rejection still logs its geometric-fallback reason.

The bundled artifact is reproducible from the paired MANO/Wuji HDF5 files with episode-disjoint splits: 0--79 fit, 80--89 safety calibration, and 90--99 held-out test.

```bash
python scripts/fit_mano45_linear_inverse.py \
  --dataset-root /absolute/path/to/spark0_bench_mano \
  --output /tmp/mano45_to_wuji20_linear_v1.json
sha256sum /tmp/mano45_to_wuji20_linear_v1.json
```

Its full held-out q20 elementwise MAE after the runtime URDF clip is 0.00602 rad left and 0.00748 rad right. The complete pre-geometry gate accepts 14,811/14,816 (99.966%) left and 14,776/14,816 (99.730%) right. The exact-runtime audit in `artifacts/mano45_to_wuji20_linear_v1.runtime_validation.json` additionally constructed the deployment Wuji URDF and geometry model on 1,879 stride-8 held-out frames per side: full linear acceptance was 100.00% left and 99.73% right, maximum geometry RMS was 10.64 mm and 11.57 mm respectively against the 15 mm limit. Final Link7 rotation error averaged 0.096 degrees left and 0.126 degrees right (p95 0.350/0.428 degrees). Re-run that audit against the intended Spark-0 checkout before using a different URDF:

```bash
python scripts/validate_mano45_linear_inverse_runtime.py \
  --dataset-root /absolute/path/to/spark0_bench_mano \
  --spark-root /absolute/path/to/Spark-0 \
  --artifact artifacts/mano45_to_wuji20_linear_v1.json \
  --artifact-sha256 f196701addfcddc7bfac70a6caf722b237a398af104aa6a15993a60ccea93940 \
  --test-ids 90-99 --frame-stride 8
```

`deploy.yml` also exposes `num_ddim_steps`, `cfg_scale`, `sample_times`, `execute_action_chunk`, and the hash-locked H1 camera calibration path/SHA. `default_fov` remains a fallback only for legacy non-H1 observations that already carry no geometric camera conversion requirement.
For a quicker protocol-only debug run, `VITRA_NUM_DDIM_STEPS` and `VITRA_EXECUTE_ACTION_CHUNK` can override the two corresponding server settings without editing the checked-in deployment config.

For an offline protocol/shape smoke test, set `EVAL_ENV_TYPE=debug`. The full model and checkpoint are still loaded because this is an integration test, not a zero-action mock:

```bash
EVAL_ENV_TYPE=debug bash eval.sh \
  spark0_bench stack_bowls cotrain tianji_marvin_wuji ee 42 0 0 vitra vitra
```

Batched XPolicyLab evaluation is supported by looping over environments because the official `predict_action` implementation currently accepts model batch size 1 only.

## Known limitations

- Spark0 HDF5 uses its recorded `vision/cam_head/{extrinsics,intrinsics}` field names; runtime observations additionally accept XPolicyLab's standard `extrinsics_matrix` and `intrinsic_matrix` keys. The exact H1 simulator fallback is documented above and is rejected for unknown image sizes, camera conventions, or artifact hashes.
- The source HDF5 has no joint-name attribute. Spark-0's alignment target and recorder permutation establish the stage-major 20-D order; old finger-major prose in mapping artifacts must not be used to index these arrays.
- `wuji20`'s sparse MANO component/sign assignment remains an uncalibrated legacy assumption. `mano45` instead uses the fitted dense labels and either the historical numerical inverse or the explicit paired-data linear inverse. Both 45-to-20 projections are approximate; monitor geometry residuals and the side-specific inverse backend/fallback counters.
- The workspace-level `env_cfg/tianji_marvin_wuji.yml` registers robot dimensions for conversion and debug evaluation. A simulator benchmark still needs that benchmark's full scene, camera, and `config.sim` settings; none are fabricated by this adapter.
- Data preparation scripts live in the workspace-level `data_scripts/` directory because that layout is required by this integration task. Move or package them with the adapter before submitting a standalone upstream XPolicyLab PR.
