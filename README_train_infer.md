# SDM-EXR LINO–SDM Inference and Comparison Guide

This guide covers only the new SDM-style EXR workflow on the
`dev-lino-sdm-comparison` branch. It starts with environment installation and
ends with paired LINO/SDM angular-error results.

## Training

There is no training command in this comparison workflow. It uses released
LINO-UniPS and SDM-UniPS checkpoints and does not retrain or change either
model architecture.

## 1. Clone LINO and install its environment

The pinned packages in `requirements.txt` use CUDA 12.4 builds, including
PyTorch, xFormers, and `spconv_cu124`. Use a compatible NVIDIA driver/GPU.

```bash
git clone -b dev-lino-sdm-comparison https://github.com/Min-KiD/LINO_UniPS.git
cd LINO_UniPS

conda create -n LINO python=3.10 -y
conda activate LINO
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The SDM repository needs its own working Python environment and dependencies.
Do not install the SDM dependencies into the LINO environment unless that is
already how your machine is configured.

## 2. Define the local paths used below

Run this from the LINO repository and replace every `/absolute/path/...`
value. These shell variables make the later commands easier to copy.

```bash
export LINO_REPO="$(pwd)"
export SDM_REPO="/absolute/path/to/PhotometricStereo"
export LINO_CKPT="/absolute/path/to/lino.pth"
export SDM_CKPT="/absolute/path/to/optimized_sdm.pth"
export SDM_PYTHON="/absolute/path/to/sdm/environment/bin/python"
export CONFIG="$LINO_REPO/configs/sdm_exr_infer.yaml"
```

Required local assets:

- `LINO_CKPT`: the released LINO normal-estimation checkpoint.
- `SDM_CKPT`: the SDM checkpoint to compare against LINO.
- `SDM_REPO`: the SDM repository containing `main.py` and
  `configs/baseline_optimized_infer.yaml`.
- `SDM_PYTHON`: the executable from the working SDM environment.

Confirm the paths before continuing:

```bash
test -f "$LINO_CKPT" &&
test -f "$SDM_CKPT" &&
test -f "$SDM_REPO/main.py" &&
test -f "$SDM_REPO/configs/baseline_optimized_infer.yaml" &&
test -x "$SDM_PYTHON" &&
echo "All required paths exist."
```

If the confirmation is not printed, at least one path must be corrected.

## 3. Prepare the SDM-style EXR dataset

`data_root` must contain one directory per object. With the checked-in
defaults, object directories end in `.data`, observations begin with `image`,
and all images are three-channel EXRs.

```text
/absolute/path/to/test/
├── object_a.data/
│   ├── image01.exr
│   ├── image02.exr
│   ├── ...
│   ├── image16.exr
│   ├── binary_mask.exr
│   └── local_normal.exr
└── object_b.data/
    ├── image01.exr
    ├── ...
    ├── image16.exr
    ├── binary_mask.exr
    └── local_normal.exr
```

Important requirements:

- Every object needs at least `max_image_num` matching observation EXRs.
- Observations, mask, and normal must have the same source resolution.
- `local_normal.exr` is used only for scoring and is hidden from model inputs.
  Set `normal_encoding: signed` for XYZ values already in signed coordinates, or
  `normal_encoding: unsigned` for [0,1] values decoded as `2 * encoded - 1`.
- `binary_mask.exr` is required when `mask_policy: "external"`.
- The mask is not supplied when `mask_policy: "full"`.

## 4. Edit the single comparison YAML

Edit:

```text
configs/sdm_exr_infer.yaml
```

At minimum, replace these values:

```yaml
checkpoint: "/absolute/path/to/lino.pth"
data_root: "/absolute/path/to/test"
output_root: "./output/sdm_lino_comparison"
max_image_num: 16
mask_policy: "external"
precision: "bf16"
device: "auto"
```

Shell variables such as `$LINO_CKPT` are not expanded inside YAML. Enter the
real path in the file.

Use only this one YAML:

- `mask_policy: "external"` uses `binary_mask.exr` as model input.
- `mask_policy: "full"` uses an all-ones model input and provides no mask to
  SDM.
- `light_selection: "seeded"` creates a deterministic selected-light manifest.
- `light_selection: "manifest"` reuses the exact file named by
  `selection_manifest` for a pinned final experiment.

The selected policy is automatically appended to `output_root`. For example:

```text
./output/sdm_lino_comparison/external/
./output/sdm_lino_comparison/full/
```

## 5. Run released-checkpoint LINO inference

Run this in the LINO environment:

```bash
cd "$LINO_REPO"
conda activate LINO
python eval.py --config "$CONFIG"
```

The command reports object 1, every 100 completed objects, and the final
object with elapsed time and ETA. On success it prints the LINO macro-object
MAE and total end-to-end inference time. This immediate MAE is computed only
after each prediction and never supplies ground truth to the model.

Example completion output:

```text
LINO progress: 2399/2399 | elapsed 38:05:57 | ETA 00:00:00
Inference complete: 2399 objects -> output/sdm_lino_comparison/external/lino
Mean MAE (2399 objects): 0.7123
Total inference time: 38:06:41
```

Runtime and VRAM are not compute-matched between the released pipelines: this
LINO route can process a square up to 2048x2048 in 512x512 tiles, while the
current optimized SDM preset uses 256x256. Treat timing as released-pipeline
timing, not an architecture-only speed comparison.

A successful run writes one prediction directory per object:

```text
<output_root>/<mask_policy>/lino/<object>.data/normal_pred.exr
<output_root>/<mask_policy>/lino/<object>.data/normal_pred.png  # when save_png: true
```

`normal_pred.exr` is the authoritative signed, source-resolution prediction.
When enabled, `normal_pred.png` is only a preview and is never used for
scoring.

## Private 256x256 EXR transfer gate (before training)

Run this dedicated, inference-only route before planning any training. It fixes
the private HDR transfer contract at exactly 16 seeded lights, an external
mask, unsigned source normals, 256x256 source geometry, one 512x512 model tile,
bf16 precision, and CUDA. The output root is isolated from the comparison
workflow. It uses the released LINO checkpoint and does not run training,
`compare_sdm.py`, SDM inference, or an SDM comparison.

Use the following commands from the LINO checkout:

```bash
cd /mnt/16TData/minhnv/LINO
conda activate LINO
python -m pip install -r requirements.txt

mkdir -p checkpoints
if [ ! -s checkpoints/lino.pth ]; then
  wget -c \
    "https://huggingface.co/houyuanchen/lino/resolve/main/lino.pth" \
    -O checkpoints/lino.pth
fi
test -s checkpoints/lino.pth

test -d /mnt/18TData/minhnv/inference
python eval.py --config configs/lino_private_transfer.yaml
```

The successful output includes one signed prediction per object and a
completion marker, for example:

```text
output/lino_private_transfer/external/lino/alpha.data/normal_pred.exr  # example object
output/lino_private_transfer/external/lino/run.json
```

`normal_pred.exr` stays signed and identity-coordinate. `run.json` is created
only after all objects finish and contains source, checkpoint, config, and
output hashes; selected lights; per-object mask, HDR, and geometry diagnostics;
official MAE; the constant baseline; the coordinate diagnostic; elapsed time;
and peak CUDA memory. The 256x256 source geometry and 512x512 model geometry
are deliberately different. The coordinate diagnostic is for convention
diagnosis only and cannot be quoted as the official model MAE. This command is
inference-only: it does not run `compare_sdm.py` or SDM.

Interpret the transfer results with this decision table:

| Observation | Decision |
|---|---|
| Best coordinate MAE is materially lower than identity | Resolve the coordinate convention before training. |
| Identity is close to best and near/worse than the constant baseline | Released-checkpoint transfer failed; design LINO training/fine-tuning next. |
| Identity clearly beats the baseline but misses the project target | Transfer is partial; use the checkpoint as the fine-tuning start. |
| Identity is acceptable | Postpone training and design a separately controlled SDM-versus-LINO comparison. |

No numeric pass threshold is imposed for this private HDR distribution.

### Where `run.json` is created

LINO writes this file only after every object finishes successfully:

```text
<output_root>/<mask_policy>/lino/run.json
```

With the default relative output and external-mask policy, it is:

```text
output/sdm_lino_comparison/external/lino/run.json
```

`run.json` is runtime provenance, not a source file or checkpoint. It records
the LINO config, checkpoint and manifest hashes, selected inputs, output EXR
hashes, device/precision, and runtime information. A failed or interrupted run
intentionally leaves no valid `run.json`, so a partial run cannot be scored as
complete.

Check it after inference:

```bash
test -f "output/sdm_lino_comparison/external/lino/run.json" \
  && echo "LINO run complete"
```

Adjust the path if `output_root` or `mask_policy` differs.

## 6. Prepare the GT-hidden SDM request

Stay in the LINO environment. This command validates the current assets,
persists the exact selected-light manifests, creates a ground-truth-hidden SDM
input view, and creates a fresh request/output directory.

```bash
cd "$LINO_REPO"
conda activate LINO

python compare_sdm.py prepare-view \
  --config "$CONFIG" \
  --sdm-repo "$SDM_REPO" \
  --sdm-checkpoint "$SDM_CKPT" \
  --sdm-python "$SDM_PYTHON"
```

`prepare-view` does not run SDM inference. It prints three important lines:

1. The generated GT-hidden SDM input-view path.
2. The exact SDM inference command, beginning with `SDM_PYTHON`.
3. The exact LINO-environment `finalize-sdm` command containing the generated
   request JSON path.

Keep this terminal output. The request ID is unique for every preparation.

## 7. Run the exact printed SDM command

Copy the complete SDM inference command printed by Step 6 and execute it
exactly. Do not reconstruct or shorten it. It includes the shared selected
lights, mask policy, GT-hidden input view, fresh output directory, seed, and
other fairness-critical arguments.

The command already starts with the absolute `SDM_PYTHON` executable, for
example:

```text
/absolute/path/to/sdm/environment/bin/python \
  /absolute/path/to/PhotometricStereo/main.py infer ...
```

The omitted arguments represented by `...` are required. Use the actual
printed command, not this abbreviated example.

Wait for SDM inference to finish before continuing.

## 8. Finalize the SDM result

Return to the LINO environment and run the complete `finalize-sdm` command
printed by Step 6. Its form is:

```bash
cd "$LINO_REPO"
conda activate LINO

python compare_sdm.py finalize-sdm \
  --config "$CONFIG" \
  --request "/absolute/generated/path/sdm_requests/<request-id>.json"
```

Use the exact generated request path from your terminal. Do not type the
literal `<request-id>` text.

Finalization checks the request, current source data, GT-hidden view, SDM
runtime inputs, output allowlist, EXR geometry, finite values, and prediction
hashes. On success it prints the completion JSON path.

## 9. Score LINO against SDM

Set `REQUEST` to the same request JSON path used for finalization:

```bash
export REQUEST="/absolute/generated/path/sdm_requests/<request-id>.json"
```

Then run:

```bash
cd "$LINO_REPO"
conda activate LINO
python compare_sdm.py score --config "$CONFIG" --request "$REQUEST"
```

The command prints:

- LINO macro mean angular error (MAE).
- SDM macro MAE.
- LINO minus SDM macro MAE.

Both models are scored from signed source-resolution EXRs using the same valid
source-normal support.

This paired score remains the authoritative comparison even though LINO now
prints its standalone MAE after `eval.py`. It additionally validates that LINO
and SDM used the intended selected lights, mask policy, checkpoints,
manifests, requests, hashes, and finalized prediction artifacts.

## 10. Output inventory

All paths below are relative to:

```text
<policy_root> = <output_root>/<mask_policy>
```

```text
<policy_root>/
├── input_manifest.json
├── selected_lights.json
├── lino/
│   ├── run.json
│   └── <object>.data/
│       ├── normal_pred.exr
│       └── normal_pred.png            # when save_png: true
├── sdm_input/
│   └── <object>.data/                 # GT-hidden linked observations
├── sdm_requests/
│   └── <request-id>.json
├── sdm/
│   └── <request-id>/
│       ├── <object-stem>_pred.exr
│       └── ...
├── sdm_completions/
│   └── <request-id>.json
└── comparison/
    ├── per_object.csv
    └── summary.json
```

Important files:

- `selected_lights.json`: exact ordered observations shared by LINO and SDM.
- `lino/run.json`: successful LINO-run provenance.
- `sdm_requests/<request-id>.json`: immutable instructions for one SDM run.
- `sdm_completions/<request-id>.json`: finalized SDM prediction hashes.
- `comparison/per_object.csv`: per-object LINO and SDM angular metrics.
- `comparison/summary.json`: aggregate metrics and artifact provenance.

## 11. Run both mask protocols

### External-mask comparison

Set:

```yaml
mask_policy: "external"
```

Then repeat Steps 5–9. Results go beneath `<output_root>/external/`.

### Mask-free comparison

Change only:

```yaml
mask_policy: "full"
```

Then repeat Steps 5–9. Results go beneath `<output_root>/full/` and do not
overwrite the external-mask run.

### Pin the selected lights for a final repeated experiment

The first seeded run creates:

```text
<output_root>/<mask_policy>/selected_lights.json
```

For a final repeated run, edit the same YAML:

```yaml
light_selection: "manifest"
selection_manifest: "/absolute/path/to/selected_lights.json"
```

Rerun LINO inference and prepare a new SDM request. This prevents an accidental
change in selected-light order or membership.

## 12. Troubleshooting

### `run.json` is missing

LINO did not complete every object, the process was interrupted, or a rerun
invalidated the previous provenance before failing. Read the first LINO error,
fix it, and rerun Step 5. Do not create `run.json` manually.

### OpenCV reports that EXR is disabled or unsupported

The code enables `OPENCV_IO_ENABLE_OPENEXR=1` before importing OpenCV, but the
installed OpenCV build must still include EXR support. Confirm that the active
environment uses the pinned `opencv-python-headless` from `requirements.txt`
and that no conflicting OpenCV wheel shadows it.

### A checkpoint, SDM repository, or Python executable is rejected

Repeat the five `test` commands in Step 2 and use absolute paths. The SDM
Python path must be executable, not merely the environment directory.

### `prepare-view` reports a stale or mismatched manifest

Do not reuse manifests after changing the dataset, selected observations, or
configuration. Return to `light_selection: "seeded"` to generate a fresh
selection, rerun LINO, and prepare a new SDM request.

### Finalization or scoring rejects the request

Use the exact request path printed for the current run. Do not reuse an older
request with a new SDM output, changed checkpoint, changed source data, or
different mask policy.

### SDM output is missing

`prepare-view` only prints the SDM command. Confirm that you copied and ran the
entire printed command and waited for it to finish before `finalize-sdm`.

### Comparison files are absent

`comparison/per_object.csv` and `comparison/summary.json` are created only by
the Step 9 `score` command after both a valid LINO `run.json` and finalized SDM
completion exist.
