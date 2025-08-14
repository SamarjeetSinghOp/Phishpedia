# RF-DETR integration guide

This project now supports using RF-DETR as the detector (while keeping the same Siamese logo matcher) and a side-by-side comparison runner.

## Prerequisites
- Python environment that runs the existing project
- Install RF-DETR and its dependencies:
  - rfdetr
  - supervision
  - pillow (already used here)

Optional GPU acceleration is supported if your PyTorch install is CUDA-enabled.

## Install packages (Windows PowerShell)
Optional commands, run inside your active environment:

- pip install rfdetr supervision

If you use pixi, you can also add these as dependencies in your environment.

## Configure
Update `configs.yaml` or use CLI flags:

- To use RF-DETR globally via config:
  - ELE_MODEL.DETECTOR_TYPE: rfdetr
  - ELE_MODEL.RFD_WEIGHTS_PATH: models/test_weights.pt (or your checkpoint)
  - ELE_MODEL.RFD_THRESHOLD: 0.35 (adjust as needed)
  - ELE_MODEL.RFD_RESOLUTION: 512 (optional)

- To override per run via CLI without changing config:
  - --detector rfdetr  (or rcnn)

## Run main pipeline
The main script accepts `--detector` to override and `--export_pip_list` to save the installed packages list at the end of the run.

Example (PowerShell):
- pixi run python phishpedia.py --folder datasets/test_sites --detector rfdetr --analysis_mode --export_pip_list

Outputs:
- analysis_YYYYMMDD/ with per-sample analysis and detection_visual.png
- pip_list.txt saved into the analysis folder (when --export_pip_list is used)

## Side-by-side comparison
Use `compare_detectors.py` to run both RCNN and RF-DETR on the same inputs and feed detections into the Siamese matcher.

Example (PowerShell):
- pixi run python compare_detectors.py --folder datasets/test_sites --max_samples 20 --output compare_run

Outputs:
- compare_run/<sample>/rcnn/detection_visual.png
- compare_run/<sample>/rfdetr/detection_visual.png
- compare_run/summary.json with per-detector timings, counts, and matches.

## Notes
- RF-DETR weights file (e.g., models/test_weights.pt) must exist; update the path in configs.yaml accordingly.
- If RF-DETR isn’t installed and you select it, the detector stage will error out and be recorded in the analysis logs.
- The visualization remains minimal: only matched brand boxes and brand label below the box; no confidence overlay and no RCNN-only boxes.
