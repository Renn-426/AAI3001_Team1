# Child Safety Monitor

A team computer-vision prototype combining hazard detection, a person reference, relative-depth/spatial analysis, a Streamlit interface and Telegram caregiver alerts.

## Project overview

The prototype addresses camera monitoring of potentially hazardous household objects, including batteries, small items, medicines and sharp tools. It aims to support caregiver awareness while retaining the detection and spatial limitations described below.

This project is a prototype monitoring assistant. It watches a camera feed, image or video and:

1. **Detects household hazards** with a custom-trained YOLOv8 model (seven hazard groups).
2. **Detects people** with a stock COCO-pretrained YOLOv8n person detector, used as the reference point for proximity. For demo purposes, any detected person stands in for the child.
3. **Estimates relative depth** with Depth Anything V2 Small to compare frame-relative depth at the person and hazard boxes. These values are not physical distances.
4. **Scores spatial risk** from the image-plane gap, the depth gap and the hazard type, then displays a `SAFE`, `LOW`, `MEDIUM` or `HIGH` assessment. The spatial explanation separately uses geometric zones named SAFE, CAUTION, WARNING and CRITICAL.
5. **Supports caregiver alerts** through Telegram when the configured level and duration conditions are met, with a cooldown to limit repeated messages.

## Team and Contributions

**Team:** AAI3001 — Lab P1, Group 1

| Team Member | Admin Number | Contribution |
|:---:|:---:|:---|
| Ee Chew Fong Olivia | 2102743 | Streamlit interface, camera and image/video testing, Telegram alerts and application integration |
| Thet Htar San | 2502046 | Streamlit interface, camera and image/video testing, Telegram alerts and application integration |
| Teng Tze Hui | 2502437 | Dataset preparation, annotation validation, augmentation, model training, experiment evaluation and model-results documentation |
| Fujita Ayumi | 2502515 | Spatial analysis, proximity calculations, risk scoring and spatial integration |
| Low Yi Hang Nicholas | 2500831 | Spatial analysis, proximity calculations, risk scoring and spatial integration |

**Shared contributions:** system integration, testing, project discussions, README, report and slides review.

## Documentation

| Guide | What it covers |
|:---:|:---|
| [UI and application guide](docs/UI_GUIDE.md) | Installation, camera controls, image/video testing, spatial/depth integration, Telegram and deployment |
| [Model experiments, results and limitations](docs/MODEL_RESULTS.md) | Baseline and six experiments, evaluation protocol, selected configuration, findings, limitations, future work and artifact records |
| [Spatial analysis and integration](docs/SPATIAL_ANALYSIS.md) | Relative box distance, depth sampling, risk formula, temporal handling, tests and seven-class risk mapping |
| [Cloud deployment guide](docs/DEPLOYMENT.md) | Streamlit Community Cloud, Google Cloud setup, camera relay, costs and operational checks |
| [Executed notebook](Child_Safety_Hazard_Detection_YOLOv8.ipynb) | Full preparation/training/evaluation code and recorded outputs |
| [Deliverables](Deliverables/) | Final poster, presentation slides, and group contributions and AI reflection |

> **Reading the report on GitHub:** the executed notebook is approximately 38 MB and may be too large for GitHub's notebook preview. The [model guide](docs/MODEL_RESULTS.md) is written to stand alone, with readable result tables and figures. Download the notebook and open it in Jupyter or VS Code if you need the full executed cells.

## Repository structure

```text
AAI3001_Team1/
├── app.py                         # Streamlit app: live monitoring, test image/video, alerts
├── Child_Safety_Hazard_Detection_YOLOv8.ipynb   # Dataset prep, training and evaluation
├── modules/
│   ├── detector.py                # Loads the trained YOLOv8 hazard detector
│   ├── classifier.py              # Connects detector → spatial analysis → UI
│   ├── spatial.py                 # Spatial risk analysis (distance → danger level)
│   ├── spatial_pipeline.py        # Shared YOLO + depth + spatial glue for app and tests
│   ├── depth_estimator.py         # Depth Anything V2 relative depth
│   ├── camera.py                  # Live camera via streamlit-webrtc
│   ├── rtc_config.py              # TURN relay settings for cloud camera connections
│   ├── notifier.py                # Telegram alert sending
│   ├── subscriptions.py           # Telegram homes, rooms and subscribers
│   ├── wholepipeline.py           # Earlier combined video pipeline (person + hazard + spatial)
│   └── tests/                     # Unit and live tests (spatial, depth, pipeline, TURN)
├── weights/final_best.pt          # Selected detector (YOLOv8s, 640 px)
├── configs/model_training/        # Selected detector settings and artifact manifest
├── results/                       # Validation, test and training-history CSVs, split membership
├── figures/                       # Dataset examples and evaluation figures
├── docs/                          # UI guide, model results, spatial analysis, deployment
├── deploy/gcp/                    # Google Cloud VM setup scripts
├── Deliverables/                  # Poster, slides, contributions and AI reflection
├── .streamlit/config.toml         # Streamlit settings
├── requirements.txt               # Python dependencies
└── packages.txt                   # Linux packages for Streamlit Community Cloud
```

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Obtain the selected checkpoint separately and place it at `weights/final_best.pt`, or configure `HAZARD_MODEL_PATH`. Weights are ignored by Git. Without the trained checkpoint, the app uses a stock-model demonstration. If pipeline construction fails, the app displays an error and uses a random-score fallback. Neither is equivalent to the selected seven-group detector. See the [UI guide](docs/UI_GUIDE.md).

## Dataset

### How the dataset was built

The hazard dataset combines **public online images** and **photographs the team took and labelled**:

#### 1. Public datasets (selected sources)

The table below lists the public sources that the team recorded. It is not the full list: some images also came from other public datasets (mainly on Roboflow Universe and Kaggle) whose links were not saved during collection, and from online marketplace listings (e.g. Carousell). These unrecorded sources are acknowledged as a group rather than individually.

| Source | Images/categories sourced | Link |
|:---|:---|:---|
| **ChildSUn** (Kaggle) | Source dataset contains 5,350 images categorised as safe or unsafe for children. This is the source dataset size, not a verified count of images retained in our final dataset. | [kaggle.com/datasets/fairuzkhan002/datasetvgg19](https://www.kaggle.com/datasets/fairuzkhan002/datasetvgg19) |
| **Open Images Dataset V7** | Knives, scissors, kitchen scenes, medicine | [Open Images V7 visualiser](https://storage.googleapis.com/openimages/web/visualizer/index.html?type=all&set=train&c=%2Fm%2F01lsmm) |
| **LVIS** | Knives, scissors, bottles, bottle caps | [platform.ultralytics.com/ultralytics/datasets/lvis](https://platform.ultralytics.com/ultralytics/datasets/lvis) |
| **Baby-Danger** (Roboflow Universe) | Unsafe objects for children | [https://universe.roboflow.com/fyp-ytggz/baby-danger](https://universe.roboflow.com/fyp-ytggz/baby-danger) |
| **Harmful Objects** (Roboflow Universe) | Harmful objects that pose risk | [https://universe.roboflow.com/harmfull-objects/harmful-objects-wmmdi](https://universe.roboflow.com/harmfull-objects/harmful-objects-wmmdi) |
| **Medicine Tablet Pack Image Dataset** (Kaggle) | Pill packets / blister packs | [kaggle.com/datasets/nitesh31mishra/medicine-tablet-pack-image-dataset](https://www.kaggle.com/datasets/nitesh31mishra/medicine-tablet-pack-image-dataset) |
| **Drug Detection Dataset (7 classes)** (Kaggle) | Medicines | [kaggle.com/datasets/mfajarmaulana/drug-detection-dataset-7-classes](https://www.kaggle.com/datasets/mfajarmaulana/drug-detection-dataset-7-classes) |
| **Marble** (Roboflow Universe) | Marbles | [universe.roboflow.com/1-_/marble-vzkli](https://universe.roboflow.com/1-_/marble-vzkli) |
| **Pen Cap** (Roboflow Universe) | Pen caps | [universe.roboflow.com/serahs-workspace/pen-cap](https://universe.roboflow.com/serahs-workspace/pen-cap) |
| **Bottle Cap** (Roboflow Universe) | Bottle caps | [universe.roboflow.com/work2-rjplo/bottle-cap-oiuwp](https://universe.roboflow.com/work2-rjplo/bottle-cap-oiuwp) |
| **Online marketplace listings** (e.g. Carousell) | Product photos of household hazard items | Collected for non-commercial academic use; individual listings not recorded |

Each source keeps its own licence and terms of use. Check the linked dataset page before redistributing any images. The full image dataset is stored separately. This repository includes split assignments and selected illustrative images, such as the examples below.

#### 2. Self-collected photographs

The team contributed about **450 self-collected photos**, taken by all five members, covering batteries, bottle caps, coins, marbles, pen caps, pill bottles, pill packets, scissors, small toy parts and medicine syrup. Team members photographed the hazard objects in real homes: on beds, blankets, clutter, floors and tables, under ordinary indoor lighting. These photos add cluttered backgrounds and everyday phone-camera viewpoints. Their coverage of the intended monitoring conditions still needs evaluation.

<table align="center">
  <tr>
    <td align="center"><img src="figures/dataset/pill_bottle_raw.jpg" alt="Self-collected photo of a pill bottle on a cluttered desk" width="280"><br><em>Raw self-collected photo</em></td>
    <td align="center"><img src="figures/dataset/pill_bottle_labelled.jpg" alt="Same photo with a green bounding box labelled pill_bottle" width="280"><br><em>Labelled Ourselves</em></td>
  </tr>
</table>

#### 3. Annotation

- **Tool:** the team annotated the self-collected photos in tools like **[Roboflow](https://roboflow.com/)**, drawing a bounding box around each hazard object and exporting the labels in YOLO format.
- **Public images:** labels were checked or relabelled where needed so that every source uses the same 14-class scheme.

**Example: from photo to label.** For the pill bottle photo above, the exported YOLO label file ([pill_bottle_label.txt](figures/dataset/pill_bottle_label.txt)) has one line per object:

```text
12 0.552754 0.581933 0.218487 0.231092
```

| Value | Field | Meaning | In pixels (960 × 1280 image) |
|:---:|:---|:---|:---|
| `12` | class ID | `pill_bottle` in the original 14-class scheme | Mapped to final group **6 `medicine`** during preparation |
| `0.552754` | x centre | Box centre ÷ image width | 0.552754 × 960 ≈ **531 px** |
| `0.581933` | y centre | Box centre ÷ image height | 0.581933 × 1280 ≈ **745 px** |
| `0.218487` | width | Box width ÷ image width | 0.218487 × 960 ≈ **210 px** |
| `0.231092` | height | Box height ÷ image height | 0.231092 × 1280 ≈ **296 px** |

All four coordinates are **normalised to 0–1**, so the label stays valid if the image is resized. The box corners are `(x1, y1) ≈ (426, 597)` and `(x2, y2) ≈ (636, 893)`. 

### Classes

The original annotations use **14 classes**, which are grouped into **7 hazard groups** for the final detector:

| Final ID | Hazard group | Original classes |
|:---:|:---|:---|
| 0 | `battery` | battery |
| 1 | `coin_battery` | coin_battery |
| 2 | `coin` | coin |
| 3 | `small_object` | bottlecap, dice, marble, small_toy |
| 4 | `pen` | pen, pencap |
| 5 | `sharp` | knife, scissors |
| 6 | `medicine` | medicine_syrup, pill_bottle, pill_packet |

Grouping reflects the intended hazard categories. It is a design choice, not a demonstrated accuracy improvement. Predictions identify the group rather than the original item, such as knife versus scissors. The current spatial mapping recognises all seven groups; their risk weights, including the combined pen/pen-cap group, still require validation.

### Cleaning and splitting

| Step | Result |
|:---|:---|
| Raw images | 18,073 |
| Removed exact duplicates | 923 |
| Removed duplicates with conflicting annotations | 14 |
| Removed empty-label images | 2 |
| **Prepared images** | **17,134** |

- **Leakage control:** near-duplicate photos are grouped by visual hashing, and each photo group stays in a single split.
- **Split:** 11,996 train / 2,570 validation / 2,568 test images (20,184 / 4,370 / 4,173 objects).
- **Training-only additions:** underrepresented classes are oversampled, background (negative) images and object-paste examples are added, and online augmentation varies scale, framing, lighting, blur and noise. Validation and test images are not augmented. Separate *stress* copies of validation and test apply synthetic degradation for robustness evaluation.

Per-image split assignments are in [results/split_membership.csv](results/split_membership.csv). The images themselves are not in Git. See [data preparation details](docs/MODEL_RESULTS.md#why-data-preparation-matters).

## Selected hazard detector

**Experiment 3: YOLOv8s, input size 640, confidence threshold 0.30, detector-only.** It has the highest mAP@50–95 among the six detectors across clean validation, validation stress, clean test and test stress. Clean-test mAP@50–95 is **0.6549**, and test-stress mAP@50–95 is **0.6080**.

Experiment 6 obtains **96.63% crop-validation accuracy**, but adding it to the detector increases missed hazards and false detections. It is retained as an experiment, not recommended for the final operating configuration.

## Experiment records in this repository

- `results/validation/`: recorded validation scores and threshold sweeps.
- `results/final_test/`: detector, pipeline, class, size and controlled-condition CSV results.
- `results/training_history/`: original epoch histories and run arguments for the completed baseline and Experiments 1–6.
- `results/split_membership.csv`: exported filenames, categories, photo groups and split assignments; no image files.
- `figures/model_training/`: selected evaluation and training figures.
- `configs/model_training/selected_detector.json`: recommended settings and seven-class mapping. This is a record, not an automatically loaded app configuration.
- `configs/model_training/artifact_manifest.json`: source runs, checkpoint hashes, notebook hash and copied-record checksums.

Large datasets, generated image folders and model checkpoints remain outside normal Git tracking. Their locations and access requirements are documented in the [model guide](docs/MODEL_RESULTS.md#artifact-locations-and-verification).

## System scope and integration status

```text
Camera/image/video -> trained hazard detector ----+
                  -> COCO person detector --------+-> relative spatial assessment -> UI / optional alerts
                  -> relative-depth estimation --+
```

The selected checkpoint supplies seven hazard groups. The current spatial mapping includes all seven, including `small_object`, `sharp` and `medicine`. Group coverage is implemented, while the risk policy still requires validation. In particular, the combined `pen` group includes pen caps but retains the legacy pen weight. See [current label handling](docs/SPATIAL_ANALYSIS.md#seven-class-integration-and-remaining-checks).

The model study evaluates hazard detection on annotated images. The [spatial guide](docs/SPATIAL_ANALYSIS.md) explains proximity assumptions, depth freshness and score persistence; the [UI guide](docs/UI_GUIDE.md) covers camera operation and alert configuration.

## Known Limitations

### Detection trade-off

The project prioritises recovering hazards while monitoring false detections. The confidence threshold `0.30` was selected on validation to minimise false detections per image among tested thresholds meeting overall recall of at least `0.85`. This is a project criterion, not a validated safety standard. False alerts can also distract or fatigue caregivers, so their cost should not be dismissed.

### Hazard detection

| Selected detector at confidence 0.30 | Clean test | Synthetic stress |
|:---|:---:|:---:|
| False detections | 546 | 366 |
| Missed objects | 469 | 721 |

- **Errors in both directions remain.** The clean test contains 2,568 images. These are object-level counts, not caregiver-alert counts. Under stress, fewer false detections occur alongside more missed hazards.
- **Small objects remain difficult.** Recall is about `0.65` for objects with a shorter side of 16–32 model-input pixels, compared with `0.92` at 64+ pixels. These measurements do not establish a physical operating distance.
- **Some classes are weaker.** Stress recall is approximately `0.70` for pen, `0.75` for sharp objects and `0.77` for battery.
- **Camera conditions can differ from the test data.** Synthetic blur, noise and lighting changes do not replace independent testing in representative homes. A frame with no detections does not establish that the scene is safe.

See [model failures and limitations](docs/MODEL_RESULTS.md#failure-analysis-and-practical-consequences) for class, size and confidence analyses.

### Spatial analysis and relative depth

- **Proximity is relative, not measured in metres.** The image-plane calculation divides box-edge separation by the person's box height. Posture, camera angle and truncated detections affect it; overlapping boxes do not establish contact.
- **Depth is relative, not metric.** The pipeline normalises the depth map within each frame and samples the central part of each detection box. The person's sampled region does not necessarily represent a reaching hand. Reflective surfaces, transparent objects and very small objects require additional evaluation.
- **Depth can describe an earlier frame.** Live assessment accepts a depth map only while its source-frame age is at most `1.5 seconds`; older maps fall back to image-plane proximity. Even an accepted map can lag a moving scene.
- **Displayed scores can persist.** A one-second score hold and smoothing reduce flicker but delay clearance after the raw risk falls. This is separate from model processing latency.
- **Risk weights are manually assigned.** Hazard weights, distance thresholds and the `0.30` foreground-held-object depth allowance have not been validated against physical reach or injury risk. All seven labels are recognised, but the merged pen/pen-cap group still uses the legacy pen weight of `0.30` and needs review.
- **A detected person is only a reference.** The model does not identify age. SAFE is a computed application state, not a safety certification.

See [spatial assumptions and remaining checks](docs/SPATIAL_ANALYSIS.md#seven-class-integration-and-remaining-checks) for formulas, timing behaviour and implementation details. Detector evaluation alone does not validate the complete camera/depth/alert pipeline.

## Cloud Deployment

**Live application:** [Child Safety Monitor](https://aai3001team1-cvdl-child-safety-monitor.streamlit.app)

Streamlit Community Cloud is the main deployment, with a Google Cloud VM as the backup. Both use Cloudflare TURN when the browser camera needs a relay. TURN improves connectivity rather than model processing speed.

| Deployment | Role | Operational consideration |
|:---|:---|:---|
| Streamlit Community Cloud | Main hosted application | Shared resources; local SQLite storage may not survive rebuilds or restarts |
| Google Cloud VM | Separately managed backup | Dedicated resources; compute incurs charges while running, and disk storage has separate charges |

The team recorded successful uploaded-image inference, live monitoring and Telegram delivery. These functional checks do not establish real-world detection or false-alert rates. The Google Cloud CPU configuration has recorded timings below; its GPU option remains untested.

### Measured processing latency (CPU server)

Team-recorded timings from **Performance details → Latency**:

| `MODEL_THREADS` | Reported processing latency per frame |
|:---:|:---:|
| 2 (laptop default) | 140–190 ms |
| **4 (VM default)** | **100–130 ms** |
| 6 | 115–135 ms |

These panel measurements are not camera-to-notification latency or a guaranteed video frame rate. Capture, network transport, background-depth age and rendering contribute separate delays. More CPU threads did not consistently improve these measurements.

### Setup and operation

See the [deployment guide](docs/DEPLOYMENT.md) for prerequisites, secrets, server options, deployment commands, cost estimates and start/stop procedures.

- Stop the paid backup VM after use; its disk can continue to incur storage charges.
- Use the latest tunnel URL after restarting the backup deployment.
- Run only one Telegram poller per bot token to avoid command conflicts between deployments.

## References

Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and flexible image augmentations. *Information, 11*(2), 125. https://doi.org/10.3390/info11020125

fairuzkhan002. (n.d.). *ChildSUn* [Data set]. Kaggle. https://www.kaggle.com/datasets/fairuzkhan002/datasetvgg19

fyp-ytggz. (n.d.). *Baby-Danger* [Data set]. Roboflow Universe. https://universe.roboflow.com/fyp-ytggz/baby-danger

Gupta, A., Dollár, P., & Girshick, R. (2019). LVIS: A dataset for large vocabulary instance segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 5356–5364). https://doi.org/10.1109/CVPR.2019.00550

Harmfull Objects. (n.d.). *Harmful objects* [Data set]. Roboflow Universe. https://universe.roboflow.com/harmfull-objects/harmful-objects-wmmdi

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Version 8.4.198) [Computer software]. https://github.com/ultralytics/ultralytics

Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., Duerig, T., & Ferrari, V. (2020). The Open Images Dataset V4. *International Journal of Computer Vision, 128*(7), 1956–1981. https://doi.org/10.1007/s11263-020-01316-z

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. In *Computer Vision – ECCV 2014* (pp. 740–755). Springer. https://doi.org/10.1007/978-3-319-10602-1_48

*Marble* [Data set]. (n.d.). Roboflow Universe. https://universe.roboflow.com/1-_/marble-vzkli

mfajarmaulana. (n.d.). *Drug detection dataset (7 classes)* [Data set]. Kaggle. https://www.kaggle.com/datasets/mfajarmaulana/drug-detection-dataset-7-classes

nitesh31mishra. (n.d.). *Medicine tablet pack image dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/nitesh31mishra/medicine-tablet-pack-image-dataset

serahs-workspace. (n.d.). *Pen cap* [Data set]. Roboflow Universe. https://universe.roboflow.com/serahs-workspace/pen-cap

work2-rjplo. (n.d.). *Bottle cap* [Data set]. Roboflow Universe. https://universe.roboflow.com/work2-rjplo/bottle-cap-oiuwp

Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2024). *Depth Anything V2*. arXiv. https://doi.org/10.48550/arXiv.2406.09414

*Note.* Additional images came from other public datasets whose links were not recorded, online marketplace listings (e.g. Carousell) and photos taken by the team. All were used for non-commercial academic purposes.
