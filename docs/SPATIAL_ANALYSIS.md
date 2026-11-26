# Spatial Analysis, Relative Depth and Risk Scoring

[Project overview](../README.md) · [Model findings](MODEL_RESULTS.md) · [UI guide](UI_GUIDE.md)

This guide documents the current spatial implementation and its assumptions. It explains how detector outputs become a relative proximity score, how depth and temporal handling change that score, and what remains unvalidated. It is based on repository source, not on a new physical-camera experiment. Model mAP and crop-classification results are described separately in the model guide.

## Purpose and boundary of the component

Recognising a hazardous object does not by itself determine whether a person is near it. Spatial analysis combines detected person and hazard boxes with optional relative depth, producing a score and an explanation for the interface. It is a hand-designed heuristic. It does not estimate contact, reachability, injury probability or physical distance, and it does not identify a detected person's age.

The code uses the label `person` as the child reference. Names such as `children` and messages saying “child” are application terminology, not evidence of an age-specific model. Missed people or hazards propagate into the spatial result; a correct formula cannot recover missing detector inputs.

## Data flow and component responsibilities

```mermaid
flowchart LR
    F[Image or camera frame] --> H[Trained hazard detector]
    F --> P[COCO person detector]
    F --> D[Relative depth model]
    H --> B[Boxes, labels and confidence]
    P --> B
    B --> S[Depth sampling and person-hazard pairing]
    D --> S
    S --> R[Relative distance and weighted score]
    R --> T[Live score hold and display smoothing]
    T --> U[UI assessment and configured alerts]
```

<table align="center">
  <thead>
    <tr><th align="center">Source</th><th align="center">Responsibility</th></tr>
  </thead>
  <tbody>
    <tr><td align="center"><a href="../modules/detector.py"><code>modules/detector.py</code></a></td><td align="center">Load the trained checkpoint and produce hazard boxes with model class names; default hazard confidence 0.30</td></tr>
    <tr><td align="center"><a href="../modules/classifier.py"><code>modules/classifier.py</code></a></td><td align="center">Combine trained hazard and stock person detections; call spatial analysis; map score to UI level</td></tr>
    <tr><td align="center"><a href="../modules/depth_estimator.py"><code>modules/depth_estimator.py</code></a></td><td align="center">Run Depth Anything V2 Small, sample object regions and normalise relative depth</td></tr>
    <tr><td align="center"><a href="../modules/spatial_pipeline.py"><code>modules/spatial_pipeline.py</code></a></td><td align="center">Shared detection conversion, demo/person thresholds and maximum live-depth age</td></tr>
    <tr><td align="center"><a href="../modules/spatial.py"><code>modules/spatial.py</code></a></td><td align="center">Pairwise distance, depth adjustment, hazard weights, proximity zones, raw risk score and score hold</td></tr>
    <tr><td align="center"><a href="../app.py"><code>app.py</code></a>, <code>TimedClassifier</code></td><td align="center">Background live depth, freshness fallback and temporal display behaviour</td></tr>
  </tbody>
</table>

The class name `HazardSpatialClassifier` refers to app integration; it is **not the Experiment 6 crop classifier**. The recommended trained-model configuration uses Exp3 alone for hazard detection, with the separate person/depth/spatial components still available.

`get_classifier()` chooses the trained pipeline when a checkpoint is found. Without it, the stock demo recognises person and stock-supported hazards such as knife/scissors; it is not equivalent to running the final seven-group detector. In the trained path the person model is YOLOv8n; the stock demo uses YOLOv8s. These are separate from the selected YOLOv8s hazard checkpoint.

## Input contract

Each `Detection` carries a label, confidence, pixel-coordinate box `(x1, y1, x2, y2)` and optional relative depth. Detections without a box are excluded. Labels are lowercased and spaces changed to underscores. The spatial analyzer selects people by `CHILD_LABELS = {"person"}` and hazards by membership in `DANGER_WEIGHTS`.

All person-hazard pairs are evaluated and the highest pair score is returned. This avoids selecting only the most confident person and missing a hazard near another person. It is not object tracking, and the selected pair can change between frames.

If there is no person, or no recognised hazard, the raw analyzer returns zero with an explanatory message. That is a limitation of available inputs, not a declaration that the physical scene is safe. An unknown label can trigger the same absence of recognised hazards. All seven final groups are recognised by the current mapping; their weight choices still need validation, as described below.

## Image-plane proximity

Distance is the shortest separation between the bounding-box edges. For two boxes, the horizontal and vertical gaps are clipped at zero, then combined with Euclidean distance. Overlapping or touching boxes have zero edge separation.

```text
dx = max(hazard_left - person_right, person_left - hazard_right, 0)
dy = max(hazard_top - person_bottom, person_top - hazard_bottom, 0)
pixel_gap = sqrt(dx^2 + dy^2)
planar_distance = pixel_gap / person_box_height
```

Normalising by person height gives a relative unit. For example, a 60-pixel gap beside a 300-pixel-tall person gives 0.20 person-box heights. This reduces sensitivity to uniform image scaling, but not to posture, truncated boxes, perspective or incorrect person localisation. It is not a calibrated reach model, and overlapping 2D boxes do not prove physical contact.

### Worked example: gap ÷ person height

The app does not measure metres. It measures the gap between the person's box and the hazard's box, then divides it by the person's box height. Dividing by the person's height makes the result independent of how far the camera is, so no camera calibration is needed.

**Example.** The person's box is 300 px tall. A coin battery's box is 60 px from the edge of the person's box.

- Distance = 60 ÷ 300 = **0.20 person-heights**
- 0.20 is below 0.5, so the geometric zone is **CRITICAL**
- With no depth gap, proximity = 1 − 0.20 / 3.0 ≈ 0.93, and score = 0.93 × 1.0 (coin battery weight) ≈ **0.93**

If the same person stood twice as close to the camera (box 600 px tall) with the object 120 px away, the answer is still 0.20, so the zone does not change.

**Visual example** (side view, not to scale):

    +-----------+
    |           |
    |  person   |                +------+
    |  box      |<-- 60 px gap ->| coin |
    |  (300 px  |                | batt |
    |   tall)   |                +------+
    +-----------+

    distance = 60 / 300 = 0.20 person-heights  ->  CRITICAL

If the two boxes overlap, the gap is 0. Depth, when available, is added as a separate distance term (see the next section).

## Relative-depth sampling and adjustment

Depth Anything V2 Small is loaded and reused. The implementation sets depth input dimensions to **252 × 252** for the processing trade-off recorded in the code. No end-to-end speed guarantee follows from that setting.

Depth is sampled as the median of the **central 50% of each box's width and height**, reducing influence from box corners. It is normalised using the same map's **2nd and 98th percentiles**, clipped to [0, 1], with larger values treated as nearer the camera. A nearly flat map gives the neutral value 0.5. These are frame-relative values, not metres and not stable coordinates across different frames.

When both detections have depth, the absolute depth difference adds an extra separation term:

```text
depth_gap = abs(person_depth - hazard_depth)
effective_distance = sqrt(planar_distance^2 + (3.0 * depth_gap)^2)
```

If the boxes overlap and the hazard is nearer the camera than the person reference, the implementation subtracts up to **0.30** from the depth gap before combining it. This heuristic prevents some hand/lap-held foreground objects from being treated as far away solely because the person's body has a different depth. It is an assumed tolerance, not measured arm reach.

Without usable depth, effective distance equals planar distance. For fixed boxes the depth term can only maintain or increase separation and thus maintain or reduce the raw score. Errors in depth can therefore suppress an otherwise close 2D pair. Conversely, planar fallback can overstate proximity when the objects lie at different physical depths. Neither path is established as a guarantee against under-warning in real scenes.

## Raw risk score and displayed levels

For each recognised person-hazard pair:

```text
proximity = max(0, 1 - effective_distance / 3.0)
pair_score = proximity * hazard_weight
raw_score = maximum pair_score across all pairs
```

The returned raw score is rounded to three decimals. Confidence controls which detections reach this stage but is **not multiplied into this formula**. The hazard weights and thresholds are manually specified design parameters, not learned injury probabilities or clinically validated severity measures.

There are two distinct classifications:

<table align="center">
  <thead>
    <tr><th align="center">Effective distance (person-box heights)</th><th align="center">Geometric zone used in explanation</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Less than 0.5</td><td align="center">CRITICAL</td></tr>
    <tr><td align="center">0.5 to below 1.5</td><td align="center">WARNING</td></tr>
    <tr><td align="center">1.5 to below 3.0</td><td align="center">CAUTION</td></tr>
    <tr><td align="center">3.0 or more</td><td align="center">SAFE / zero proximity</td></tr>
  </tbody>
</table>

<table align="center">
  <thead>
    <tr><th align="center">Score</th><th align="center">UI assessment level</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Below 0.25</td><td align="center">SAFE</td></tr>
    <tr><td align="center">0.25 to below 0.50</td><td align="center">LOW</td></tr>
    <tr><td align="center">0.50 to below 0.75</td><td align="center">MEDIUM</td></tr>
    <tr><td align="center">0.75 or above</td><td align="center">HIGH</td></tr>
  </tbody>
</table>

A geometric CRITICAL zone does not automatically imply HIGH: score also depends on hazard weight. For example, a coin battery with weight 1.0 at effective distance 0.20 gives approximately 0.933 and HIGH; a recognised `pen` with weight 0.30 at zero separation gives 0.30 and LOW. These are illustrative calculations from the formula, not observed camera-test results. The low pen weight is especially problematic when the final model's `pen` group also includes pen caps.

The zero/SAFE label is a computational state. It must not be read as certified safety when inputs are absent, detections fail, labels are filtered or the model misses small hazards.

## Live timing, fallback and score persistence

Image/video test calls can compute depth from the frame being assessed. Live monitoring uses a background worker with one outstanding depth job, so detection can continue while depth processing is slower.

The current `TimedClassifier` measures map age from the **submission time of its source frame**. A completed depth map is used only while its age is at most **1.5 seconds**, and incompatible map dimensions are rejected by the delegate. Older maps fall back to image-plane-only assessment and receive an explicit status message. The UI may still display an old map with its age; displaying it does not mean it was used for scoring. Even a map within the age limit can describe an earlier moving scene.

Live scores rise immediately but persist through brief drops using a **one-second `ScoreHold`**, followed by an additional limit on how quickly the displayed score can fall. The displayed level can therefore outlast the current frame's raw result. This reduces flicker but introduces persistence and clearance delay; it is not tracking or proof a hazard remains present. Alert behaviour depends on additional UI duration/cooldown settings, not just the raw spatial zone.

<a id="seven-class-integration-gap"></a>

## Seven-class integration and remaining checks

The current `DANGER_WEIGHTS` mapping recognises all seven final detector groups as well as legacy class names. `SpatialAnalyzer.assess()` includes these labels in pairwise risk scoring. The table records the implemented weights, not validated injury probabilities.

| Final detector label | Current weight | Interpretation / remaining check |
|:---|:---:|:---|
| `battery` | 0.85 | Included in pairwise scoring |
| `coin_battery` | 1.00 | Included in pairwise scoring |
| `coin` | 0.80 | Included in pairwise scoring |
| `small_object` | 0.75 | Shared weight for bottle caps, dice, marbles and small toys |
| `pen` | 0.30 | Legacy pen weight also covers pen caps; review this grouped policy |
| `sharp` | 1.00 | Shared weight for knives and scissors |
| `medicine` | 0.95 | Shared weight for the merged medicine classes |

The legacy `medicine_syrup` to `liquid_medicine` alias remains available. The final `medicine` group has its own explicit entry and does not depend on that alias.

Recognised labels can reach the spatial calculation, but dictionary coverage alone does not validate the chosen severity weights, alert thresholds or real-camera performance. Group predictions cannot recover the original fine-grained item, so the pen-cap distinction needs an explicit group-level policy.

## Existing checks and what they do not establish

<table align="center">
  <thead>
    <tr><th align="center">Record</th><th align="center">Coverage visible in source</th></tr>
  </thead>
  <tbody>
    <tr><td align="center"><a href="../modules/tests/test_hazard_pipeline.py"><code>test_hazard_pipeline.py</code></a></td><td align="center">Stand-in detector/person outputs, near/far behaviour, absent person, depth differences</td></tr>
    <tr><td align="center"><a href="../modules/tests/test_depth_logic.py"><code>test_depth_logic.py</code></a></td><td align="center">Controlled depth cases, fallback, multi-person pairing and foreground-held-object behaviour</td></tr>
    <tr><td align="center"><a href="../modules/tests/test_distance_demo.py"><code>test_distance_demo.py</code></a></td><td align="center">Manual geometry examples and a direct analyzer example</td></tr>
    <tr><td align="center"><a href="../modules/tests/live_spatial_test.py"><code>live_spatial_test.py</code></a></td><td align="center">Interactive spatial/depth diagnostic script</td></tr>
  </tbody>
</table>

These sources show the intended logical checks. They do not provide a labelled real-world spatial benchmark, calibrated centimetre error, reachability accuracy or missed/false-alert rate. The stand-in hazard tests include legacy examples and a check that every loaded model class has a danger weight. This checks label coverage, not whether each class has an appropriate risk weight. These documentation edits do not report a new test run.

For an appropriate local environment, the existing automated checks can be invoked with:

```powershell
.\.venv\Scripts\python.exe -m pytest modules/tests/test_depth_logic.py modules/tests/test_hazard_pipeline.py -q
```

## Limitations and next validation steps

1. **Validate grouped-label treatment.** All seven labels are now recognised. Check their agreed risk weights, the pen-cap grouping decision and the resulting UI/alert behaviour.
2. **Define a spatial reference task.** Decide whether a positive case means close in the image, same depth, within reach or actual contact. These are different targets; the current formula addresses relative box/depth separation.
3. **Collect recorded camera cases.** Fix camera height/resolution, record object/person positions, include foreground/background overlap, occlusions, truncated people, multiple people and no-person/empty scenes. Physical measurements must be observed, not inferred from box heights.
4. **Assess timing as well as geometry.** Measure capture-to-detection, depth age, displayed-score persistence and notification delay on deployment hardware, including stale/depth-failure fallback.
5. **Calibrate policies against evidence.** Report detection failures separately from spatial errors and alert-policy errors. Evaluate near/far mistakes and false/missed warnings before changing hand-set weights or thresholds.

Model evaluation already shows small-object and degraded-image misses. Those limitations remain upstream risks here. Relative depth can help distinguish some projected overlaps but cannot compensate for an undetected hazard, incorrect class or badly localised person. No-detection and SAFE outputs must be interpreted with these limits visible.
