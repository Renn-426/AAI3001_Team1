# Model Experiments, Results and Limitations

[Project overview](../README.md) · [UI guide](UI_GUIDE.md) · [Executed notebook](../Child_Safety_Hazard_Detection_YOLOv8.ipynb)

This is the standalone technical account of the model work: the problem, experimental decisions, evidence, limitations and deployment implications. Readers do not need to open the notebook to understand the findings. The notebook remains the implementation and executed evidence record; CSVs and figures provide smaller, directly readable supporting artifacts.

> **Notebook preview:** the executed notebook is approximately 38 MB and contains extensive training logs and embedded outputs. GitHub may fail to render it or report that it is too large to display. Read this guide and the linked CSVs/figures in GitHub; download the `.ipynb` and open it locally in Jupyter or VS Code when the executed cells are needed. A preview failure does not mean the results are missing. The recorded outputs remain the evidence for the results below.

## Reading guide

- [Problem and design rationale](#problem-and-design-rationale)
- [Experiment design and selection](#experiment-design-and-selection)
- [Detector results](#detector-results)
- [What each experiment establishes](#what-each-experiment-establishes)
- [Why Experiment 6 is not retained](#why-experiment-6-is-not-retained)
- [Failure analysis and practical consequences](#failure-analysis-and-practical-consequences)
- [Findings and limitations](#findings-and-limitations)
- [Future work and application handoff](#future-work-and-application-handoff)
- [Conclusion](#conclusion)
- [Reproduction and records](#reproduction-and-records)
- [Artifact locations and verification](#artifact-locations-and-verification)

## Problem and design rationale

The detector is intended to supply candidate household hazards to a child-safety monitoring prototype. It must locate objects in a complete image, assign one of the project hazard groups, and provide bounding boxes and confidence scores to downstream spatial analysis. A missed battery or sharp object is therefore consequential even when the overall precision looks high. Detection quality is assessed through recovery of annotated objects and false-box counts, rather than classification accuracy alone.

The model study addresses a specific question: **which tested configuration gives the strongest detection under the same prepared data and training budget, and how does it fail when image conditions change?** It does not train a child detector, estimate physical distance, learn the spatial risk formula or measure caregiver-notification quality. Those are separate components documented in the [spatial guide](SPATIAL_ANALYSIS.md) and [UI guide](UI_GUIDE.md).

### Why seven hazard groups

The fourteen original labels are consolidated into groups that match the application's intended outputs: batteries, coin batteries, coins, small objects, pens/pen caps, sharp objects and medicines. Grouping reduces the number of fine-grained distinctions the final model is asked to make, but the rerun does not contain a controlled fourteen-versus-seven-class experiment. It must therefore be described as an output-design choice, not a demonstrated accuracy improvement.



The table lists the original annotation IDs and their final detector IDs. IDs are reassigned during export; the model's seven-class order must be preserved by downstream code.

<table align="center">
  <thead>
    <tr><th align="center">Before: original ID and label</th><th align="center">After: final ID</th><th align="center">After: hazard-group label</th><th align="center">Meaning retained by the output</th></tr>
  </thead>
  <tbody>
    <tr><td align="center"><code>0 battery</code></td><td align="center">0</td><td align="center"><code>battery</code></td><td align="center">Standard battery group</td></tr>
    <tr><td align="center"><code>1 coin_battery</code></td><td align="center">1</td><td align="center"><code>coin_battery</code></td><td align="center">Coin/button battery group</td></tr>
    <tr><td align="center"><code>3 coin</code></td><td align="center">2</td><td align="center"><code>coin</code></td><td align="center">Coin group</td></tr>
    <tr><td align="center"><code>2 bottlecap</code>, <code>4 dice</code>, <code>5 marble</code>, <code>6 small_toy</code></td><td align="center">3</td><td align="center"><code>small_object</code></td><td align="center">Shared small-object group; original item type is not predicted</td></tr>
    <tr><td align="center"><code>7 pen</code>, <code>8 pencap</code></td><td align="center">4</td><td align="center"><code>pen</code></td><td align="center">Pen and pen cap combined</td></tr>
    <tr><td align="center"><code>9 knife</code>, <code>10 scissors</code></td><td align="center">5</td><td align="center"><code>sharp</code></td><td align="center">Shared sharp-object group</td></tr>
    <tr><td align="center"><code>11 medicine_syrup</code>, <code>12 pill_bottle</code>, <code>13 pill_packet</code></td><td align="center">6</td><td align="center"><code>medicine</code></td><td align="center">Shared medicine group</td></tr>
  </tbody>
</table>

**What changes:** the exported class IDs and class names. This mapping step does not itself change box coordinates or manually correct missing/incorrect annotations. The original source labels remain separate from the prepared seven-class export.

Grouping also loses information. A `sharp` prediction cannot identify whether the original item was a knife or scissors, and `pen` cannot distinguish a pen body from a pen cap. Downstream risk policies need group-level decisions rather than assuming the original fine-grained label can be reconstructed.

### Why data preparation matters

Preparation checks readability and annotation format, removes redundant exact copies and repeated boxes, converts the class mapping and keeps detected photo groups within one split. The source count decreases from **18,073 to 17,134 images** after removing 923 extra exact copies, 14 conflicting-annotation duplicate images and two empty-label images. This provides a traceable prepared population, rather than silently training on whatever files happen to load.

Near-duplicate grouping reduces leakage risk when similar photographs could otherwise occur in training and evaluation. It does not prove independence: hashes can miss related scenes, and some large candidate buckets are skipped. Likewise, valid YOLO coordinates do not prove that all visible hazards were annotated correctly. Thirty-seven folder/class mismatches were retained for review, because multiple object classes can legitimately appear in a photograph.

### Why balancing and augmentation were retained

The training recipe increases exposure to underrepresented classes through repeated image entries and adds background and object-paste examples. Online augmentation varies apparent scale, framing, lighting and image quality. These choices aim to expose the model to conditions relevant to the prototype, but all final detectors share the recipe. The present comparisons identify differences among model/training configurations; they do not isolate whether balancing, synthetic examples or augmentation itself caused an improvement.

## Experiment design and selection

The source annotations have fourteen classes, mapped to seven hazard groups: `battery`, `coin_battery`, `coin`, `small_object`, `pen`, `sharp`, and `medicine`. The prepared split contains **11,996 training, 2,570 validation and 2,568 test images**, with **20,184 / 4,370 / 4,173 objects** respectively. Photo-group assignments stay within one split. Runs were executed separately, with matching exported split membership.

All six detectors use the same prepared data, balancing-plus-extras recipe and augmentation strengths. Training uses AdamW, a cosine schedule, seed 42, batch 8, maximum 40 epochs and patience 15. The baseline learning rate is 0.0005. The balanced training recipe contains 12,352 original-image entries (including repetitions), 600 background images and 1,500 object-paste images. Repetitions are not new independent images.

<table align="center">
  <thead>
    <tr><th align="center">Run</th><th align="center">Main change</th><th align="center">Input</th><th align="center">Primary reference</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Baseline</td><td align="center">YOLOv8n, learning rate 0.0005</td><td align="center">640</td><td align="center">Reference</td></tr>
    <tr><td align="center">Exp1</td><td align="center">Learning rate 0.0001</td><td align="center">640</td><td align="center">Baseline</td></tr>
    <tr><td align="center">Exp2</td><td align="center">YOLOv8n at higher resolution</td><td align="center">960</td><td align="center">Baseline</td></tr>
    <tr><td align="center">Exp3</td><td align="center">Larger YOLOv8s model</td><td align="center">640</td><td align="center">Baseline</td></tr>
    <tr><td align="center">Exp4</td><td align="center">YOLOv8s-P2 finer prediction grid</td><td align="center">640</td><td align="center">Exp3</td></tr>
    <tr><td align="center">Exp5</td><td align="center">Higher-resolution YOLOv8s-P2</td><td align="center">960</td><td align="center">Exp4</td></tr>
    <tr><td align="center">Exp6</td><td align="center">YOLOv8n crop classifier, learning rate 0.001, 25 epochs</td><td align="center">96-pixel crops</td><td align="center">Selected detector alone versus detector + classifier</td></tr>
  </tbody>
</table>

P2 uses compatible pretrained-weight transfer and has additional layers requiring training. Comparisons assess the implemented configurations under the shared budget, not architecture in isolation.

**Selection protocol:** choose the detector by clean-validation mAP@50–95. On validation, choose the confidence threshold with fewest false boxes per image among tested thresholds meeting overall recall >= 0.85. Exp3 with confidence 0.30 meets that validation target. Classifier rejection thresholds are also evaluated on validation; none meets the target, so 0.20 is provisional. Test data assess the fixed configurations, not a new threshold search.

## Detector results

All numbers below are **mAP@50–95**, higher is better.

<table align="center">
  <thead>
    <tr><th align="center">Model</th><th align="center">Validation</th><th align="center">Validation stress</th><th align="center">Test</th><th align="center">Test stress</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Baseline</td><td align="center">0.6032</td><td align="center">0.5585</td><td align="center">0.6052</td><td align="center">0.5662</td></tr>
    <tr><td align="center">Exp1</td><td align="center">0.5836</td><td align="center">0.5526</td><td align="center">0.5996</td><td align="center">0.5565</td></tr>
    <tr><td align="center">Exp2</td><td align="center">0.4909</td><td align="center">0.5364</td><td align="center">0.4893</td><td align="center">0.5437</td></tr>
    <tr><td align="center"><strong>Exp3</strong></td><td align="center"><strong>0.6441</strong></td><td align="center"><strong>0.5945</strong></td><td align="center"><strong>0.6549</strong></td><td align="center"><strong>0.6080</strong></td></tr>
    <tr><td align="center">Exp4</td><td align="center">0.6279</td><td align="center">0.5880</td><td align="center">0.6382</td><td align="center">0.5929</td></tr>
    <tr><td align="center">Exp5</td><td align="center">0.5248</td><td align="center">0.5405</td><td align="center">0.5193</td><td align="center">0.5518</td></tr>
  </tbody>
</table>

Source: [final model comparison](../results/final_test/final_model_comparison.csv). Increasing model capacity is the only tested change improving this metric against its intended reference across all four sets. Exp3's clean-test gain over baseline is **4.97 percentage points**. P2 at 640 is second-best overall; higher resolution and lower learning rate do not improve aggregate mAP under this recipe.

mAP is supplied by YOLO's evaluator. The following operating-point results use separate class-aware, one-to-one matching at IoU 0.50 and detector confidence 0.30. These values need not equal YOLO's automatically reported precision/recall during training validation.

## What each experiment establishes

### Baseline and Experiment 1: learning-rate sensitivity

The baseline establishes what the smaller YOLOv8n can achieve with the shared recipe. Experiment 1 reduces only its initial learning rate from 0.0005 to 0.0001. Its training losses still decline and validation performance stabilises, so the experiment is a completed learning run rather than a training failure. However, the small validation recall gain is accompanied by lower precision and mAP. Clean-test mAP declines from **0.6052 to 0.5996**. Under this fixed budget, slower initial updates do not yield the desired detection benefit; the experiment does not show what would happen with a separately tuned longer schedule.

### Experiment 2: more input pixels are not sufficient

YOLOv8n is trained and evaluated at 960 rather than 640 pixels. The complete higher-resolution configuration is being compared: it changes both training and inference scale. Clean-test mAP falls to **0.4893**, approximately **11.59 percentage points below the baseline**. This is evidence against this particular 960-pixel recipe, not proof that increased resolution is intrinsically harmful. Resizing also cannot recreate detail missing in the captured image. Changes to learning schedule, resource budget or data quality would need new experiments, not an explanation inferred from the current curves.

### Experiment 3: model capacity provides the clearest gain

Replacing YOLOv8n with YOLOv8s while keeping 640-pixel input improves mAP against the baseline on all four sets. Test mAP rises to **0.6549** and test-stress mAP to **0.6080**, gains of **4.97 and 4.18 percentage points** respectively. This consistency supports the validation-selected configuration. It is still a single run per model, so the study does not establish a distribution of performance across random seeds or a statistically significant advantage.

### Experiments 4 and 5: a small-object-oriented design still needs evidence

Experiment 4 introduces the P2 prediction grid with YOLOv8s at 640 pixels. Its overall scores are second-best, but below plain YOLOv8s: **0.6382 test mAP versus 0.6549**. Some individual class/bin measurements can favour P2, yet this does not establish an overall benefit or justify selecting it against the predefined criterion. Compatible pretrained layers are transferred and additional layers require training, so architecture and its implemented training initialisation must be considered together.

Experiment 5 raises P2 input resolution to 960 and obtains **0.5193 test mAP**, a decline of **11.89 percentage points** relative to Experiment 4. The two higher-resolution experiments therefore do not support increasing resolution as the next deployment step under this recipe. The more useful next investigation is to examine missing detail, annotations and representative small-object data before assuming an input-size increase will solve the problem.

### Why selection happens before test evaluation

Exp3 is selected on clean-validation mAP, then its confidence threshold is selected using validation predictions. At confidence **0.30**, validation recall is approximately **0.866** and false boxes per image are **0.234**. Raising the threshold to 0.40 gives fewer false boxes but recall falls to approximately 0.829, below the project's 0.85 target. This makes 0.30 a transparent operating trade-off, not an arbitrary default or a threshold chosen after inspecting test outcomes.

The target is a project constraint, not a validated safety standard. It applies to aggregate validation recall. It cannot guarantee recall for an individual class, a moving camera or an unseen household scene. The test results assess how that fixed decision transfers.

![Overall mAP comparison](../figures/model_training/evaluation/final_model_comparison.png)

The chart visualises the recorded comparison. Some 960-pixel models score higher on the transformed set than the clean set; altered scale/composition and annotation filtering mean this is not evidence that degradation generally improves detection.

## Why Experiment 6 is not retained

The classifier's **96.63% accuracy** measures labelling of supplied validation crops. It is not full-image detection mAP and cannot be directly compared with Exp3's mAP. A classifier cannot recover hazards absent from detector proposals.

<table align="center">
  <thead>
    <tr><th align="center">Evaluation</th><th align="center">Method</th><th align="center">Precision</th><th align="center">Recall</th><th align="center">F1</th><th align="center">FP</th><th align="center">FN</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Clean test</td><td align="center">Exp3 alone</td><td align="center">0.872</td><td align="center">0.888</td><td align="center">0.879</td><td align="center">546</td><td align="center">469</td></tr>
    <tr><td align="center">Clean test</td><td align="center">Exp3 + classifier</td><td align="center">0.863</td><td align="center">0.861</td><td align="center">0.862</td><td align="center">571</td><td align="center">578</td></tr>
    <tr><td align="center">Stress</td><td align="center">Exp3 alone</td><td align="center">0.904</td><td align="center">0.827</td><td align="center">0.864</td><td align="center">366</td><td align="center">721</td></tr>
    <tr><td align="center">Stress</td><td align="center">Exp3 + classifier</td><td align="center">0.791</td><td align="center">0.595</td><td align="center">0.679</td><td align="center">656</td><td align="center">1691</td></tr>
  </tbody>
</table>

Source: [pipeline comparison](../results/final_test/detector_pipeline_comparison.csv). Adding the classifier introduces **109 additional missed objects and 25 additional false detections on clean test**, and **970 additional misses and 290 additional false detections under stress**. It reclassifies proposals as well as rejecting them; changing a correct class to an incorrect one can create both an FP and an FN. Detector-only operation is supported under these measured settings. Crop/proposal distribution differences are plausible contributors but are not proven causes. Other detector–classifier pairings were not exhaustively tested.

### What the crop classifier actually measures

The classifier learns from crops around ground-truth boxes, with a background class and crop-level augmentation. The detector pipeline instead supplies predicted boxes, which may be off-centre, include neighbouring items or contain little usable target detail. Strong accuracy on supplied crops therefore addresses one subproblem, not whether the complete system finds and correctly labels every object in a scene.

The evaluated pipeline keeps the detector's boxes and scores, replaces accepted class labels with the classifier's predicted group, and rejects proposals classified as background or below the classifier threshold. It has no mechanism to create a missing detector proposal or repair box localisation. Its rejection and reclassification errors can outweigh corrections.

No tested classifier threshold met the validation recall target. The displayed 0.850 recall at thresholds 0.20 and 0.35 is rounded: **3713 / 4370 is approximately 0.84966**. The provisional 0.20 threshold was nevertheless carried into the documented test comparison to measure the pipeline, not presented as meeting the requirement. Clean-test performance is relatively close to the detector alone, but both false detections and misses move in the wrong direction; stress exposes a substantially larger loss. This is why Exp6 is retained as a documented negative result rather than silently omitted or selected on its headline accuracy.

![Detector versus classifier pipeline](../figures/model_training/evaluation/detector_pipeline_comparison.png)

## Failure analysis and practical consequences

### Misses persist despite strong aggregate mAP

At confidence 0.30, Exp3 detects **3,704 of 4,173 objects** on clean test and misses 469. Under stress it detects 3,452 and misses 721. Precision increases under stress while recall decreases, illustrating why a higher precision alone cannot be interpreted as a safer detector. The system issues fewer incorrect boxes but also recovers fewer actual hazards.

The object-outcome analysis identifies **214 low-confidence and 132 no-useful-proposal cases on clean test**, rising to **359 and 237 under stress**. Low-confidence cases suggest a threshold trade-off, while absence of a useful cached proposal cannot necessarily be repaired by lowering the final threshold. Wrong-class and poor-localisation examples show additional patterns. These are operational categories under matching rules, not a diagnosis of the physical cause of each error; predictions below the cache floor of 0.05 are unavailable.

### Difficult classes need targeted attention

Pen recall is the lowest of the seven groups for the selected model, decreasing from **0.755 to 0.702** under stress. Sharp objects and batteries also remain weaker than several other groups. More general training data alone may not resolve the patterns; the next data collection should include the relevant views, occlusions, clutter and confusing backgrounds, with annotation review. The existing figures identify priorities, not proof that a particular augmentation or class-weight change will fix them.

### Small-object findings must retain their sample counts

For Exp3, clean-test recall is **0.649 at 16–32 input pixels**, **0.797 at 32–64**, and **0.917 at 64+**, using the shorter bounding-box side. Only six clean-test objects fall in the 8–16-pixel group and three under 8, so apparent extremes there are uncertain. Pixel groups are model-input measurements, not centimetres or camera distances. The same original object can move between bins at 640 versus 960; those groups are not matched populations across resolutions.

The one-factor condition study uses the same **150 source images**, largely retaining 236 objects (235 after rotation). Shrinking to 0.25 gives the largest recall loss: Exp3 falls from **0.835 to 0.682** and false boxes per image rise from approximately **0.113 to 0.220**. Some moderate transformations produce equal or higher scores on this subset. This supports a specific limitation around severe loss of apparent size, not a universal claim that every brightness or blur change reduces accuracy.

![Recall by object size](../figures/model_training/evaluation/test_pixel_size_recall.png)

## Findings and limitations

The results support a **relative choice among the configurations tested**, but do not establish that the resulting application reliably recognises every hazard or correctly warns a caregiver. The limitations concern three different things: what the detector still misses, how much confidence can be placed in the experiment, and what has not yet been established in the complete camera application.

### Detection limitations: what the measurements show

The main operational weakness is missed hazards when usable visual detail is limited. Exp3's clean-test recall is **0.8876**, leaving **469 missed objects**; stress recall falls to **0.8272**, leaving **721**. The reduction persists despite higher stress precision. This means an output containing fewer incorrect boxes can still omit more actual hazards. The validation recall requirement is an operating-point rule, not a guarantee that a scene has been adequately inspected.

The failures are unevenly distributed. Pen, sharp and battery stress recall are approximately **0.702, 0.748 and 0.767**, respectively. The selected model also has substantially lower clean-test recall at 16–32 input pixels (**0.649**) than at 64+ pixels (**0.917**). Consequently, the aggregate score cannot be used to promise the same recovery for every hazard category or apparent object size. These findings motivate targeted data and error review rather than claiming that model selection alone resolves the prototype's requirements.

<table align="center">
  <thead>
    <tr><th align="center">Observed limitation</th><th align="center">Supporting measurement</th><th align="center">Practical implication</th><th align="center">Appropriate next check</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Reduced recovery under combined degradation</td><td align="center">Exp3 recall: 0.888 clean test versus 0.827 stress</td><td align="center">The validation threshold does not ensure comparable recovery in degraded scenes</td><td align="center">Evaluate representative camera footage with recorded conditions</td></tr>
    <tr><td align="center">Uneven class recovery</td><td align="center">Pen / sharp / battery stress recall: 0.702 / 0.748 / 0.767</td><td align="center">Some hazard groups remain less reliably recovered than the average</td><td align="center">Review class-specific false negatives and confusing annotations</td></tr>
    <tr><td align="center">Sensitivity to apparent size</td><td align="center">Recall 0.649 at 16–32 input pixels versus 0.917 at 64+</td><td align="center">Small objects can remain undetected even when larger examples are recognised</td><td align="center">Collect representative small-object views and retain support counts</td></tr>
    <tr><td align="center">Severe shrinking affects both measures</td><td align="center">On 150 images, scale 0.25 reduces recall 0.835 to 0.682 and raises FP/image about 0.113 to 0.220</td><td align="center">This condition produces both more misses and more false boxes</td><td align="center">Check controlled scale reduction against genuine camera-distance cases</td></tr>
    <tr><td align="center">Crop-stage performance does not transfer</td><td align="center">Exp6 adds 109 clean-test misses and 970 stress misses</td><td align="center">A strong crop-accuracy headline is insufficient for selecting a pipeline</td><td align="center">Inspect detector-generated crops and reclassification/rejection errors</td></tr>
  </tbody>
</table>

### Experimental limitations: how broadly the results can be interpreted

**Dataset independence is limited.** Public-source and self-collected images are split by detected photo groups, and exact-copy checks reduce obvious overlap. Nevertheless, visual hashing cannot identify every related scene, and four large candidate buckets were skipped. In addition, these photographs informed earlier project development. The final split is excluded from training in this rerun, but it is not a completely unseen benchmark for the project as a whole. These safeguards reduce leakage risk; they do not justify claiming complete independence.

**Annotation validity differs from annotation quality.** Readability, coordinate and class-ID checks confirm that the data can be processed. They do not prove that every visible object has a correct, tightly fitting box. The 37 folder/class mismatches were retained for review because images may legitimately contain multiple classes. The available record does not establish a complete manual correction audit. Background examples also depend on the absence of unlabelled hazards, while pasted objects can introduce unnatural edges or retained background.

**The comparisons are single-run, recipe-specific evidence.** Each detector configuration uses one seed and the fixed training budget; no repeated-run confidence intervals are available. All detectors share balancing, extras and augmentation, so the study does not isolate the contribution of any one of those data choices. The P2 configuration also uses transferred compatible layers and newly initialised additional layers. The evidence supports the tested configurations, not a universal conclusion that higher resolution, P2 or classification stages are inferior.

**Synthetic conditions and size bins have different populations.** Stress transformations change appearance and geometry, and visibility filtering can remove annotations. The controlled-condition analysis uses only 150 source images. Pixel bins contain different objects across 640- and 960-pixel inputs, and the smallest clean-test groups contain only three and six objects. These details constrain comparisons: a higher stress or small-bin score is not proof that degradation helps, and a zero recall in a tiny group is not a universal detection-size boundary.

### Application limitations: what the detector study does not validate

The experiment measures annotated-image detection, not the correctness of spatial risk scores or parent notifications. Physical camera-to-object range, measured lighting limits, true 3D viewpoint changes, end-to-end delay and temporal alert behaviour have not been established by these tables. An available image-plane/depth heuristic is not an independently validated reachability or safety model.

<table align="center">
  <thead>
    <tr><th align="center">Unresolved application issue</th><th align="center">Why the current evidence is insufficient</th><th align="center">Required before making the corresponding claim</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Physical detection range</td><td align="center">Input pixels do not specify distance in metres; object size and camera optics vary</td><td align="center">Record actual object dimensions, distances, camera setup and successes/failures</td></tr>
    <tr><td align="center">Empty-scene / caregiver false alarms</td><td align="center">Evaluation contains no labelled-empty scenes; false boxes are not notifications</td><td align="center">Label empty and realistic background scenes, then evaluate complete alert events</td></tr>
    <tr><td align="center">Seven-group spatial integration</td><td align="center">All seven groups are recognised, but group risk weights lack real-world validation; merged <code>pen</code> still uses a legacy weight</td><td align="center">Agree on grouped risk policies and test every final label through the complete app</td></tr>
    <tr><td align="center">Live depth and timing</td><td align="center">Frame-relative depth, freshness fallback and score persistence affect the displayed result</td><td align="center">Measure detection latency, depth age, score clearance and notification delay on deployment hardware</td></tr>
    <tr><td align="center">Unseen-household generalisation</td><td align="center">Synthetic variants reuse the photographed objects/scenes</td><td align="center">Evaluate a separately collected, representative camera set before tuning against it</td></tr>
  </tbody>
</table>

The absent empty-scene alarm rate is **unavailable, not zero**. Similarly, an app-level SAFE or no-detection output cannot establish that there is no hazard: it can reflect a missing detection, absent person reference, filtered class, stale/missing depth or the selected scoring policy. These are reasons to keep the prototype's scope explicit, not reasons to hide the measured improvements.

Evidence: [class results](../results/final_test/selected_model_classes.csv), [size results](../results/final_test/recall_by_pixel_size.csv), [controlled conditions](../results/final_test/controlled_condition_results.csv), [pipeline results](../results/final_test/detector_pipeline_comparison.csv), and notebook Sections 8–10. Spatial behaviour is documented separately in the [spatial guide](SPATIAL_ANALYSIS.md); it is not a detector benchmark result.

## Future work and application handoff

1. Review difficult-class annotations and confusing/missed examples; collect representative live-camera data and independently labelled empty scenes.
2. Evaluate the same objects under recorded camera distance, lighting, placement and capture resolution; measure end-to-end delay and alert outcomes.
3. Repeat key configurations across seeds before treating small differences as conclusive.
4. Investigate classifier proposal crops, rejected true hazards and reclassification errors before further pipeline use.
5. Use `final_best.pt` with input 640 and confidence 0.30. Preserve the seven model class names, keep the Exp6 crop classifier out of the recommended path, and validate the grouped spatial weights, particularly the combined pen/pen-cap policy.

The current spatial mapping includes all seven final groups. The remaining issue is validation of the risk policy, rather than missing group entries. See [implemented weights and remaining checks](SPATIAL_ANALYSIS.md#seven-class-integration-and-remaining-checks).

## Conclusion

The final study selects **Experiment 3: YOLOv8s at 640-pixel input and confidence 0.30, without the crop classifier**, following the clean-validation selection rule and validation threshold procedure. It leads the six standalone detectors in mAP@50–95 on clean validation, validation stress, clean test and test stress. Its clean-test score of **0.6549** improves on the baseline's **0.6052**, while test-stress mAP improves from **0.5662 to 0.6080**. The supported result is therefore a consistent improvement within the prepared dataset and shared training recipe.

The experiments also identify which intuitively attractive changes did not help under the tested conditions. The lower learning rate and both 960-pixel configurations reduce aggregate mAP against their intended references. P2 at 640 is competitive but below plain YOLOv8s overall. Most significantly, **96.63% standalone crop accuracy does not translate into a better full detection pipeline**: Exp6 increases false detections and missed hazards, particularly under stress. Recording these outcomes provides a reasoned deployment choice rather than selecting whichever model has the largest headline metric.

<table align="center">
  <thead>
    <tr><th align="center">Decision</th><th align="center">Evidence</th><th align="center">Boundary of the conclusion</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">Retain Exp3 as the selected detector</td><td align="center">Highest mAP across all four reported sets; 0.888 clean-test recall at the fixed operating threshold</td><td align="center">Best among these tested runs, not a guarantee for unseen camera scenes</td></tr>
    <tr><td align="center">Keep Exp6 out of the recommended configuration</td><td align="center">More FP and FN on clean and stress test; no tested classifier threshold meets validation recall target</td><td align="center">Applies to this classifier, proposal pipeline and settings</td></tr>
    <tr><td align="center">Prioritise small-object and difficult-class investigation</td><td align="center">Size-dependent recall, weak pen/sharp/battery recovery and severe-scale failures</td><td align="center">Specific causes and remedies still need targeted validation</td></tr>
    <tr><td align="center">Complete spatial and live-camera validation before application claims</td><td align="center">Grouped-label weights remain unvalidated; no physical-range or end-to-end alert benchmark</td><td align="center">Dataset detection scores do not certify the complete monitoring system</td></tr>
  </tbody>
</table>

The contribution of this work is a traceable preparation and evaluation record, a defensible detector selection, and an evidence-based account of remaining failures. The next milestone is **validation of grouped risk policies and representative camera/alert testing**, rather than treating the higher mAP alone as proof that the monitoring problem is solved. A missing detection must never be presented as proof of a safe scene.

## Reproduction and records

The root notebook is a compilation of separate sessions with preserved outputs, not proof of one uninterrupted Run All. Reproduction requires the raw dataset for training and the saved bundle for final evaluation. The bundle loader reloads historical checkpoints; retraining earlier cells does not automatically replace that bundle.

The final evaluation log records **Python 3.13.16, Ultralytics 8.4.198, Albumentations 2.0.8, Torch 2.11.0+cu130 and Tesla T4**. The allocated accelerator was T4 x2 but code used device 0. This differs from the local app's Python/dependency context; the root requirements file is preserved, not claimed to be an exact Kaggle lockfile.

Per-run histories and original `args.yaml` files are under [training_history](../results/training_history/). Arguments can contain historical Kaggle paths or framework-specific YAML tags and are evidence, not portable configs to load blindly. [Split membership](../results/split_membership.csv) retains exported filename, category, split and photo-group ID, without image data or source-path columns.

The baseline source session was stopped after completing baseline and Exp1; its partial Exp2 is excluded. Exp2–6 records use their completed standalone sessions. See the [artifact manifest](../configs/model_training/artifact_manifest.json) for session references, hashes and exact local records copied into this repo.

## Artifact locations and verification

The repository contains the result tables, figures, training histories and split record needed to follow the documented findings. Full datasets and the other experiment checkpoints are stored separately. The selected checkpoint is included at `weights/final_best.pt`; source-session records and checkpoint checksums remain in the [artifact manifest](../configs/model_training/artifact_manifest.json).

The README lists the [main public-image sources](../README.md#1-public-datasets-selected-sources) and a full [reference list](../README.md#references). Remaining images came from other public datasets whose links were not saved during collection, online marketplace listings (e.g. Carousell) and about 450 team-taken photos. These unrecorded sources are acknowledged as a group rather than individually.

To verify a downloaded model on Windows:

```powershell
Get-FileHash -Algorithm SHA256 .\weights\final_best.pt
```

Compare the result with the `final_best.pt` entry in the manifest. Existing `weights/`, runtime databases and secrets stay ignored. The executed notebook is around 38 MB, above GitHub's browser upload limit; push it using Git or distribute the executed file separately. Do not clear its saved results just to upload it.

## References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/): model family.
- [Ultralytics validation](https://docs.ultralytics.com/modes/val/): mAP and evaluator documentation.
- [Albumentations](https://albumentations.ai/docs/): image and bounding-box transformations. Current documentation may describe newer releases; recorded versions above govern this experiment.
- Internal evidence: notebook Sections 6–10 and the linked CSVs/training histories.
- Dataset sources and full APA references: see the [README references](../README.md#references).
