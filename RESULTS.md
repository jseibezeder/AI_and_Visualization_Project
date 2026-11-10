# Results and reading guide

## Run the saved analysis

From the repository root, install `requirements.txt` in an isolated Python
environment and select its kernel in JupyterLab. Run `git lfs pull` if checkpoint
files are still LFS pointers. Run the notebooks in this order:

1. [01_baseline.ipynb](01_baseline.ipynb) checks the three TEST baselines,
   image indices, labels, probabilities, logits and checkpoint/configuration
   hashes. The default run performs no inference or writes. Optional full-TEST
   checkpoint replay is off and verifies the saved predictions without replacing them.
2. [02_model_behavior.ipynb](02_model_behavior.ipynb) examines disagreement,
   confidence, input changes and matched error properties. It then compares IG
   and neutrophil misclassifications and summarizes the model-specific results.
3. [03_robustness.ipynb](03_robustness.ipynb) shows label-change rates and
   accuracy across the full TEST set under each transformation.

After setup, these runs use saved binary results for the 3,421 TEST images. They
need no training, new model queries, downloads or live service. The
[offline explorer](visualizations/transformation_explorer.html) includes all eight
transformations and ten levels, counts recovered and new errors, and links zoom
across models. The behavior notebook regenerates its data. Static results remain
readable without JavaScript.

## Supplements and archive

The main behavior analysis ends at its conclusions after 54 cells and 12 figures.
[Supplement A](02_model_behavior.ipynb#supplement-galleries) contains all 39
shared-error images and 18 confidence-selected exclusive examples.
[Supplement B](02_model_behavior.ipynb#supplement-roi) contains the optional
24-image approximate-region study, mask checks and unequal-dose limitations.
All evidence remains available.

Confusion heatmaps show counts only. The reliability diagram includes image
counts for all ten confidence bins and marks bins with fewer than 30 images.
This cutoff is a display choice, not a significance test. It does not change ECE.

[archived/analysis_history.ipynb](archived/analysis_history.ipynb) preserves earlier
code, outputs and interpretations, including superseded claims, training and
validation exports, entropy and PCA work. Do not Run All in it. The old
`archived/legacy_visualization/predictions.ipynb` redirects to the current
workflow so that its older ResNet50 result is not mixed with the reported one.
The paused editor and workbench are in `archived/workbench/`, with extra
dependencies in `archived/requirements.txt`. Neither is required for the analysis
or used as evidence.

The earlier central-occlusion comparison used broader matches and suggested about
10 percentage points of extra loss in HC errors. With the reported
confidence-caliper matches, the differences are +1.4, −1.2 and +3.3 points, and
every 95% interval includes zero. The archive also preserves the full notebook
from before the presentation edits.

## Findings

HC errors are wrong predictions with confidence of at least 90%. The model
comparison includes each checkpoint's preprocessing.

| Model and preprocessing | TEST accuracy | HC errors | Observed behavior |
|---|---:|---:|---|
| ResNet18 with nearest-neighbor resize to 224 | 96.61% | 70 | Brightness, color and noise alter predictions under the original preprocessing. IG has its highest within-class HC-error rate. |
| Supplied ResNet50 at native 28 | 95.73% | 79 | Largest ECE and strong rotation sensitivity. Rotation by +20° creates 668 errors and recovers 60. |
| Local ViT with bilinear resize to 224 | 96.99% | 52 | More stable under the tested original brightness and noise changes, but accuracy falls to 52.3% under the strongest blur. |

All three models share 18 confidently wrong images and predict the same wrong
labels on them. Confidence and agreement do not guarantee correctness.

Robustness depends on preprocessing as well as the checkpoint. Common resizing
narrows the noise-loss gap in the selected control but changes clean accuracy.
Different transformation ranges do not represent equal-strength changes.

Reducing saturation sends both CNN error/control cohorts toward erythroblast.
The confidence-matched occlusion comparison does not establish extra central
dependence in HC errors. The approximate-region study cannot establish anatomical
localization.

IG images misclassified as neutrophils show clearer appearance differences than
errors in the reverse direction. These are associations in a selected cohort,
not evidence of a nuclear mechanism, a label-error diagnosis or an intervention
that selectively corrects the errors.

## Limits and untested questions

The cohorts and questions were selected after inspecting TEST behavior. Bootstrap
intervals keep images and pairs together and average repeats and patches within
images. They condition on the selected cohort, exclude training, annotation and
matching uncertainty, and have no correction for multiple comparisons. Small
class groups can produce intervals with zero width. RGB measurements are
correlated, masks are approximate, intervention strengths differ, and synthetic
edits can invalidate the original labels.

A saturation intervention on the matched IG/neutrophil pairs was not run. The
broader saturation cohort cannot substitute for it. Original-resolution images,
IG subtype labels, expert review and comparisons that control training and
preprocessing would address separate causal questions. They are possible future
studies, not missing evidence for the completed descriptive analysis.
