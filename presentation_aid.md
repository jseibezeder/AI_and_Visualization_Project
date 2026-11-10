# BloodMNIST presenter aid

Open `presentation_view.ipynb` for screen sharing. It has no code cells and needs no execution. Keep these points separate from the notebook.

## Opening context

- We compare ResNet18, ResNet50 and ViT on the same 3,421 TEST images.
- These are trained models with different preprocessing, not a controlled comparison of architecture alone.

## High accuracy leaves confident mistakes

- Accuracy is between 95.73% and 96.99%, yet the models make 52 to 79 confidently wrong predictions each.
- Here, high confidence means a top-class probability of at least 90%. A high reported probability does not guarantee a correct label.

## Even model agreement can be confidently wrong

- The ResNets share 36 high-confidence errors. Each CNN shares 22 or 23 with ViT, but these pair counts overlap.
- All three give the same wrong label on 18 images. Agreement would not correct these cases.

## Many confident mistakes involve immature granulocytes

- IG is the true or predicted class in 58/70 ResNet18 errors, 61/79 ResNet50 errors and 45/52 ViT errors within the high-confidence subset.
- Only part of that group confuses IG with neutrophils. The lighter segments show IG errors involving other classes.

## There is no universal robustness winner

- At brightness ×0.8, ViT retains 94.4% accuracy, compared with 46.9% and 50.1% for the ResNets.
- At blur radius 1.5, all three fall to roughly 52% to 55%. The advantage depends on the input change.

## Preprocessing changes the robustness comparison

- In the separate 96-image ResNet18 control, the noise-induced accuracy loss is 47.9 percentage points with nearest-neighbor resizing and 7.3 with bilinear.
- Clean accuracy also drops from 95.8% to 80.2%. This is a tradeoff, not an unambiguous improvement, and it prevents an architecture-only explanation of the noise comparison.

## IG mistakes show patterns, not proven causes

- For IG images predicted as neutrophils, the matched comparisons have brighter, less saturated point estimates and lower boundary-gradient fractions. The intervals show uncertainty, especially for ViT brightness and saturation.
- The reverse direction is less consistent, and the tested edits did not establish selective correction. These are appearance associations in selected images, not identified biological causes.

The image pair illustrates one ResNet18 error and its same-class, similar-confidence control. The plotted intervals summarize separate matched cohorts for each model. Boundary-gradient fraction measures image-edge gradient energy, not nuclear segmentation.

## What high accuracy leaves out

- Confident failures can be shared across models.
- Robustness depends on input changes and preprocessing.
- IG and neutrophil errors show unexplained appearance patterns.

The deck is designed for about 2:45 without the opening context, or about 2:55 with it. These are estimates, not a timed rehearsal. Keep `presentation.ipynb` if you need the executable version; use `presentation_view.ipynb` when presenting.
