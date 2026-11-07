# Experiment 4 progress

Goal: color dependence on the exact Experiment 3 48-image explanation set; approximate cell/periphery interventions on a visually reviewed subset with area/intensity controls.

Constraints: preserve prior experiments and weights; no training/downloads; batched local inference; cached uint8 inputs shared by both models; frozen protocol and equivalence audits; explicitly separate mild changes from feature ablations. Masks are approximate agent-reviewed geometric regions, not expert segmentation.

Plan:
1. Inspect original images and select/review approximate masks without using new prediction outcomes.
2. Add skript/color_background.py, focused tests, results/color_background/README.md and protocol/config/manifest.
3. Validate transformations, pixel budgets, cache integrity and baseline equivalence; run deterministic GPU inference.
4. Generate 4A matrices/example galleries, 4B matched-region comparisons, metrics and limitations; inspect results and summarize.

Completed: skript/color_background.py and skript/test_color_background.py added; five focused tests pass. Pilots and result figures inspected. All 16 masks individually accepted as approximate assistant-reviewed regions with per-image limitations; no human/clinical approval claimed. Batched CUDA run succeeded in 41.28 seconds (RTX 4000, PyTorch cuda:0): 2,144 intervention + 96 original + 32 repeat-audit evaluations. Original decisions exactly match trusted exports on all 48 images for both models; maximum probability error 3.76e-6 ResNet / 1.67e-6 ViT. Resume validated all caches and regenerated reports without inference. Regional outside-mask invariance and exact 64-pixel matched budgets independently confirmed.

Findings: 106 mild one-model-only flip cases across 31 images (92 ResNet-only, 14 ViT-only). Red x0.95 flips 8/48 ResNet versus 2/48 ViT; #194 changes correct neutrophil->monocyte only for ResNet. Both models have larger mean central-region responses under matched controls, with individual exceptions. Sensitivity is not correctness; set is disagreement-enriched.

Final documentation: README.md (protocol), FINDINGS.md (results/limitations), summary.json, six CSV tables, nine final figures, three pilot figures, exact subset/mask manifests, and per-model reproducibility audits. Run evidence: logs/run_20261107T105816Z/run-report.json and run.log. Compact results ZIP includes all user-facing artifacts and source/tests, excluding internal caches/logs.

Repository: HeuVik at f07eae3; only new Experiment 4 files; previous tracked files unchanged. No commit or push made. No disposable task artifacts created outside standard self-cleaning test temporary directories.
