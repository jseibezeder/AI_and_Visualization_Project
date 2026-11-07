# Experiment 5 progress

Goal: qualitative counterfactual decision-boundary exploration along horizontal translation and rotation, selecting eight informative images from Experiments 1–4 before new inference.

Plan: persist provenance-based selection within the existing 48-image explanation set; reuse original-resolution geometry semantics; coarse grid then global fine guard grid in both signed directions; refine every observed top-1 transition from either model to the final tick resolution. Do not assume a unique or monotonic boundary. Report first observed switches, all observed transitions and returns to baseline. State finite-grid/range and uint8/interpolation limitations.

Validation: exact-byte equivalence to existing geometric transforms at integer translations/rotation levels; fresh-baseline equivalence against trusted predictions; common-level equivalence to Experiment 2; deterministic shared caches and explicit environment guards; synthetic non-monotonic cases and numeric output validation.

Completed: skript/counterfactual.py, skript/test_counterfactual.py, skript/audit_counterfactual.py, results/counterfactual/ documentation/artifacts added. Six focused tests pass; both scripts compile. Selection fixed before E5 inference: #65/#906 (E1), #809/#2494 (E2), #105/#432 (E3), #194/#2878 (E4), all within the existing 48-image explanation set.

Search: full signed rotation ±30 degrees, global scan 0.25 degrees, observed-transition refinement 0.025 degrees; horizontal translation ±3 pixels, global scan 0.125 pixels, refinement 0.0125 pixels. No monotonicity assumed; finite-grid caveats explicit. Main GPU run succeeded in 59.51 seconds. Per model 3,247 saved tick entries; 3,231 new unique intervention inputs plus eight originals and sixteen repeat-audit evaluations. 137 observed transitions, 57 returns to baseline, 24/64 signed rays with no observed switch. Example #2878 ResNet positive rotation: class 5->3 at 5.025 degrees, back to 5 at 5.05, to 3 again at 5.075. ViT stays correct on this example. Counterexample #809: ViT switches at -0.1 horizontal pixels, ResNet at -0.85 (but ResNet originally wrong).

Validation: original decisions match trusted exports on all eight images; 72 E2 geometry comparisons pass, maximum probability error 3.55e-6. Additional endpoint audit repeated all 255 unique transition endpoints three times for both models (1,530 extra evaluations): zero argmax mismatches, zero within-audit repeat variation, max saved-probability errors 5.07e-7 / 1.17e-9. Closest top-two margins 1.43e-4 / 2.04e-4. Completed resume validates all caches and regenerates reports with no search inference.

Final artifacts: README.md, FINDINGS.md, summary.json, protocol/config, selection/manifest, four main CSVs, validation tables/audits, 34 final figures + selected-image pilot, compact results ZIP including sources/tests and user-facing artifacts. Internal input/prediction caches and logs retained separately. Main run evidence: logs/run_20261107T111434Z/run-report.json and run.log. Primary protocol 0558573cfb3a1ec92cc86819f645d57971e6ed19e90e1e3ad5a978be946ddae6.

Prior tracked files and E4 scripts/results unchanged; all E5 files are new. HeuVik base f07eae3; no commit/push. No disposable task artifacts outside normal self-cleaning test temporary directories.
