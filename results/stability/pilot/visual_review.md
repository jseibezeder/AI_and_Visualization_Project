# Exploratory visual pilot

Reviewer: coding assistant (visual inspection of the eight-class pilot montages), not a clinical expert.

Reviewed rotation, translation, brightness, blur, Gaussian noise, and all three channel-scaling montages before starting transformed inference.

- Rotation and translation keep the sampled cell bodies recognizable, but interpolation, border fill, and occasional peripheral cropping are visible confounds.
- Brightness and RGB-channel changes preserve gross shape in the pilot. Brightness ×1.2 clips some bright background pixels.
- Blur σ=1.5 visibly suppresses fine texture and softens nuclear lobes; noise σ=0.1 substantially obscures fine details, especially in the platelet example. Gross cell regions remain visible in these pilot images. Retain these proposed levels as explicit **stress tests**, not verified label-preserving interventions.
- The 28×28 source already limits morphological detail. None of these judgments establishes diagnostic recognizability or label preservation; expert review is needed for that claim.

All proposed levels are retained for the exploratory sensitivity experiment. Correct-label retention at strong interventions must be read with the above limitations. No level was selected based on model outcomes.
