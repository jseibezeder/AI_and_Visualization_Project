# AI_and_Visualization_Project

Project github for the course **UE AI and Visualization**.

Initial assets for visualizing BloodMNIST and comparing a pretrained CNNs with a
Vision Transformer. The original dataset and both models are included.

## Dataset and models

- [BloodMNIST data, labels, splits and attribution](data/README.md): 17,092 images,
  eight classes, fixed train/validation/test splits.
- [BloodMNIST ResNet-18 and ResNet-50] : official 28 x 28 checkpoints and
  matching small-image architecture.
- [BloodMNIST ViT]: complete local Hugging Face inference
  files and loading/preprocessing example; input is resized to 224 x 224.

Both models share the official class order. Their preprocessing differs.

## Collaboration

The initial assets are set up directly on `main`. The AI assisted group did their work in the `HeuVik` branch, the AI-unassisted team in `TheJak` branch. The results of the no-Gen-AI team are located in the [Visualization file](visualization/visuals.ipynb) on the `TheJak` branch.

## Sources and licenses

- [MedMNIST dataset](https://zenodo.org/records/10519652) and
  [project](https://github.com/MedMNIST/MedMNIST): BloodMNIST, CC BY 4.0.
- [Official experiment code](https://github.com/MedMNIST/experiments): Apache-2.0;
  [pretrained weights](https://zenodo.org/records/7782114): CC BY 4.0.
- [KiViDrag/ViT_bloodmnist_std_15](https://huggingface.co/KiViDrag/ViT_bloodmnist_std_15):
  Apache-2.0 as declared by its publisher.

Upstream files are unmodified. Attribution and applicable license notices are
kept with each asset; no single replacement license is imposed on them.
This project is for research and teaching, not clinical diagnosis.
