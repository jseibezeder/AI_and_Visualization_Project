---
library_name: transformers
license: apache-2.0
base_model: google/vit-base-patch16-224
tags:
- generated_from_trainer
datasets:
- medmnist-v2
metrics:
- accuracy
- f1
model-index:
- name: ViT_bloodmnist_std_15
  results:
  - task:
      name: Image Classification
      type: image-classification
    dataset:
      name: medmnist-v2
      type: medmnist-v2
      config: bloodmnist
      split: validation
      args: bloodmnist
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9698918444899153
    - name: F1
      type: f1
      value: 0.9662097792355901
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ViT_bloodmnist_std_15

This model is a fine-tuned version of [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) on the medmnist-v2 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1123
- Accuracy: 0.9699
- F1: 0.9662

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch  | Step  | Validation Loss | Accuracy | F1     |
|:-------------:|:------:|:-----:|:---------------:|:--------:|:------:|
| 0.4602        | 0.0595 | 200   | 0.2846          | 0.9019   | 0.8995 |
| 0.19          | 0.1189 | 400   | 0.2421          | 0.9118   | 0.9056 |
| 0.1612        | 0.1784 | 600   | 0.1811          | 0.9299   | 0.9222 |
| 0.1443        | 0.2378 | 800   | 0.1166          | 0.9556   | 0.9491 |
| 0.1105        | 0.2973 | 1000  | 0.1091          | 0.9603   | 0.9560 |
| 0.0996        | 0.3567 | 1200  | 0.1631          | 0.9433   | 0.9292 |
| 0.0913        | 0.4162 | 1400  | 0.1724          | 0.9393   | 0.9285 |
| 0.0708        | 0.4756 | 1600  | 0.1206          | 0.9591   | 0.9540 |
| 0.0829        | 0.5351 | 1800  | 0.0888          | 0.9685   | 0.9625 |
| 0.0624        | 0.5945 | 2000  | 0.1379          | 0.9579   | 0.9519 |
| 0.0652        | 0.6540 | 2200  | 0.1158          | 0.9685   | 0.9667 |
| 0.0495        | 0.7134 | 2400  | 0.1169          | 0.9655   | 0.9642 |
| 0.0425        | 0.7729 | 2600  | 0.0944          | 0.9679   | 0.9643 |
| 0.0405        | 0.8323 | 2800  | 0.1280          | 0.9650   | 0.9605 |
| 0.0382        | 0.8918 | 3000  | 0.0762          | 0.9778   | 0.9755 |
| 0.0336        | 0.9512 | 3200  | 0.1064          | 0.9708   | 0.9697 |
| 0.0318        | 1.0107 | 3400  | 0.1001          | 0.9720   | 0.9682 |
| 0.0162        | 1.0702 | 3600  | 0.1018          | 0.9737   | 0.9720 |
| 0.0165        | 1.1296 | 3800  | 0.1431          | 0.9614   | 0.9537 |
| 0.0133        | 1.1891 | 4000  | 0.0808          | 0.9766   | 0.9736 |
| 0.0146        | 1.2485 | 4200  | 0.0912          | 0.9737   | 0.9707 |
| 0.0091        | 1.3080 | 4400  | 0.1006          | 0.9761   | 0.9747 |
| 0.0074        | 1.3674 | 4600  | 0.1114          | 0.9702   | 0.9680 |
| 0.0134        | 1.4269 | 4800  | 0.1200          | 0.9725   | 0.9705 |
| 0.012         | 1.4863 | 5000  | 0.1063          | 0.9720   | 0.9694 |
| 0.0099        | 1.5458 | 5200  | 0.1239          | 0.9690   | 0.9667 |
| 0.006         | 1.6052 | 5400  | 0.1308          | 0.9731   | 0.9677 |
| 0.0057        | 1.6647 | 5600  | 0.1479          | 0.9702   | 0.9682 |
| 0.0107        | 1.7241 | 5800  | 0.1194          | 0.9720   | 0.9684 |
| 0.0122        | 1.7836 | 6000  | 0.1083          | 0.9708   | 0.9691 |
| 0.0081        | 1.8430 | 6200  | 0.1087          | 0.9725   | 0.9690 |
| 0.0055        | 1.9025 | 6400  | 0.1063          | 0.9766   | 0.9731 |
| 0.0039        | 1.9620 | 6600  | 0.1530          | 0.9679   | 0.9631 |
| 0.0075        | 2.0214 | 6800  | 0.1052          | 0.9778   | 0.9764 |
| 0.0022        | 2.0809 | 7000  | 0.1340          | 0.9673   | 0.9628 |
| 0.0024        | 2.1403 | 7200  | 0.1034          | 0.9761   | 0.9742 |
| 0.0014        | 2.1998 | 7400  | 0.1039          | 0.9772   | 0.9751 |
| 0.0007        | 2.2592 | 7600  | 0.1032          | 0.9801   | 0.9792 |
| 0.0008        | 2.3187 | 7800  | 0.0984          | 0.9807   | 0.9797 |
| 0.0013        | 2.3781 | 8000  | 0.1034          | 0.9766   | 0.9752 |
| 0.0013        | 2.4376 | 8200  | 0.1049          | 0.9766   | 0.9749 |
| 0.0013        | 2.4970 | 8400  | 0.1006          | 0.9772   | 0.9756 |
| 0.0018        | 2.5565 | 8600  | 0.1157          | 0.9749   | 0.9703 |
| 0.0011        | 2.6159 | 8800  | 0.1049          | 0.9784   | 0.9779 |
| 0.0007        | 2.6754 | 9000  | 0.1167          | 0.9755   | 0.9721 |
| 0.0003        | 2.7348 | 9200  | 0.1058          | 0.9772   | 0.9746 |
| 0.0008        | 2.7943 | 9400  | 0.1049          | 0.9796   | 0.9767 |
| 0.0009        | 2.8537 | 9600  | 0.1084          | 0.9807   | 0.9787 |
| 0.0005        | 2.9132 | 9800  | 0.0999          | 0.9807   | 0.9787 |
| 0.0001        | 2.9727 | 10000 | 0.1001          | 0.9813   | 0.9796 |


### Framework versions

- Transformers 4.45.1
- Pytorch 2.4.0
- Datasets 3.0.1
- Tokenizers 0.20.0
