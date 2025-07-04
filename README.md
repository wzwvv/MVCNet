# MVCNet

This repository provides the official implementation of [**MVCNet: Multi-View Contrastive Network for Motor Imagery Classification**](https://arxiv.org/abs/2502.17482). The code of baseline models is in our another respository [**DBConformer**](https://github.com/wzwvv/DBConformer), serving as a benchmark codebase for EEG decoding models.

MVCNet is a dual-branch framework that integrates multi-view data augmentation, CNN–Transformer parallel modeling, and supervised contrastive learning to improve representation learning and decoding generalizability of EEG-based MI Classification.

<img width="916" alt="image" src="https://github.com/user-attachments/assets/2cf32c83-5fc5-422c-b731-51243483feff" />



## 📁 Project Structure

The codebase is organized as follows:

```
MVCNet/
│
├── MVCNet_CO.py        # Main script for the Chronological Order (CO) scenario
├── MVCNet_CV.py        # Main script for the Cross-Validation (CV) scenario
├── MVCNet_LOSO.py      # Main script for the Leave-One-Subject-Out (LOSO) scenario
│
├── models/             # Implementations of MVCNet and baseline models
│   ├── IFNet.py
│   ├── Conformer.py
│   └── ...
│
├── data/             # datasets
│   ├── BNCI2014001
│   ├── Zhou2016
│   └── ...
│
├── utils/              # Utility functions
│   ├── data_augment.py     # Data augmentation (e.g., time, frequency, spatial)
│   ├── contrastive_loss.py # Contrastive loss definitions
│   ├── network.py          # encoder, decoder, etc
│   └── ...
│
└── README.md
```


## 🧪 Experimental Scenarios

MVCNet supports three standard MI decoding paradigms:

- **CO (Chronological Order):** Within-subject, time-based data split
- **CV (Cross-Validation):** Within-subject, stratified 5-fold validation. The data partitions were structured chronologically while maintaining class-balance, following FBCNet.
- **LOSO (Leave-One-Subject-Out):** Cross-subject generalization evaluation

## 📊 Comparison with Baseline Models

Classification Accuracy (%) ± Std on Five MI Datasets under CO setting:

| Dataset        | EEGNet        | SCNN          | DCNN          | FBCNet        | ADFCNN        | EEGConformer  | IFNet         | **MVCNet (Ours)** |
|----------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|-------------------|
| BNCI2014001    | 69.05 ± 1.00  | 73.57 ± 2.36  | 59.29 ± 1.64  | 68.97 ± 1.26  | 73.73 ± 2.26  | 78.57 ± 0.66  | 77.94 ± 0.93  | **83.17 ± 0.74**  |
| Zhou2016       | 80.13 ± 3.35  | 75.03 ± 6.15  | 78.03 ± 2.37  | 63.33 ± 2.29  | 71.42 ± 1.95  | 73.87 ± 4.51  | 81.70 ± 2.08  | **84.11 ± 2.68**  |
| Blankertz2007  | 78.79 ± 2.78  | 76.71 ± 2.09  | 70.00 ± 4.12  | 75.93 ± 1.31  | 76.07 ± 1.41  | 82.29 ± 2.62  | 84.00 ± 0.57  | **87.07 ± 0.52**  |
| BNCI2014002    | 66.07 ± 2.76  | 79.07 ± 1.96  | 64.07 ± 2.70  | 69.50 ± 0.95  | 73.00 ± 1.95  | 76.21 ± 1.46  | 78.29 ± 1.68  | **81.29 ± 2.31**  |
| BNCI2015001    | 75.58 ± 1.69  | 83.71 ± 1.34  | 71.08 ± 1.82  | 74.92 ± 0.97  | 78.75 ± 0.62  | 82.63 ± 0.54  | 83.83 ± 0.90  | **85.67 ± 0.55**  |
| Average    | 73.92         | 77.62         | 68.49         | 70.53         | 74.59         | 78.71         | 81.15         | **84.26**         |

## 🔬 Ablation Study

Classification Accuracy (%) ± Std on Five MI Datasets under CO Setting:

| Dataset        | MVCNet         | MVCNet (cvc)   | MVCNet (cmc)   |
|----------------|----------------|----------------|----------------|
| BNCI2014001    | 83.17 ± 0.74   | 82.70 ± 1.05   | 82.46 ± 0.46   |
| Zhou2016       | 84.11 ± 2.68   | 83.18 ± 3.07   | 82.22 ± 2.61   |
| Blankertz2007  | 87.07 ± 0.52   | 85.71 ± 1.30   | 86.21 ± 1.42   |
| BNCI2014002    | 81.29 ± 2.31   | 81.29 ± 1.82   | 81.21 ± 1.92   |
| BNCI2015001    | 85.67 ± 0.55   | 85.67 ± 0.08   | 85.42 ± 0.32   |
|   Average      |   84.26        | 83.71          | 83.51          |

- MVCNet: using both CVC and CMC contrastive modules.
- MVCNet (cvc): only using the cross-view contrasting module.
- MVCNet (cmc): only using the cross-model contrasting module.

## 📂 Dataset

For reproducibility, the preprocessed EEG dataset **BNCI2014001** can be accessed at (put X.npy into /data/BNCI2014001/):

```
🔗 https://pan.baidu.com/s/19osNsaDnNliQTXxiK3ncOA  (提取码: pdtg)
```

All datasets can be downloaded from [the Mother Of All BCI Benchmarks (MOABB)](https://moabb.neurotechx.com)

---

## 💡 Citation

If you find this work helpful, please consider citing our paper:

```
@article{wang2025mvcnet,
      title={MVCNet: Multi-View Contrastive Network for Motor Imagery Classification}, 
      author={Ziwei Wang and Siyang Li and Xiaoqing Chen and Wei Li and Dongrui Wu},
      journal={arXiv preprint arXiv:2502.17482},
      year={2025}
}
```

## 🙌 Acknowledgments

Special thanks to Jiaheng for providing the source code of [IFNet](https://github.com/Jiaheng-Wang/IFNet), which served as a valuable foundation for our implementation.

We appreciate your interest and patience. Feel free to raise issues or pull requests for questions or improvements.
