# MVCNet

This repository provides the official implementation of **MVCNet: Multi-View Contrastive Network for Motor Imagery Classification**.

MVCNet is a dual-branch framework that integrates multi-view data augmentation, CNN–Transformer parallel modeling, and supervised contrastive learning to improve representation learning and decoding generalizability of EEG-based MI Classification.


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

## 🔬 New Results

Classification Accuracy (%) ± Std on Five MI Datasets under Chronological Order (CO) Setting:

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

For reproducibility, the preprocessed EEG dataset **BNCI2014001** can be accessed at:

```
🔗 https://pan.baidu.com/s/19osNsaDnNliQTXxiK3ncOA  (提取码: pdtg)
```

Other datasets can be downloaded from [the Mother Of All BCI Benchmarks (MOABB)](https://moabb.neurotechx.com)

---

## 💡 Citation

If you find this work helpful, please consider citing the corresponding paper:

```
@article{wang2025mvcnet,
      title={MVCNet: Multi-View Contrastive Network for Motor Imagery Classification}, 
      author={Ziwei Wang and Siyang Li and Xiaoqing Chen and Wei Li and Dongrui Wu},
      journal={arXiv preprint arXiv:2502.17482},
      year={2025}
}
```

## 🙌 Acknowledgments

We appreciate your interest and patience. Feel free to raise issues or pull requests for questions or improvements.
