# Koopman4Rob

A PyTorch-based light-weight framework for learning **Koopman operators** in complex robot systems, enhanced with **Elastic Weight Consolidation (EWC)** for continual learning and fine-tuning of the operator.

## Reference
This repository's implementation is largely based on the following repositories:

- **PyKoopman**: 
    https://github.com/dynamicslab/pykoopman
- **Elastic-Weights-Consolidation**: 
    https://github.com/Yuxing-Wang-THU/Elastic-Weights-Consolidation

## 🛠 Installation

```bash
conda create -n koopman4rob python=3.8
conda activate koopman4rob
pip3 install -r descriptive_file/requirements.txt
```

## 📁 Project Structure

```
Koopman4Rob/
├── data/                               # Input datasets (.npy files)
├── models/
│ ├── deep_koopman.py                   # Deep Koopman implementation
│ └── ewc.py                            # EWC implementation
├── runner/
│ └── koopman_runner.py                 # Train/test loop with optional EWC
├── utils/
│ └── utils.py
├── logs/                               # Saved models & fisher info
├── main_demo.py                        # Main function for training/testing
├── requirements.txt                    # Env dependencies
└── README.md
```

## 🎯 Usage

### 🚀 Train a Koopman model

```bash
python3 main_demo.py
```
在 main_demo.py 中你可以配置如下内容：
```bash
run(
    mode="train",
    data_path="data/path to .npy file",
    model_dir="logs/dir of your model"
)
```

### 🧪 Test a saved model
```bash
python3 main_demo.py
```
在 main_demo.py 中你可以配置如下内容：
```bash
run(
    mode="test",
    data_path="data/path to .npy file",
    model_dir="logs/dir of your model"
)
```

### 📊 Visualize fisher info
```bash
python3 visualize_fisher.py --file_path="path to your .pt file" --threshold=None"
```



## 💡 Tips

- **Data format**: each row should be `[x_t, a_t, x_{t+1}]`. General shape [N, x+a+x].
- **Normalization**: enable with `normalize=True` in `KoopmanRunner`, but requires self-defining.
- **Trajectory smoothing**: edit `smooth_curve()` in `utils/utils.py`.
- **Multi-task setting**: currently enabling only single training.



