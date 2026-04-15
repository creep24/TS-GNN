This repository provides the anonymized implementation of our proposed model **TS-GNN** (Temporal-Semantic Graph Neural Network) for credit card fraud detection. TS-GNN is designed to combat fraud camouflage by leveraging both temporal and semantic discrepancies across transaction graphs. The model is composed of three core modules:

* **TSDM**: Temporal-Semantic Discrepancy Modeling, which quantifies temporal and semantic differences between connected nodes.
* **DAT**: Discrepancy-Aware Attention Mechanism, which assigns attention weights based on discrepancy-aware edge credibility.
* **ARGU**: Anomaly-Retaining Gated Unit, which adaptively updates node states while preserving abnormal patterns.

We evaluate TSGNN on two publicly available datasets: **IEEE-CIS** and **CreditCard**, demonstrating its effectiveness in identifying fraud cases disguised as normal transactions.

---

## 📁 Project Structure

```
.
├── models/                  # Model modules: tsdm.py, dat.py, argu.py, tsgnn.py
├── data1.zip                # Zipped dataset files (IEEE-CIS or CreditCard)
├── main.py                  # Entry point for training and evaluation
├── requirements.txt         # Python dependencies
└── README.md                # Project overview and instructions
```

---

## 📦 Requirements

* Python >= 3.8
* PyTorch >= 2.0
* PyTorch Geometric >= 2.2
* pandas, scikit-learn, numpy

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

### 1. Unzip the Dataset

Unpack `data1.zip` in the root directory. It will automatically create a `data/` folder containing the processed dataset files:

```bash
unzip data1.zip
```

### 2. Verify Data Contents

The `data/` directory should contain files such as:

* `transaction.csv`: node features
* `label.csv`: node labels
* `test_identity.csv`: transaction identity in test
* `train_identity.csv`: transaction identity in train

Make sure the structure looks like this:

```
data/
├── transaction.csv
├── label.csv
├── test_identity.csv
├── train_identity.csv
```

### 3. Train the Model

Execute the following command to train TS-GNN on a specific dataset:

```bash
python main.py --dataset ieee
```

You may customize additional arguments such as:

* `--layers`: number of GNN layers
* `--hidden_dim`: hidden state dimension
* `--lr`: learning rate
* `--epochs`: total training epochs

For example:

```bash
python main.py --dataset ieee --layers 2 --hidden_dim 64 --lr 1e-3 --epochs 200
```

