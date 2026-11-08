# HARMONY: Heterogeneous Graph Learning for Multi-Omics Disease Classification

**HARMONY** is a PyTorch-based research implementation for integrating multiple omics modalities through **heterogeneous patient–feature graphs**, **bidirectional graph message passing**, and **cross-attention fusion**. The framework learns patient representations from DNA methylation, mRNA expression, and miRNA expression data for multi-class disease classification.

> **Repository status:** Research implementation. The uploaded source code requires the two minor fixes documented in [Implementation notes](#implementation-notes) before it can run as a standalone Python script. No benchmark scores or publication metadata are asserted here unless independently provided.

## Overview

Multi-omics datasets describe the same patients through complementary molecular measurements. Instead of concatenating all features into a single vector, HARMONY constructs a separate heterogeneous graph for each modality. Each graph represents both **patients** and **molecular features**, allowing the model to learn from relationships within and between these node types.

The workflow is:

```text
DNA methylation ──> Preprocessing ──> Heterogeneous graph ──> Graph encoder ──┐
mRNA expression ──> Preprocessing ──> Heterogeneous graph ──> Graph encoder ──┼─> Multi-head attention ─> Classifier
miRNA expression ──> Preprocessing ──> Heterogeneous graph ──> Graph encoder ──┘
```

### Key components

- **Three relation types per modality:** patient–patient, feature–feature, and feature–patient links.
- **Data-driven sparsification:** cosine similarities, top-*k* neighbors, and adaptive thresholds (Otsu by default).
- **Learnable edge relevance:** neural scoring modules compute updated edge weights during graph encoding.
- **Bidirectional cross-type propagation:** feature-to-patient and patient-to-feature messages are combined with same-type neighborhood information.
- **Sorted LSTM aggregation:** neighbors are ordered by their learned edge weights before sequence aggregation.
- **Gated updates:** separate patient-side and feature-side gates balance intra-type and cross-type messages.
- **Multi-omics attention:** modality-specific patient embeddings are fused with multi-head self-attention across modalities.
- **Imbalance handling:** configurable, training-only SMOTE-style synthetic oversampling and focal loss.

## Methodology

### 1. Data preparation

Each modality uses the same patient order and class labels. The implementation:

1. Reserves **20% of the provided training data** for stratified validation.
2. Standardizes each modality using a `StandardScaler` fitted on the resulting training subset only.
3. Selects up to **512 features per modality** using mutual information with training labels.
4. Applies a custom SMOTE-style interpolation algorithm to the training subset only; it concatenates the selected modalities before generating synthetic samples to preserve their alignment.
5. Leaves validation and test examples free from synthetic oversampling.

The test dataset is supplied separately; it is not created by the script.

### 2. Modality-specific heterogeneous graphs

Each graph contains two node types:

- **Patient nodes:** initialized by projecting each patient's selected molecular feature vector.
- **Feature nodes:** initialized from six summary statistics computed across the patients in that split: mean, standard deviation, median, skewness, kurtosis, and interquartile range.

Three edge relations are constructed:

| Edge relation | Construction | Purpose |
| --- | --- | --- |
| Patient → patient | Top-*k* cosine similarity with adaptive thresholding and self-loops | Capture similar patient profiles |
| Feature → feature | Top-*k* cosine similarity between feature profiles with adaptive thresholding and self-loops | Capture relationships among molecular measurements |
| Feature → patient | Top-*k* patients per feature based on normalized absolute feature values and an adaptive threshold | Connect molecular features to patients |

The feature–patient links are explicitly reversed inside the encoder to support **patient → feature** message passing as well. The graph builder supports Otsu, percentile, and mean-plus-standard-deviation thresholds, although **Otsu** is the default configuration.

### 3. Heterogeneous message passing

The modality-specific `HeteroInductiveEncoder` runs three message-passing layers by default. Each layer:

1. Computes learned edge scores with relation-specific MLPs.
2. Aggregates patient–patient and feature–feature neighborhoods using weight-sorted LSTMs.
3. Propagates feature information to patients and patient information back to features using separate cross-type LSTM modules.
4. Uses softmax gates to combine same-type and cross-type representations.
5. Normalizes the updated node embeddings before the next layer.

The encoder outputs patient embeddings of dimension **128** by default. Individual modality classifiers also predict disease classes from these embeddings.

### 4. Cross-attention fusion

For each patient, the three modality-specific embeddings are stacked as tokens and passed to a **four-head multi-head attention layer**. A residual connection, layer normalization, dropout, and a feed-forward classifier produce the final class logits.

Attention is applied **across modality embeddings**, not across raw genomic sequences. The provided implementation does not export attention weights as explanations.

### 5. Training objective

The framework optimizes focal loss on both the fused prediction and auxiliary predictions from the three modality-specific classifiers:

\[
\mathcal{L}_{total}=\mathcal{L}_{fusion}+0.3\sum_{m=1}^{3}\mathcal{L}_{m}.
\]

The training setup uses Adam, a ten-epoch linear learning-rate warmup, and a cosine-annealing schedule afterward. At each evaluation interval, the best checkpoint is selected using validation **weighted F1** by default (configurable to macro F1).

## Repository layout

Suggested structure for publishing the supplied code:

```text
HARMONY/
├── README.md
├── harmony.py                 # Rename the supplied Python source to this
├── requirements.txt           # Optional: create from the dependency list below
└── data/
    ├── 1_tr.csv
    ├── 2_tr.csv
    ├── 3_tr.csv
    ├── 1_te.csv
    ├── 2_te.csv
    └── 3_te.csv
```

> The current source expects CSV files in the **working directory**, not in `data/`. Either place the CSVs next to `harmony.py`, or update `TRAIN_FILES` and `TEST_FILES` to point to `data/`.

## Installation

Python 3.10+ is a reasonable starting environment; the supplied code does not specify an exact Python or package version. Install the required packages:

```bash
pip install numpy pandas scipy scikit-learn torch
```

A CUDA-capable GPU is recommended for practical experimentation, but the code automatically selects the CPU when CUDA is unavailable.

## Dataset format

The script expects **six headerless CSV files**:

| File | Modality | Partition |
| --- | --- | --- |
| `1_tr.csv` | DNA methylation | Training/validation source |
| `2_tr.csv` | mRNA expression | Training/validation source |
| `3_tr.csv` | miRNA expression | Training/validation source |
| `1_te.csv` | DNA methylation | Held-out test |
| `2_te.csv` | mRNA expression | Held-out test |
| `3_te.csv` | miRNA expression | Held-out test |

Each row represents a patient, all columns except the last contain numerical molecular measurements, and **the final column contains an integer class label**. Rows and labels must match across modalities within each partition. By default, the classifier expects **five classes**, encoded as integers `0` through `4`.

Example schematic CSV row:

```csv
0.12,-0.37,1.04,0.52,2
```

The four measurements shown are illustrative; actual feature counts vary by modality. The same number of patients and consistent patient ordering are required across all three modality files in each partition. The source does not implement patient-ID matching.

## Configuration

The following defaults are defined near the top of `harmony.py`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `SEED` | `42` | NumPy and PyTorch random seed |
| `N_FEATURES` | `512` | Maximum selected features per modality |
| `K_PAT` / `K_FEAT` | `15` / `15` | Candidate neighbor count for patient and feature graphs |
| `THR_METHOD` | `'otsu'` | Adaptive edge threshold |
| `EMBED_DIM` | `128` | Node embedding dimension |
| `N_LAYERS` | `3` | Heterogeneous encoder depth |
| `NUM_CLASSES` | `5` | Output classes |
| `N_HEADS` | `4` | Attention heads |
| `MAX_NBR` | `15` | Maximum neighbors in LSTM aggregation |
| `DROPOUT` | `0.3` | Main dropout setting |
| `EPOCHS` | `100` | Maximum training epochs |
| `LR` | `0.001` | Base learning rate |
| `WARMUP_EPOCHS` | `10` | Linear warmup duration |
| `FOCAL_GAMMA` | `1.0` | Focal-loss focusing parameter |
| `SMOTE_RATIO` | `0.5` | Fraction of each class's gap to majority size filled with synthetic examples |
| `USE_CLASS_WEIGHTS` | `False` | Apply class-balanced weights in focal loss |
| `VAL_SIZE` | `0.2` | Validation share of provided training data |
| `SELECT_METRIC` | `'weighted'` | Best-checkpoint validation F1 metric |

Some additional settings (for example, `N_SAMPLE`) are declared in the supplied source but are not used by the executed workflow.

## Running HARMONY

After addressing the implementation notes below and placing your CSV files at the configured paths:

```bash
python harmony.py
```

The script prints the selected device, dataset dimensions, class distributions, graph edge counts and threshold values, and epoch-level validation accuracy, macro F1, and weighted F1. After training, it generates fused test predictions and probabilities. See the note below for completing confusion-matrix output.

### Evaluation

The supplied code calculates validation:

- Accuracy
- Macro-averaged F1
- Weighted F1

It also obtains predictions for the held-out test partition. To report **test** accuracy, macro F1, weighted F1, per-class precision/recall, and the confusion matrix, add the following lines after test predictions are computed:

```python
print(f"Test accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"Test macro F1: {f1_score(y_test, preds, average='macro'):.4f}")
print(f"Test weighted F1: {f1_score(y_test, preds, average='weighted'):.4f}")
print(classification_report(y_test, preds, zero_division=0))
cm = confusion_matrix(y_test, preds)
print(cm)
```

**No quantitative test results are included in this README**, because none were supplied alongside the code.

## Implementation notes

Before publishing or running the supplied text as a Python script:

1. **Remove the standalone `-` line** directly after `DROPOUT = 0.3`; otherwise Python raises a syntax error.
2. **Define `cm` before using it.** The last loop references `cm` without assigning it; the evaluation snippet above shows the required assignment.
3. **Update file paths** if CSVs are stored under `data/` rather than the working directory.
4. **Ensure integer class labels match the configured output size.** The code assumes zero-based labels when computing cross-entropy loss.

**Scope and reproducibility considerations:** Graphs are built separately for training, validation, and test partitions from their own features. Consequently, evaluation processes a collection of held-out patients together, rather than one unseen patient independently; describe this setting precisely when reporting inductive generalization. The implementation sets NumPy/PyTorch seeds but does not enable fully deterministic GPU algorithms. There is no CLI argument parser, automated dataset downloader, checkpoint export, repeated-run summary, or cross-validation loop in the provided code.

## Research extensions

Potential follow-up experiments include ablations of each edge relation, gated bidirectional message passing, auxiliary modality losses, and cross-attention fusion; comparisons with concatenation and attention-free fusion; performance across multiple random seeds; class-specific analysis under imbalance; and evaluation on completely independent cohorts. These are **suggested experiments**, not claims about completed work.

## Citation

If you use this implementation, please cite the corresponding HARMONY paper **once the authors provide its verified bibliographic details**. A BibTeX entry can be added here after publication information is finalized.

## License

No license was specified with the supplied source. Add a `LICENSE` file after choosing appropriate distribution terms. Until then, do not assume the repository is released under an open-source license.
