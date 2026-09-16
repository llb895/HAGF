# HAGF

## An Interpretable Hierarchical Adaptive Gating Network for Multimodal cfDNA Fusion in Cancer Detection and Tissue-of-Origin Inference

HAGF is a PyTorch framework for integrating four complementary cell-free DNA
(cfDNA) profiles:

- fragment-size ratio (FSR),
- end-motif distribution (EDM),
- copy-number variation (CNV), and
- DNA methylation.

The model was developed for binary cancer detection and multiclass
tissue-of-origin inference. It combines dynamic local feature grouping,
learnable sparse masks, group-level Transformer encoding, a hierarchical
fidelity path, and cross-modal gated fusion. The implementation in this
repository is the code used for the revised APBC 2026 study.

## Architecture

For each modality, HAGF performs the following operations:

1. **Dynamic feature grouping** partitions the ordered feature vector into
   local groups according to the group-ratio hyperparameter.
2. **Sparse adaptive masking** learns multiple entmax-normalized masks within
   each group.
3. **Group-level transformation and encoding** maps masked features into a
   shared hidden space and models interactions among local groups with a
   Transformer encoder.
4. **Hierarchical extraction with a fidelity path** aggregates representations
   across layers while retaining a projection of the original modality.
5. **Cross-modal gated fusion** learns sample-specific modality contributions
   before classification.

The primary model implementation is in [`hagf/model.py`](hagf/model.py), and
the leakage-controlled repeated cross-validation workflow is in
[`hagf/runner.py`](hagf/runner.py).

## Cohorts and tasks

| Accession | Study label used in the manuscript | Primary use |
| --- | --- | --- |
| CRA001537 | Zhang et al. dataset | HCC detection |
| PRJNA929650 | Pham et al. dataset | Breast-cancer detection and independent validation |
| HRA003209 | Bie et al. dataset | Multi-cancer detection, tissue-of-origin inference, and independent validation |

The datasets and derived feature matrices are not redistributed in this
repository. Obtain the source data under the terms of the corresponding study
and prepare the four aligned feature matrices described in
[`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md).

## Repository structure

```text
HAGF/
|-- hagf/
|   |-- data.py                 # strict sample-ID-aligned data loading
|   |-- model.py                # HAGF architecture and ablation switches
|   `-- runner.py               # CV, validation, checkpoint, and audit workflow
|-- configs/
|   `-- baseline.json           # manuscript baseline hyperparameters
|-- analysis/                   # reusable audit and reviewer-analysis scripts
|-- tests/                      # model, runner, and data integration checks
|-- docs/
|   `-- DATA_LAYOUT.md
|-- pyproject.toml
`-- requirements.txt
```

## Environment

The revised experiments were run with Python 3.10.20, PyTorch 2.6.0 with CUDA
12.4, cuDNN 9.1, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, SciPy 1.15.3,
statsmodels 0.14.6, and Matplotlib 3.10.9 on Ubuntu 22.04.

Create an isolated environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the CUDA-enabled PyTorch build appropriate for your system when the
default package index does not provide the required build.

## Data configuration

Pass the dataset root explicitly with `--subject-root`, or set
`HAGF_SUBJECT_ROOT`. Outputs default to `outputs/` and can be redirected with
`--output-root` or `HAGF_OUTPUT_ROOT`.

```bash
export HAGF_SUBJECT_ROOT=/path/to/SUBJECT
export HAGF_OUTPUT_ROOT=/path/to/hagf_outputs
```

Every modality table must use the first column as the sample identifier.
Sample identifiers and labels must agree across all four modalities. The
loader fails on duplicate IDs, mismatched samples or labels, nonnumeric
features, and nonfinite values.

## Training and evaluation

Run the full HAGF model for cancer detection:

```bash
python -m hagf.runner \
  --subject-root /path/to/SUBJECT \
  --output-root outputs \
  --cohort CRA001537 \
  --task detection \
  --variant full
```

Run development-cohort cross-validation followed by evaluation on the
prespecified independent cohort:

```bash
python -m hagf.runner \
  --subject-root /path/to/SUBJECT \
  --output-root outputs \
  --cohort PRJNA929650 \
  --task detection \
  --variant full \
  --evaluate-independent
```

Run tissue-of-origin inference for the Bie et al. cohort:

```bash
python -m hagf.runner \
  --subject-root /path/to/SUBJECT \
  --output-root outputs \
  --cohort HRA003209 \
  --task tissue \
  --variant full \
  --class-weighted-loss \
  --evaluate-independent
```

The baseline configuration uses hidden dimension 100, group ratio 0.2, four
masks per group, two hierarchical layers, four attention heads, entmax alpha
1.1, dropout 0.1, Adam with learning rate 0.001, batch size 10, a maximum of
300 epochs, and early stopping after 15 epochs without validation improvement.
Development performance is assessed with five repeated 10-fold
cross-validation runs.

## Ablation and robustness controls

Component ablations are selected with `--variant`:

```text
full
no_grouping
no_sparse_masks
no_transformer
no_positional_embedding
no_fidelity_path
no_cross_modal_fusion
one_layer
```

Examples:

```bash
# Retrained leave-one-modality-out analysis
python -m hagf.runner --cohort HRA003209 --task detection \
  --variant full --drop-modality Methylation

# Hidden-dimension sensitivity
python -m hagf.runner --cohort CRA001537 --task detection \
  --variant full --hidden-size 64

# Feature-order robustness for one modality
python -m hagf.runner --cohort PRJNA929650 --task detection \
  --variant full --permute-modality EDM --permutation-seed 20260914
```

Use `--selection-only` for development-set model selection. Use
`--evaluate-independent` only after the configuration is fixed. This separation
keeps independent cohorts outside hyperparameter selection.

## Outputs

Each experiment directory contains the resolved configuration, fold-level
metrics, held-out predictions, completion markers, and optional checkpoints or
sparse masks. Use `--save-checkpoints` and `--save-masks` only when those
artifacts are required, because they can be large.

The scripts in [`analysis/`](analysis/) summarize saved predictions, component
ablations, hidden-dimension sensitivity, modality contribution, individual
attribution, feature-order robustness, and biological enrichment. See
[`analysis/README.md`](analysis/README.md) for their input requirements.

## Tests

The model and runner smoke tests do not require study data:

```bash
python tests/test_model.py
python tests/test_runner.py
```

The data integration test requires the complete prepared dataset hierarchy:

```bash
HAGF_SUBJECT_ROOT=/path/to/SUBJECT python tests/test_data.py
```

## Reproducibility notes

- Modalities are aligned by sample identifier before fitting.
- Scaling is fitted on each training fold and then applied to validation and
  independent samples.
- A validation subset of the training fold controls early stopping.
- Random seeds and resolved configurations are saved with each run.
- Independent cohorts are not used for hyperparameter selection.
- Dataset-level AUC and repeat-level uncertainty should be reported as distinct
  quantities.

## Citation

Please cite the associated manuscript when using this code:

```bibtex
@misc{lu_hagf_2026,
  title  = {HAGF: An Interpretable Hierarchical Adaptive Gating Network for
            Multimodal cfDNA Fusion in Cancer Detection and Tissue-of-Origin
            Inference},
  author = {Lu, Libo and Sheng, Xinwei and Xu, Zixian and Li, Xue and
            Zeng, Fanxin and Zhou, Xionghui},
  year   = {2026},
  note   = {APBC 2026 revised manuscript}
}
```

Update this entry with the final proceedings citation and DOI after publication.

## Data and privacy

No participant-level data, derived feature matrices, trained checkpoints, or
credentials are included in this source release. Users are responsible for
complying with the access conditions, privacy requirements, and ethical terms
of the original cohorts.
