# Data layout

HAGF expects four CSV matrices per cohort split. The files are not included in
the repository.

## Directory hierarchy

```text
SUBJECT/
|-- data/
|   `-- dataset/
|       |-- CRA001537_cross/
|       |   |-- FSR.csv
|       |   |-- end_motifs.csv
|       |   |-- CopyNumber.csv
|       |   `-- methy_selected.csv
|       |-- PRJNA929650_cross/
|       |   |-- FSR.csv
|       |   |-- end_motifs.csv
|       |   |-- CopyNumber.csv
|       |   `-- methy_new_standard_1.csv
|       |-- PRJNA929650_validation/
|       |   `-- [same four filenames as PRJNA929650_cross]
|       |-- HRA003209_cross/
|       |   |-- FSR.csv
|       |   |-- end_motifs.csv
|       |   |-- CopyNumber.csv
|       |   `-- methy.csv
|       `-- HRA003209_validation/
|           `-- [same four filenames as HRA003209_cross]
`-- CRA001537/
    `-- csv/
        |-- HRA003209_cross/
        |   |-- FSR_adjusted.csv
        |   |-- end_motifs_adjusted.csv
        |   |-- CopyNumber_adjusted.csv
        |   `-- methy_adjusted.csv
        `-- HRA003209_validation/
            `-- [same four adjusted filenames]
```

The `data/dataset` hierarchy is used for binary cancer detection. The adjusted
HRA003209 hierarchy is used for tissue-of-origin inference.

## Table schema

All modality files for a cohort split must contain the same samples and labels.
Rows may initially appear in different orders because the loader aligns them by
sample identifier.

| Cohort/task | Sample ID | Label | First feature |
| --- | --- | --- | --- |
| CRA001537 detection | column 1 | column 2 | column 3 |
| PRJNA929650 detection | column 1 | column 2 | column 3 |
| HRA003209 detection | column 1 | column 3 | column 5 |
| HRA003209 tissue of origin | column 1 | column 3 | column 5 |

CRA001537 and PRJNA929650 detection labels must be numeric. HRA003209 detection
maps `healthy` to 0 and every cancer label to 1. HRA003209 tissue-of-origin
labels are case-insensitive and must be one of `healthy`, `BRCA`, `COREAD`,
`ESCA`, `STAD`, `LIHC`, `NSCLC`, or `PACA`.

Every feature cell must be numeric and finite. Duplicate sample IDs, sample-set
mismatches, label mismatches, and missing or infinite values cause an explicit
error.

## Configuration

Pass the hierarchy root directly:

```bash
python -m hagf.runner --subject-root /path/to/SUBJECT ...
```

Alternatively, set the environment variable before invoking the runner:

```bash
export HAGF_SUBJECT_ROOT=/path/to/SUBJECT
```

On PowerShell:

```powershell
$env:HAGF_SUBJECT_ROOT = "D:\\path\\to\\SUBJECT"
```
