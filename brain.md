# Brain GCC

## Generate embedding from raw data

AUROC: 0.5

## Generate embedding for each module

### Step

1. In a subject's functional connection, generating embeddings for each within module' network.
2. For each within module, train a SVM for this within module embedding across all subjects.
3. Based on the voting result of these SVMs, make predictions.

### Result

AUROC: 0.5

## TODO

