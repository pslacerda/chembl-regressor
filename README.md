ChEMBL Regressor
================

Install
-------

```bash
$ pip install chembl-regressor
```

Usage
-----

A program for biological activity prediction trained with ChEMBL assays.

```
$ chembl-regressor train --help  
Usage: chembl-regressor train [OPTIONS] CHEMBL_INPUT

  This program train a regressor based on ChEMBL dataset.

Options:
  --test-size FLOAT               fraction of test size
  --random-seed INTEGER
  --cross-val INTEGER             number of cross validation folds
  --drop-na / --no-drop-na        drop NaN values
  --n-jobs INTEGER                number of jobs
  --target-var [pChEMBL Value|Standard Value]
                                  target variable
  --sample-fraction FLOAT         use only a sample fraction of the input data
  --log10 / --no-log10            may be useful for standartize the target
                                  variable
  --plot / --no-plot              display plots
  --output-model PATH             output model filename
  --help                          Show this message and exit.
```

```
$ chembl-regressor predict --help
Usage: chembl-regressor predict [OPTIONS] COMPOUNDS_SMI

  This program predict the pChEMBL value for a list of compounds.

Options:
  --input-model TEXT    output model filename
  --output-scores PATH
  --help                Show this message and exit.
```

Example
-------

```bash
$ chembl-regressor train \
    --sample-fraction 0.25 \
    --drop-na \
    --target-var 'pChEMBL Value' \
    data/AChE.zip
```

```bash
$ chembl-regressor predict \
    --output-scores OXTR-fda.csv \
    ~/my_path/fda.smi
```
