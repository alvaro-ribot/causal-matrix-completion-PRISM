# Predicting Context-Dependent Drug Effects using Causal Matrix Completion

## Setup

To set up a virtual environment with the required packages, run:

```
bash setup.sh
```

## Real Data Reproduction

To reproduce the results using the PRISM dataset, run:

```
python3 -m src/run_experiment.py
```

In the *Notebooks* folder, there are several jupyter notebooks with plots showing this results.

## Simulating data

The code for simulating a matrix coming from a given DAG can be found in the *Simulations* folder. This was used for plotting the ROC of the hypothesis tests.