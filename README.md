
# Track reconstruction
Graph-based reconstruction of long-lived particle tracks in the CERN LHCb experiment.
The goal is to select the correct SciFi continuation for a given UT track among multiple false candidates.

The goal is to select the correct SciFi continuation for a given UT track among multiple false candidates.

Each UT–SciFi pair is represented as a graph and classified as:

- 1 - correct continuation  
- 0 - false (ghost) candidate

## Requirements

- Python 3.8+  
- PyTorch  
- PyTorch Geometric  
- NumPy  
- Pandas  
- Matplotlib  

## Install dependencies:
pip install torch torch-geometric numpy pandas matplotlib

## How to Run:
### Run training and evaluation:

python3 train.py


## Output

- Trained GNN model
- Probability distributions for true and false candidates
- Top-k reconstruction efficiency plots
