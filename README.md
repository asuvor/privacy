# Supplementary Material: SLDP Experiments on Synthetic Data

This repository contains the source code and Jupyter notebooks required to reproduce the experimental results presented in the paper. The experiments evaluate the performance of the proposed Semi-Local Differential Privacy (SLDP) method against Centralized Differential Privacy (DP) and Standard Local Differential Privacy (LDP).

## Repository Structure

The code is organized as follows:

### 1. Core Implementations (`.py` files)
These files contain the underlying algorithms and must be located in the same directory as the notebooks.
* `sldp_opt_utils.py`: Implementation of the optimized **SLDP QuadTree** algorithm (Our Method). Contains the `run_dp_quadtree_optimized` function and `Rectangle` class.
* `privtree.py`: Implementation of the **PrivTree** algorithm (Baseline), used for comparison in the spatial range query experiments.

### 2. Experiments (`.ipynb` files)
* **`section_5_1_mean_estimator.ipynb`**: 
    * **Purpose:** Reproduces the Mean Estimation experiments.
    * **Figures:** Generates **Figure 1** (Visual comparison of signal reconstruction/heatmaps) and **Figure 2** (MSE comparison and Error distribution boxplots).
    * **Metrics:** Compares Central DP, Standard LDP, and SLDP (Ours).

* **`Appendix_spatial_queries_synthetic.ipynb`**:
    * **Purpose:** Reproduces the Spatial Range Query experiments (Counting Queries).
    * **Figures:** Generates the Privacy-Utility Trade-off plot (Mean Relative Error vs. Epsilon).
    * **Metrics:** Compares PrivTree, LDP Kd-Tree, and SLDP QuadTree.

## Installation & Requirements

The code is written in **Python 3**. To install the necessary dependencies, you can run:

```bash
pip install -r requirements.txt