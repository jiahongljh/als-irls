# ALS-IRLS: Outlier-Robust Autocovariance Least-Squares Estimation

This repository provides the official MATLAB implementation of the **ALS-IRLS** algorithm, as proposed in the paper:

> **"Outlier-Robust Autocovariance Least-Squares Estimation via Iteratively Reweighted Least Squares"**  
> *Jiahong Li, Fang Deng*  


## 📌 Overview

The Autocovariance Least-Squares (ALS) method is a powerful tool for estimating the unknown process and measurement noise covariance matrices ($Q$ and $R$) for Kalman filters. However, the standard ALS relies on the least mean squares (LMS) criterion, making it highly vulnerable to non-Gaussian measurement outliers. A single severe outlier can completely corrupt the empirical innovation autocovariances, causing the standard ALS to yield grossly inaccurate covariance estimates.

**ALS-IRLS** solves this critical vulnerability by:
1. Recasting the ALS regression as an **outlier-robust regression problem** using the Huber cost function.
2. Employing the **Iteratively Reweighted Least Squares (IRLS)** technique to dynamically down-weight the outlier-corrupted autocovariance entries.
3. Incorporating **closed-form Positive Semi-Definite (PSD) cone projections** to ensure physically admissible covariance estimates.

Our extensive simulations demonstrate that ALS-IRLS reduces the covariance estimation RMSE by **over two orders of magnitude** compared to standard ALS, allowing downstream state estimation to approach the Oracle lower bound even under severe $\epsilon$-contamination.

## 🚀 Repository Structure

The codebase is organized into a modular MATLAB framework.

### 1. Main Execution Script
- `als_irls_main.m`
  - The main script to run the comprehensive Monte Carlo simulation.
  - It sequentially runs the **Oracle KF**, **KF+ALS (Standard)**, **KF+ALS-IRLS (Ours)**, **Student's-t KF**, and **Maximum Correntropy KF (MCKF)**.
  - Generates the comparative RMSE results and visual scatter plots.

### 2. Core Algorithm (Our Contribution)
- `irls_huber.m`
  - Implements **Algorithm 1**: The IRLS solver for Huber-loss robust regression.
- `build_ALS_matrix.m`
  - Constructs the ALS design matrix $\mathcal{A}$ based on the system dynamics and current Kalman gain.
- `compute_b_LS.m`
  - Computes the empirical innovation autocovariance vector $b$.
- `symtran.m`
  - Utility function for matrix symmetrization and Positive Semi-Definite (PSD) projection via eigen-decomposition (Remark 3 in the paper).

### 3. Baseline Robust Filters
- `student_t_kf.m`
  - Implementation of the Student's-t Robust Kalman Filter.
- `mckf.m`
  - Implementation of the Maximum Correntropy Kalman Filter (MCKF).
- `run_kf.m`
  - Standard Kalman Filter routine used by Oracle, ALS, and ALS-IRLS.

## 💻 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jiahongljh/als-irls.git
   cd als-irls
Run the simulation in MATLAB:
Simply open MATLAB, navigate to the folder, and run the main script:

matlab
als_irls_main
Expected Output:

The script will execute the third-order LTI system simulation described in Section IV of the paper.

It will output the state estimation RMSE for all 5 methods directly into the MATLAB console.

It will generate plots showcasing the joint scatter of the estimated $Q$ and $R$ parameters (similar to Fig. 1 in the paper).

📊 Citation
If you find this code useful for your research, please cite our paper:

text
@article{li202Xalsirls,
  author={Li, Jiahong and Deng, Fang},
  title={Outlier-Robust Autocovariance Least-Squares Estimation via Iteratively Reweighted Least Squares},
  journal={IEEE Signal Processing Letters},
  year={202X},
  volume={},
  number={},
  pages={}
}
(Note: Citation details will be updated upon publication).

📧 Contact
For any questions or discussions regarding the algorithm, please feel free to open an issue or contact:

Jiahong Li (jqr_jiahong@buu.edu.cn)

Fang Deng (dengfang@bit.edu.cn)