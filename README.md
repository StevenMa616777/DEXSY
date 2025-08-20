# DEXSY Reconstruction with ResUNet-XL

This project implements a **deep learning framework** for reconstructing **diffusion exchange spectroscopy (DEXSY)** data using a **ResUNet-XL** model.  
The goal is to predict the **diffusion distribution f(D1,D2)** directly from the measured diffusion signal matrix.

---

## Background

- **DEXSY** characterizes water exchange between intra- and extracellular spaces by mapping diffusion coefficients in a 2D domain.  
- Traditional Laplace inversion is **ill-posed** and highly sensitive to noise.  
- We propose a **ResUNet-XL** model that directly learns the mapping from noisy diffusion signals to ground-truth diffusion distributions.  
- The output is a **non-negative, normalized 64×64 matrix** representing the diffusion structure distribution.

---

## Dataset

The dataset is **synthetically generated** to mimic **2-compartment and 3-compartment exchange models** in DEXSY.

1. **Diffusion Coefficient Grid**  
   - A 1D array `D_vals.npy` (length = 64 by default) stores log-spaced diffusion coefficients.  
   - A 2D meshgrid `(D1, D2)` is constructed from `D_vals`.  

2. **Ground Truth Distribution (`f_*.npy`)**  
   - Two types of synthetic models are used:  
     - **2-compartment model (4 peaks):**  
       - Intra- and extracellular compartments represented by two Gaussian blobs each.  
       - Blobs are positioned within predefined ranges in `(D1, D2)` space to simulate restricted ↔ free diffusion exchange.  
     - **3-compartment model (9 peaks):**  
       - Adds an intermediate compartment.  
       - Three Gaussian blobs per compartment (total 9 peaks).  
   - Each `f` is a **64×64 matrix**, non-negative, normalized to sum=1.  

3. **Signal Data (`S_clean_*.npy`, `S_noisy_*.npy`)**  
   - Forward DEXSY signal computed as:  
     \[
     S(b_1,b_2) = \sum_{i,j} f(D_1^i,D_2^j) \, \exp(-b_1 D_1^i - b_2 D_2^j)
     \]  
   - `S_clean`: noiseless forward signal.  
   - `S_noisy`: signal with Rician noise and b0-normalization (`S(0,0)=1`).  

4. **Folder Structure Example**  

```
data/
├── f_0.npy              # 64×64 ground truth distribution
├── S_clean_0.npy        # clean signal (64×64)
├── S_noisy_0.npy        # noisy signal (64×64)
├── f_1.npy
├── S_clean_1.npy
├── S_noisy_1.npy
...
├── D_vals.npy           # 1D array of diffusion coefficients
```

- **Model Input:** `(S, logD1, logD2)` as 3 channels.  
- **Model Output:** `f_hat`, normalized probability distribution.  

---

## Model

- **ResUNet-XL**  
  - 5-level encoder–decoder (64→2 resolution)  
  - Residual double conv blocks with SiLU activation  
  - Squeeze-and-Excitation (SE) channel attention  
  - Attention Gates for skip connections  
- **Output**  
  - Softplus activation for non-negativity  
  - Global normalization to ensure sum=1  

---

## Loss Function

Composite loss:

- KL divergence (distribution alignment)  
- L1 loss (robust reconstruction)  
- Total Variation (smoothness prior)  

\[
\mathcal{L} = \mathrm{KL}(f \parallel \hat f) + 0.1 \|f-\hat f\|_1 + 10^{-5}\mathrm{TV}(\hat f)
\]

---

## Training

Main script: `run_train.py`

```bash
python run_train.py --root ./data     --batch_size 8     --epochs 200     --lr 3e-4     --use_noisy
```

**Setup**:
- Optimizer: AdamW (lr=3e-4, betas=(0.9,0.95), weight_decay=1e-4)  
- Scheduler: 5% warmup + cosine decay  
- Mixed precision training (AMP with GradScaler)  
- Gradient clipping (max norm=1.0)  
- Exponential Moving Average (EMA, decay=0.999)  

---

## Evaluation & Visualization

- Every **10 epochs**, one validation example (prediction vs. ground truth) is saved in  
  `runs/.../val_examples/`.  
- TensorBoard logs available under `runs/`.

---

## Example Results

- Input: noisy diffusion signal `S_noisy` (64×64)  
- Output: reconstructed distribution `f_hat` closely matches ground truth `f`  

Evaluation metrics:
- KL/JS divergence  
- Top-K peak accuracy  
- Wasserstein distance (optional)  

---

## Pipeline Diagram

```mermaid
flowchart TD
    A[Input S_noisy (64x64)] --> B[+ Coord Channels (logD1, logD2)]
    B --> C[ResUNet-XL]
    C --> D[Softplus + Normalization]
    D --> E[Predicted f_hat (64x64)]
    E --> F[Loss: KL + L1 + TV]
    F --> C
```

---

## Project Structure

```
.
├── data/                 # dataset
├── models/
│   └── ResUNet_big.py    # model definition
├── run_train.py          # training entry point
├── dataset.py            # DataLoader
├── utils.py              # loss, EMA, visualization
└── README.md             # project description
```

---

## Acknowledgments

- Inspired by **Helaly et al. (2021)** and related work on deep learning for diffusion MRI.  
- Model design draws from **ResUNet**, **Attention U-Net**, and **SE-ResNet**.  
- Developed in the context of DEXSY research at **UCL / The Crick Institute**.  
