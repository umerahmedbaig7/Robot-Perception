<div align="center">

# 👁️ Robot Perception
### MSc Robotics and Artificial Intelligence — Course Repository

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-27AE60?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)](https://github.com/umerahmedbaig7)
[![Field](https://img.shields.io/badge/Field-Robotics%20%26%20AI-blueviolet?style=for-the-badge&logo=ros&logoColor=white)]()

<br>

> *"A robot that cannot perceive its world cannot act within it. Teaching machines to estimate wheel parameters from noisy encoders, track hidden states through non-linear camera measurements, and reconstruct three-dimensional depth from a pair of stereo images is not merely signal processing — it is the systematic construction of the perceptual intelligence that makes autonomous robotics possible."*

<br>

**Author:** Umer Ahmed Baig Mughal <br>
**Programme:** MSc Robotics and Artificial Intelligence <br>
**Specialization:** Machine Learning · Computer Vision · Human-Robot Interaction · Autonomous Systems · Robotic Motion Control <br>
**Institution:** ITMO University — Faculty of Control Systems and Robotics

</div>

---

## 📋 Table of Contents

- [📖 About This Repository](#-about-this-repository)
- [🗂️ Repository Structure](#️-repository-structure)
- [🔬 Course Overview](#-course-overview)
- [🧪 Lab Summaries](#-lab-summaries)
  - [🔵 Lab 1 — Wheel Radius Estimation via Least Squares Methods](#-lab-1--wheel-radius-estimation-via-least-squares-methods)
  - [🟠 Lab 2 — 1D Position Estimation via Extended Kalman Filter](#-lab-2--1d-position-estimation-via-extended-kalman-filter)
  - [🟣 Lab 3 — 1D Position Estimation via Unscented Kalman Filter (Sigma-Point Method)](#-lab-3--1d-position-estimation-via-unscented-kalman-filter-sigma-point-method)
  - [🟢 Lab 4 — Camera Calibration and Stereo Depth Estimation](#-lab-4--camera-calibration-and-stereo-depth-estimation)
- [🔗 Progressive Learning Pathway](#-progressive-learning-pathway)
- [⚙️ Shared Mathematical Framework — The Kalman Update Paradigm](#️-shared-mathematical-framework--the-kalman-update-paradigm)
- [🚀 Quick Start](#-quick-start)
- [📊 Results at a Glance](#-results-at-a-glance)
- [🧰 Tech Stack](#-tech-stack)
- [👤 Author](#-author)
- [📄 License](#-license)

---

## 📖 About This Repository

This repository contains the complete implementation of all four laboratory assignments from the **Robot Perception (RP)** course, part of the MSc in Robotics and Artificial Intelligence at ITMO University. The labs form a carefully sequenced progression — from least squares parameter estimation through the full Kalman filter family to a two-part geometric computer vision pipeline — covering the mathematical and algorithmic foundations of robotic perception from first principles.

The course advances in a deliberate two-phase structure. **Phase 1** (Labs 1, 2, and 3) develops the **estimation theory** backbone of robotic perception: the ability to recover unknown quantities — whether a static mechanical parameter or a dynamic position-velocity state — from noisy, incomplete, and non-linear sensor data. Lab 1 establishes the least squares estimator as the optimal solution under Gaussian noise, then extends it recursively with Kalman-gain covariance updates. Labs 2 and 3 tackle the same physical problem — estimating a robot's 1D position from a camera measuring the angle to a distant landmark — using two fundamentally different approaches to the non-linear measurement function: the EKF linearises it via Jacobians, while the UKF propagates a cloud of deterministic sigma points through the exact function, requiring no derivatives at all. **Phase 2** (Lab 4) makes a complete paradigm shift to **geometric computer vision** — calibrating a monocular camera from a planar checkerboard target using Zhang's method, computing a dense stereo disparity map using StereoSGBM, and estimating the distance to a motorcycle obstacle at 28.764 metres using the fundamental stereo depth equation.

All implementations are built from first principles using NumPy and OpenCV. Every estimator — the batch normal equations, the Kalman gain update, the EKF Jacobian computation, the sigma-point Cholesky generation, the checkerboard corner pipeline, and the stereo depth computation — is coded explicitly without relying on any pre-built filtering, calibration, or depth-estimation framework, ensuring deep mathematical understanding at every step.

### 🎯 What You Will Find Here

| 📁 Lab | 🏷️ Topic | 🧠 Core Concept | 🛠️ Key Tools |
|:------:|:--------:|:---------------:|:-------------:|
| Lab 1 | Wheel Radius Estimation | Batch LS · Affine Model · Recursive LS · Kalman Gain | NumPy · Matplotlib |
| Lab 2 | 1D Position — EKF | Jacobian Linearisation · EKF Prediction-Update Cycle | NumPy · Matplotlib |
| Lab 3 | 1D Position — UKF | Sigma-Point Transform · Cholesky Decomposition · Cross-Covariance | NumPy · Matplotlib |
| Lab 4 | Camera Calibration & Stereo Depth | Zhang Calibration · StereoSGBM · Template Matching · Z = f·b/d | OpenCV · NumPy |

---

## 🗂️ Repository Structure

```
📦 Robot-Perception/
│
├── 📁 Lab_1/
│   ├── 📄 Readme.md                                          # Lab 1 full documentation
│   ├── 📁 src/
│   │   └── 📓 Wheel_Radius_Estimation.ipynb                  # Batch LS + Affine LS + Recursive LS
│   └── 📁 results/
│       ├── 🖼️  Data_Scatter.png                              # Raw (w, v) sensor data scatter plot
│       ├── 🖼️  Batch_LS_Origin.png                           # Part 1 — fitted line through origin
│       ├── 🖼️  Batch_LS_Offset.png                           # Part 2 — fitted affine line with bias
│       └── 🖼️  Recursive_LS_Convergence.png                  # Part 3 — recursive steps vs batch solution
│
├── 📁 Lab_2/
│   ├── 📄 Readme.md                                          # Lab 2 full documentation
│   ├── 📁 src/
│   │   └── 📓 Landmark_Angle_EKF.ipynb                       # EKF single-step + optional full trajectory
│   └── 📁 results/
│       ├── 🖼️  EKF_Single_Step.png                           # Numerical outputs — P̌_k, K_k, x̂_k, P̂_k
│       └── 🖼️  EKF_Angle_Tracking.png                        # Optional — angle estimate vs measurements
│
├── 📁 Lab_3/
│   ├── 📄 Readme.md                                          # Lab 3 full documentation
│   ├── 📁 src/
│   │   └── 📓 Landmark_Angle_UKF.ipynb                       # UKF sigma-point prediction + correction
│   └── 📁 results/
│       ├── 🖼️  Sigma_Points_Predicted.png                    # Scatter plot of 5 predicted sigma points
│       └── 🖼️  UKF_Single_Step.png                           # Final updated state and covariance
│
├── 📁 Lab_4/
│   ├── 📄 Readme.md                                          # Lab 4 full documentation
│   ├── 📁 src/
│   │   └── 📓 Camera_Calibration_Stereo_Depth.ipynb          # Calibration + stereo depth pipeline
│   ├── 📁 data/
│   │   ├── 📁 calib_images/
│   │   │   └── 🖼️  left_000.jpg … left_018.jpg               # 19 checkerboard calibration images
│   │   ├── 📄 files_management.py                             # Helper — loads stereo projection matrices
│   │   └── 📁 stereo_set/
│   │       ├── 🖼️  frame_00077_1547042741L.png                # Left stereo image
│   │       └── 🖼️  frame_00077_1547042741R.png                # Right stereo image
│   └── 📁 results/
│       ├── 🖼️  Checkerboard_Corners.png                       # Detected corners drawn on calib image
│       ├── 🖼️  Undistorted_Image.png                          # Undistorted output from left_008.jpg
│       ├── 🖼️  Stereo_Pair.png                                # Left and right images side-by-side
│       ├── 🖼️  Disparity_Map.png                              # StereoSGBM dense disparity map
│       ├── 🖼️  Depth_Map.png                                  # Computed depth map (flag colormap)
│       ├── 🖼️  Cross_Correlation_Heatmap.png                  # TM_CCOEFF_NORMED matching heatmap
│       └── 🖼️  Obstacle_Bounding_Box.png                      # Left image with obstacle bounding box
│
└── 📄 README.md                                              # ← You are here
```

---

## 🔬 Course Overview

The **Robot Perception** course develops both the theoretical foundations and the practical engineering skills required to extract reliable information about the world from imperfect robotic sensors. The curriculum spans classical parameter estimation, probabilistic state estimation with the Kalman filter family, and geometric computer vision with stereo cameras — all grounded in the mathematics of optimal estimation and applied to concrete robotic scenarios.

The four labs advance in a deliberate two-phase sequence:

**Phase 1 — Estimation Theory (Labs 1, 2, and 3):** The course begins with the least squares estimator — the mathematically optimal solution to the problem of recovering an unknown parameter from noisy linear measurements. Lab 1 explores this in full: batch estimation through the origin, an affine two-parameter model with sensor bias, and the recursive formulation that processes measurements one-at-a-time using a Kalman gain. Labs 2 and 3 then make a critical leap from static parameter estimation to **dynamic state estimation**: both labs estimate the same evolving position-velocity state from the same non-linear arctangent measurement function, but through two fundamentally different algorithmic lenses. The EKF (Lab 2) approximates the measurement function locally with its first-order Jacobian; the UKF (Lab 3) avoids all approximation by propagating a set of five carefully chosen sigma points through the exact non-linear function and reconstructing the posterior statistics from the transformed cloud.

**Phase 2 — Geometric Vision (Lab 4):** With the estimation framework established, the course pivots to a two-part computer vision pipeline that connects sensor calibration to 3D scene understanding. Part 1 calibrates a monocular camera from 19 checkerboard images using Zhang's planar homography method — recovering the intrinsic matrix K, five distortion coefficients, and per-image extrinsics. Part 2 applies the calibrated stereo projection matrices to compute a dense StereoSGBM disparity map, converts disparity to metric depth using `Z = f·b/d`, and estimates the distance to a detected motorcycle obstacle at 28.764 metres.

```
  Parameter            Dynamic State      Derivative-Free      Geometric
  Estimation           Estimation         State Estimation     Vision
  (Least Squares)      (EKF)              (UKF)                (Stereo)
  ─────────────────    ─────────────────  ─────────────────    ─────────────────
   ┌─────────┐          ┌─────────┐        ┌─────────┐          ┌─────────┐
   │  LAB 1  │─────────►│  LAB 2  │───────►│  LAB 3  │─────────►│  LAB 4  │
   └─────────┘          └─────────┘        └─────────┘          └─────────┘
   Batch LS             Jacobian           Sigma-Point           Zhang Calib
   Through Origin       Linearisation      Transform             8×6 Board × 19
   Affine Model         H_k = ∂h/∂x        2N+1 = 5 Points       StereoSGBM
   Recursive LS         Prediction+Update  Cholesky Spread       28.764 m depth
```

---

## 🧪 Lab Summaries

---

### 🔵 Lab 1 — Wheel Radius Estimation via Least Squares Methods

<div align="center">

[![Lab1](https://img.shields.io/badge/Lab%201-Wheel%20Radius%20Estimation-0078D7?style=flat-square&logo=python&logoColor=white)]()
[![Method](https://img.shields.io/badge/Method-Batch%20%26%20Recursive%20LS-lightblue?style=flat-square)]()
[![Parameters](https://img.shields.io/badge/Estimates-R%CC%82%20%C2%B7%20b%CC%82%20%C2%B7%20Convergence-lightblue?style=flat-square)]()
[![Data](https://img.shields.io/badge/Data-5%20Sensor%20Measurement%20Pairs-lightblue?style=flat-square)]()

</div>

#### 📌 Task Description

> Given **five (w, v) measurement pairs** from a robot's wheel encoder (angular velocity) and accelerometer (linear velocity), estimate the wheel radius `R` and sensor bias `b` satisfying the kinematic model `v = Rw + b` — using three progressively refined methods: batch estimation through the origin, batch estimation with a bias offset, and a recursive estimator with Kalman-gain covariance updates.

This lab establishes the complete **least squares estimation pipeline**, answering the foundational question: *"Given a set of noisy sensor observations from a moving robot, how do you recover the underlying physical parameters with quantified uncertainty — and how does adding a recursive formulation change both the algorithm and the result?"*

**What the task requires:**
- Formulate the **batch LS through origin** problem: construct the measurement matrix `H = [1,1,1,1,1]ᵀ` (single parameter), observation vector `y = v/w` (per-measurement ratio), and apply the normal equations `R̂ = (HᵀH)⁻¹Hᵀy` — equivalent to the mean of the five individual radius estimates `vᵢ/wᵢ`.
- Extend to the **affine two-parameter model** `v = Rw + b` by constructing the two-column measurement matrix `H = [w, 1]` (5×2) and solving the joint normal equations for `x̂_ls = [R̂, b̂]` — interpreting the recovered offset `b̂` as the constant component of systematic sensor noise.
- Implement the **Recursive Least Squares (RLS)** estimator as a student task: initialise from prior beliefs `R̂ ~ N(3, 100)` and `b̂ ~ N(0, 0.04)`, and process one measurement at a time using the three-step Kalman update — computing the gain `K_k`, updating the parameter vector `x̂_k`, and shrinking the covariance `P_k` at each step.
- **Verify convergence:** plot all five recursive line estimates on a single figure overlaid with the batch LS solution (black dashed), confirming that the recursive estimate converges toward the batch result and that the covariance `P_k` decreases monotonically with each new measurement.
- Analyse the **effect of the bias term** — demonstrating that omitting the offset inflates `R̂` by `0.254 m` (from 4.97 m to 5.22 m), a significant error in calibration applications where millimetre-level accuracy is required.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| 🚀 Kinematic Model | `v = R·w`  — no-slip constraint; `v` (m/s) from accelerometer, `w` (rad/s) from encoder |
| 📐 Batch LS Normal Equations | `x̂ = (HᵀH)⁻¹Hᵀy` — optimal closed-form estimate under i.i.d. Gaussian noise |
| 📏 Line Through Origin | `H = ones(5,1)`, `y = v/w` — single-parameter model constraining the fit to pass through zero |
| 📊 Affine Two-Parameter Model | `H = [w, 1]` (5×2) — simultaneous estimation of radius `R` and bias `b` |
| 🔁 Recursive Least Squares | Kalman-gain sequential update — mathematically equivalent to batch LS for linear Gaussian models |
| 📦 Prior Covariance `P₀` | `diag([100, 0.04])` — large variance on `R` (σ=10 m), small on `b` (σ=0.2 m/s) |
| 📉 Kalman Gain | `K_k = P_{k−1}H_kᵀ(H_kP_{k−1}H_kᵀ + σ²)⁻¹` — balances prior uncertainty against measurement noise |
| ✅ Convergence Property | For infinite measurements with uninformative prior, RLS → Batch LS exactly |

#### 📤 Key Parameter Estimation Results

```
Dataset:  5 onboard sensor pairs — angular velocity w (rad/s) + linear velocity v (m/s)
          w = [0.2, 0.3, 0.4, 0.5, 0.6]   v = [1.23, 1.38, 2.06, 2.47, 3.17]
          Measurement noise: σ² = 0.0225 (rad/s)²

                        Estimated Parameters — Three Methods
┌──────────────────────────────────┬──────────────┬───────────────┬─────────────────────────┐
│ Method                           │   R̂  (m)     │   b̂  (m/s)    │ Notes                   │
├──────────────────────────────────┼──────────────┼───────────────┼─────────────────────────┤
│ Batch LS — Through Origin (Pt 1) │   5.2247     │      —        │ Single param, H=ones    │
│ Batch LS — Affine Model  (Pt 2)  │   4.9700     │    0.0740     │ Two params, H=[w, 1]    │
│ Recursive LS             (Pt 3)  │   5.0502     │    0.0377     │ Prior influences result │
└──────────────────────────────────┴──────────────┴───────────────┴─────────────────────────┘
Bias impact: omitting b̂ inflates R̂ by 0.254 m (5.1% error)
Prior residual: 5 measurements insufficient to fully overcome P₀ = diag([100, 0.04])
```

📂 **[→ View Lab 1 Full Documentation](https://github.com/umerahmedbaig7/Robot-Perception/blob/main/Lab_1/Readme.md)**

---

### 🟠 Lab 2 — 1D Position Estimation via Extended Kalman Filter

<div align="center">

[![Lab2](https://img.shields.io/badge/Lab%202-Extended%20Kalman%20Filter-E85D04?style=flat-square&logo=python&logoColor=white)]()
[![Method](https://img.shields.io/badge/Method-Jacobian%20Linearisation-orange?style=flat-square)]()
[![Output](https://img.shields.io/badge/Output-1D%20Position%20%26%20Velocity-orange?style=flat-square)]()
[![Scope](https://img.shields.io/badge/Demo-Single--Step%20%2B%20Optional%20Trajectory-orange?style=flat-square)]()

</div>

#### 📌 Task Description

> A robot moves along a 1D track while a camera measures the **elevation angle** `φ = arctan(S/(D−p))` to a stationary landmark of known height `S = 20 m` at horizontal distance `D = 40 m`. Because this measurement function is non-linear in the robot's position, a standard Kalman Filter cannot be applied directly — implement one complete **EKF prediction-update cycle** at `Δt = 0.5 s`, computing all Jacobians, the Kalman gain, and the corrected state and covariance from the first angle measurement `y₁ = π/6 rad`.

This lab makes the critical leap from static parameter estimation to **dynamic non-linear state estimation** — the state `[p, ṗ]ᵀ` evolves over time, the motion model is linear (constant-acceleration kinematics), but the measurement model `h(p) = arctan(S/(D−p))` is non-linear, demanding the Extended Kalman Filter's Jacobian linearisation approach.

**What the task requires:**
- Set up the **state space**: state vector `x = [p, ṗ]ᵀ`, control input `u = a = −2 m/s²`, discrete-time state transition `F = [[1, Δt], [0, 1]]`, and control influence `G = [[0], [Δt]]` — then execute the linear prediction step to obtain `x̌_k` and `P̌_k`.
- Derive and compute the **measurement Jacobian** `H_k = [S / ((D − p̌_k)² + S²), 0]` — the critical linearisation step evaluated at the predicted position `p̌_k = 2.5 m`, capturing how a one-metre position change affects the measured angle at the current operating point.
- Compute the **Kalman gain** `K_k = P̌_k H_kᵀ (H_k P̌_k H_kᵀ + R)⁻¹` — a (2×1) vector whose entries reveal the relative weight given to the angle measurement in correcting position versus velocity.
- Apply the **measurement update**: evaluate the true non-linear predicted measurement `h(x̌_k) = arctan(S/(D−p̌_k))` (not the Jacobian approximation), compute the innovation `y₁ − h(x̌_k)`, and correct both the state and covariance.
- Optionally extend to a **full 20-second recursive trajectory** (200 steps, `Δt = 0.1 s`, sinusoidal control `u_k = 0.5 cos(t)`) — implementing `motion_iterate()` and `measurement_update()` functions, and plotting the EKF angle estimate against 200 pre-recorded noisy measurements.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| 📐 Non-linear Measurement | `h(p) = arctan(S / (D − p))` — angle is a trigonometric function of position, not linear |
| 🔄 EKF Linearisation | Replace `h(·)` with its first-order Taylor expansion at the predicted state each step |
| 🧮 Measurement Jacobian `H_k` | `∂h/∂x \|_{x̌_k}` = `[S/((D−p̌_k)²+S²), 0]` — state-dependent, recomputed every step |
| 🏃 Prediction Step | `x̌_k = Fx̂_{k−1} + Gu_{k−1}`, `P̌_k = FP̂_{k−1}Fᵀ + Q` — linear, no approximation |
| 🎯 Innovation | `y₁ − h(x̌_k)` — uses the exact non-linear function, not the linearised Jacobian |
| 📊 Kalman Gain `K_k` | `P̌_k H_kᵀ (H_k P̌_k H_kᵀ + R)⁻¹` — (2×1) vector for [position, velocity] correction |
| 📉 Covariance Update | `P̂_k = (I − K_k H_k) P̌_k` — uncertainty shrinks with each informative measurement |
| 🔁 Optional Extension | 200-step recursive filter — `motion_iterate()` + `measurement_update()` loop |

#### 📤 Key Single-Step EKF Results (Δt = 0.5 s)

```
System:  x̂₀ = [[0], [5]]   P̂₀ = diag([0.01, 1.0])   u₀ = −2 m/s²   y₁ = π/6 rad
         Landmark: S = 20 m,  D = 40 m;  Q = 0.1·I₂ₓ₂,  R = 0.01 rad²

                         EKF Single-Step Numerical Outputs
┌──────────────────────────────────────┬────────────────────────────────────────┐
│ Quantity                             │ Value                                  │
├──────────────────────────────────────┼────────────────────────────────────────┤
│ Predicted state   x̌_k                │ [[2.5 m],  [4.0 m/s]]                  │
│ Predicted covariance  P̌_k            │ [[0.36, 0.50], [0.50, 1.10]]           │
│ Measurement Jacobian  H_k            │ [0.01264,   0]       (1×2)             │
│ Kalman gain  K_k                     │ [[0.40],   [0.55]]   (2×1)             │
│ Predicted measurement  h(x̌_k)        │ 0.4900 rad                             │
│ Innovation  y₁ − h(x̌_k)              │ 0.0336 rad  (= π/6 − 0.4900)           │
│ Updated state  x̂_k                   │ [[2.51 m],  [4.02 m/s]]                │
│ Updated covariance  P̂_k              │ [[0.36, 0.50], [0.50, 1.10]]           │
└──────────────────────────────────────┴────────────────────────────────────────┘
Gain interpretation: position correction (0.40) < velocity correction (0.55)
because σ²_p = 0.01 ≪ σ²_ṗ = 1.0 — filter trusts initial position more than velocity
```

📂 **[→ View Lab 2 Full Documentation](https://github.com/umerahmedbaig7/Robot-Perception/blob/main/Lab_2/Readme.md)**

---

### 🟣 Lab 3 — 1D Position Estimation via Unscented Kalman Filter (Sigma-Point Method)

<div align="center">

[![Lab3](https://img.shields.io/badge/Lab%203-Unscented%20Kalman%20Filter-7B2FBE?style=flat-square&logo=python&logoColor=white)]()
[![Method](https://img.shields.io/badge/Method-Sigma--Point%20Transform-purple?style=flat-square)]()
[![Points](https://img.shields.io/badge/Sigma%20Points-2N%2B1%20%3D%205-purple?style=flat-square)]()
[![Compare](https://img.shields.io/badge/Comparison-UKF%20vs%20EKF%20(Lab%202)-purple?style=flat-square)]()

</div>

#### 📌 Task Description

> Using the **identical physical setup as Lab 2** — the same 1D robot, landmark geometry, initial conditions, and first angle measurement — implement one complete **UKF prediction-update cycle** using the sigma-point transform. Instead of computing a Jacobian, generate five deterministic sigma points from the prior distribution via Cholesky decomposition, propagate each through the exact non-linear functions, and reconstruct the posterior mean and covariance from the weighted sigma-point cloud. Compare all numerical outputs against the EKF results from Lab 2.

This lab demonstrates the **derivative-free alternative** to EKF linearisation — the UKF achieves second-order accuracy in capturing the effect of the non-linearity without ever requiring an analytical derivative of `h(p) = arctan(S/(D−p))`. The sigma-point approach uses the measurement function as a **black box**, making it directly applicable to models where Jacobians are difficult or impossible to derive analytically.

**What the task requires:**
- Compute the **UKF tuning parameters**: state dimension `N = 2`, tuning parameter `κ = 3 − N = 1`, scaling factor `√(N+κ) = √3 ≈ 1.7321`, and the five scalar weights `a₀ = 1/3` (central) and `a₁...₄ = 1/6` (symmetric).
- Generate **5 prediction-step sigma points** from `(x̂₀, P̂₀)` via Cholesky decomposition: `chol(P₀) = [[0.1, 0], [0, 1.0]]`, spreading the central point ±√3 times each column of the Cholesky factor to produce the symmetric cloud.
- Propagate all 5 sigma points through the motion model `χ̌ = Fχ + Gu`, then reconstruct the **weighted predicted mean** `x̌_k = Σ aᵢ·χ̌⁽ⁱ⁾` and **weighted predicted covariance** `P̌_k = Σ aᵢ·(χ̌⁽ⁱ⁾ − x̌_k)(χ̌⁽ⁱ⁾ − x̌_k)ᵀ + Q`.
- Re-generate **5 correction-step sigma points** from `(x̌_k, P̌_k)` using `chol(P̌_k) = [[0.6, 0], [0.833, 0.637]]`, propagate each through the exact measurement function `ζ⁽ⁱ⁾ = arctan(S/(D − γ⁽ⁱ⁾[0]))`, and compute the **predicted measurement mean** `ŷ_k`, **measurement covariance** `P_yy`, and **cross-covariance** `P_xy` between the state and measurement sigma clouds.
- Apply the UKF Kalman gain `K = P_xy / P_yy` and correction equations — then compare `K`, `x̂_k`, and `P̂_k` against the EKF outputs, confirming the ~0.8% position-gain difference arising from the UKF's more faithful non-linear propagation.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| 🔵 Sigma-Point Transform | 2N+1 = 5 deterministic points that exactly capture mean and covariance of the prior |
| 📐 Cholesky Spread | `chol(P)[:,i]` gives spread direction — sigma pts cover the elliptical uncertainty shape |
| ⚖️ Weights | `a₀ = κ/(N+κ) = 1/3` (central), `a₁...₄ = 1/(2(N+κ)) = 1/6` (symmetric); sum = 1 |
| 🔄 No Jacobian Required | `h(·)` used as a black box — no analytical derivative of `arctan` needed |
| 📊 Cross-Covariance `P_xy` | `Σ aᵢ·(γ⁽ⁱ⁾ − x̌)(ζ⁽ⁱ⁾ − ŷ)ᵀ` — captures state-measurement statistical coupling |
| 🧮 Measurement Covariance `P_yy` | `Σ aᵢ·(ζ⁽ⁱ⁾ − ŷ)² + R` — scalar for 1D measurement + noise |
| 🎯 UKF Kalman Gain | `K = P_xy / P_yy` — divides (2×1) cross-cov by scalar measurement cov |
| ⚡ Second-Order Accuracy | Sigma-point statistics capture curvature effects that first-order Jacobian misses |

#### 📤 Key Single-Step UKF Results — with EKF Comparison

```
System:  Identical to Lab 2 — x̂₀, P̂₀, u₀, y₁, S, D, Q, R all unchanged
         UKF tuning: N=2, κ=1, scale=√3≈1.7321, weights: a₀=1/3, a₁...₄=1/6

                    5 Correction Sigma Points → Measurement Propagation
┌───────┬──────────────────────────────────────┬────────────────────┐
│ Point │ State  [position (m), velocity (m/s)]│ Angle ζ⁽ⁱ⁾ (rad)   │
├───────┼──────────────────────────────────────┼────────────────────┤
│ γ⁽⁰⁾  │ [2.5000,  4.0000]                    │ 0.48996            │
│ γ⁽¹⁾  │ [3.5392,  5.4434]                    │ 0.50172  ←max      │
│ γ⁽²⁾  │ [1.4608,  2.5566]                    │ 0.47869  ←min      │
│ γ⁽³⁾  │ [2.5000,  5.1030]                    │ 0.48996  (=γ⁽⁰⁾)   │
│ γ⁽⁴⁾  │ [2.5000,  2.8970]                    │ 0.48996  (=γ⁽⁰⁾)   │
└───────┴──────────────────────────────────────┴────────────────────┘
ζ⁽³⁾=ζ⁽⁴⁾=ζ⁽⁰⁾: velocity-only spread → angle h(p) unchanged → measurement independent of ṗ

               UKF vs EKF — Quantitative Single-Step Comparison
┌──────────────────────────┬──────────────────────────┬──────────────────────────┬────────────┐
│ Quantity                 │ EKF (Lab 2)              │ UKF (Lab 3)              │ Difference │
├──────────────────────────┼──────────────────────────┼──────────────────────────┼────────────┤
│ Predicted mean   x̌_k     │ [[2.5], [4.0]]           │ [[2.5], [4.0]]           │ Identical  │
│ Predicted cov    P̌_k     │ [[0.36,0.50],[0.50,1.10]]│ [[0.36,0.50],[0.50,1.10]]│ Identical  │
│ Kalman gain K[0]         │ 0.400                    │ 0.397                    │ ~0.8%      │
│ Kalman gain K[1]         │ 0.551                    │ 0.551                    │ Negligible │
│ Updated position (m)     │ 2.510                    │ 2.513                    │ 3 mm       │
│ Updated velocity (m/s)   │ 4.018                    │ 4.019                    │ Negligible │
│ Updated cov P̂_k[0,0]     │ 0.3600                   │ 0.3584                   │ ~0.4%      │
└──────────────────────────┴──────────────────────────┴──────────────────────────┴────────────┘
Prediction step: IDENTICAL — linear motion model means sigma-point propagation = linear Kalman
Correction step: small differences — UKF samples arctan at 5 positions vs EKF's single Jacobian
```

📂 **[→ View Lab 3 Full Documentation](https://github.com/umerahmedbaig7/Robot-Perception/blob/main/Lab_3/Readme.md)**

---

### 🟢 Lab 4 — Camera Calibration and Stereo Depth Estimation

<div align="center">

[![Lab4](https://img.shields.io/badge/Lab%204-Camera%20Calibration%20%26%20Stereo%20Depth-2E7D32?style=flat-square&logo=opencv&logoColor=white)]()
[![Part1](https://img.shields.io/badge/Part%201-Monocular%20Calibration%20(Zhang)-green?style=flat-square)]()
[![Part2](https://img.shields.io/badge/Part%202-StereoSGBM%20%7C%20Depth%20Map-green?style=flat-square)]()
[![Result](https://img.shields.io/badge/Obstacle%20Distance-28.764%20m-brightgreen?style=flat-square)]()

</div>

#### 📌 Task Description

> A two-part computer vision pipeline: **Part 1** calibrates a monocular camera from 19 images of an 8×6 checkerboard — recovering the intrinsic matrix `K`, five distortion coefficients, and per-image extrinsics via Zhang's method — then applies the calibrated parameters to undistort a sample image. **Part 2** takes a pre-calibrated stereo camera pair, computes a dense StereoSGBM disparity map, decomposes the projection matrices into `K`, `R`, `t`, generates a full metric depth map using `Z = f·b/d`, localises a motorcycle obstacle via template matching, and reports the **nearest-point obstacle distance as 28.764 metres**.

This lab marks the complete **paradigm shift from probabilistic estimation to geometric computer vision** — from tracking a 1D scalar state with Kalman equations to recovering the full 3D metric structure of a real driving scene from pixel correspondences between two calibrated cameras.

**What the task requires:**
- Set up the **checkerboard object points**: a `(1, 48, 3)` array of 3D corner positions in the checkerboard plane (Z=0), spaced 25 mm apart, using `np.mgrid` — then loop over all 19 calibration images, detecting corners with `cv2.findChessboardCorners()` and refining to sub-pixel accuracy with `cv2.cornerSubPix()` (search window (8,6), ε=0.001, max 30 iterations).
- Call `cv2.calibrateCamera()` to solve for all parameters simultaneously from the 19-image overconstrained system — recovering `K = [[379.97, 0, 163.13], [0, 204.10, 135.53], [0, 0, 1]]` and distortion coefficients `dist = [−0.5444, −0.2525, −0.0377, −0.1740, 0.0748]` indicating strong barrel distortion.
- **Undistort** a sample image using `getOptimalNewCameraMatrix()` + `initUndistortRectifyMap()` + `remap()` with bilinear interpolation, then crop to the valid pixel region returned by the ROI.
- Implement `compute_left_disparity_map()` using **StereoSGBM** with `numDisparities=96`, `blockSize=7`, `P1=392`, `P2=1568`, and `mode=STEREO_SGBM_MODE_SGBM_3WAY` — dividing the 16-bit fixed-point output by 16.0 to recover true sub-pixel disparity.
- Implement `decompose_projection_matrix()` using `cv2.decomposeProjectionMatrix()` with a **homogeneous divide** `t = t[:3] / t[3]` — recovering `f=640 px`, `b = ‖t_left − t_right‖ = 0.5 m`, and `R = I₃ₓ₃` for both cameras.
- Implement `calc_depth_map()` using `Z = (f·b)/d` with invalid disparity values (≤0) replaced by `0.1` — then implement `locate_obstacle_in_image()` using `cv2.matchTemplate(TM_CCOEFF_NORMED)` and `cv2.minMaxLoc()` to locate the motorcycle at `(547, 479)`, and `calculate_nearest_point()` to report the closest valid depth in the cropped bounding box region.

#### 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| 📷 Pinhole Camera Model | `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` — maps 3D world points to 2D pixel coordinates |
| 📐 Zhang's Method | Planar homography from 2D↔3D checkerboard correspondences — solved via least squares over 19 views |
| 🌀 Distortion Model | `dist = [k1, k2, p1, p2, k3]` — radial + tangential coefficients; k1=−0.5444 → strong barrel |
| 🔷 StereoSGBM | Semi-Global Block Matching — energy minimisation with P1/P2 smoothness penalties |
| 📦 Fixed-Point Disparity | 16-bit output ÷ 16 → true float disparity (sub-pixel precision: 1/16 pixel) |
| 🔁 Projection Decomposition | `cv2.decomposeProjectionMatrix(P)` + homogeneous divide → `K`, `R`, `t` |
| 📏 Stereo Depth Equation | `Z = f·b/d` — depth grows inversely with disparity; error ∝ Z² at long range |
| 🎯 Template Matching | `TM_CCOEFF_NORMED` normalised cross-correlation → obstacle at pixel `(547, 479)` |

#### 📤 Key Calibration and Stereo Depth Results

```
Part 1 — Camera Calibration
  Dataset:  19 checkerboard images  (left_000.jpg … left_018.jpg)
  Board:    8×6 interior corners,  25 mm squares,  48 corners per image

  Recovered intrinsic matrix K:
  ┌──────────────────────────────────────────────────────────────┐
  │  K = [[379.97,    0.00,  163.13],                            │
  │       [  0.00,  204.10,  135.53],                            │
  │       [  0.00,    0.00,    1.00]]                            │
  │  fx ≠ fy (2:1 ratio) — pixel non-squareness in lens optics   │
  └──────────────────────────────────────────────────────────────┘
  Distortion coefficients:  k1=−0.5444  k2=−0.2525  p1=−0.0377  p2=−0.1740  k3=0.0748
  Dominant negative k1 → strong barrel distortion (straight lines curve inward)

Part 2 — Stereo Depth Pipeline
  Stereo pair:  frame_00077_1547042741L/R.png
  Decomposed:   f = 640 px   |   baseline b = ‖t_left − t_right‖ = 0.50 m
                r_left = r_right = I₃ₓ₃  (parallel camera alignment)

  Depth equation:  Z = (640 × 0.5) / d  =  320 / d  metres

┌──────────────────────────────────────┬───────────────────────────────────────┐
│ Detection Step                       │ Result                                │
├──────────────────────────────────────┼───────────────────────────────────────┤
│ Obstacle crop (motorcycle)           │ img_left[479:509, 547:593]  (30×46 px)│
│ Template match location              │ (547, 479) — top-left corner pixel    │
│ Disparity at obstacle                │ d ≈ 11.1 px                           │
│ Nearest-point obstacle depth         │ 28.764 m                              │
│ Depth error per 1-px disparity error │ ΔZ ≈ 2.6 m  (inverse-square growth)   │
└──────────────────────────────────────┴───────────────────────────────────────┘
```

📂 **[→ View Lab 4 Full Documentation](https://github.com/umerahmedbaig7/Robot-Perception/blob/main/Lab_4/Readme.md)**

---

## 🔗 Progressive Learning Pathway

The four labs form an interlocking progression where each lab builds directly on the mathematical framework and intuitions of the previous one. The dependency is not merely thematic — the Kalman gain formula introduced in Lab 1, the prediction-update cycle structure established in Lab 2, and the landmark geometry shared between Labs 2 and 3 are carried forward explicitly, building cumulative depth at each step.

```
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║                        ROBOT PERCEPTION — COURSE PROGRESSION                               ║
╠════════════════════╦════════════════════╦═════════════════════╦════════════════════════════╣
║   🔵 LAB 1 🔵     ║   🟠 LAB 2 🟠     ║   🟣 LAB 3 🟣      ║       🟢 LAB 4 🟢         ║
║  Wheel Radius      ║  1D Position       ║  1D Position        ║  Camera Calibration        ║
║  Estimation        ║  via EKF           ║  via UKF            ║  & Stereo Depth            ║
╠════════════════════╬════════════════════╬═════════════════════╬════════════════════════════╣
║  Static Parameter  ║  Dynamic State     ║  Dynamic State      ║  Geometric Vision          ║
║  Estimation        ║  Estimation        ║  Estimation         ║  Pipeline                  ║
║  (Known Physics)   ║  (Non-linear Meas) ║  (Derivative-Free)  ║  (Calib + 3D Depth)        ║
╠════════════════════╬════════════════════╬═════════════════════╬════════════════════════════╣
║  Batch LS origin   ║  Jacobian H_k      ║  Sigma-Point        ║  Zhang Checkerboard        ║
║  Batch LS affine   ║  = ∂h/∂x at x̌_k    ║  Transform (SPT)    ║  Corner Detection          ║
║  Recursive LS      ║  Prediction step   ║  Cholesky Spread    ║  StereoSGBM Disparity      ║
║  Kalman gain K     ║  Measurement update║  2N+1 = 5 points    ║  Z = f·b/d  depth map      ║
╠════════════════════╬════════════════════╬═════════════════════╬════════════════════════════╣
║  v = Rw + b        ║  x = [p, ṗ]ᵀ       ║  x = [p, ṗ]ᵀ        ║  19 calib images           ║
║  n = 5 sensor pairs║  φ=arctan(S/(D−p)) ║  φ=arctan(S/(D−p))  ║  K: fx=379.97, fy=204.10   ║
║  σ² = 0.0225       ║  Δt=0.5 s, F,G,Q   ║  κ=1, scale=√3      ║  b = 0.5 m  (baseline)     ║
║  R̂ = 5.05 m        ║  K=[[0.40],[0.55]] ║  K=[[0.397],[0.551]]║  depth = 28.764 m          ║
╠════════════════════╬════════════════════╬═════════════════════╬════════════════════════════╣
║  Establishes the   ║  Extends Lab 1:    ║  Extends Lab 2:     ║  Complete paradigm shift   ║
║  estimation frame- ║  static parameter  ║  Jacobian EKF →     ║  from estimation theory    ║
║  work — Kalman     ║  → dynamic state   ║  derivative-free    ║  to geometric computer     ║
║  gain and cov      ║  + non-linear meas ║  sigma-point UKF    ║  vision — calib + depth    ║
╚════════════════════╩════════════════════╩═════════════════════╩════════════════════════════╝
```

**Dependency chain:**

- 🔵 **Lab 1** establishes the complete estimation infrastructure — the batch normal equations `x̂ = (HᵀH)⁻¹Hᵀy`, the recursive Kalman gain formula `K = P Hᵀ(HPHᵀ + R)⁻¹`, and the three-step update cycle `K → x̂ → P`. The covariance update `P_k = (I − KH)P_{k−1}` derived here is reused verbatim in Labs 2 and 3. The key insight that a larger prior variance produces a larger Kalman gain — first observed in Lab 1's `P₀ = diag([100, 0.04])` — recurs in Lab 2's asymmetric gain `K = [[0.40], [0.55]]` arising from `σ²_p = 0.01 ≪ σ²_ṗ = 1.0`.

- 🟠 **Lab 2** inherits the Kalman gain structure from Lab 1 and extends it to a dynamic state with a non-linear measurement function. It introduces two new components carried into Lab 3: the prediction step `(x̌_k, P̌_k)` using the state transition matrix `F` and process noise `Q`, and the landmark geometry `S = 20 m`, `D = 40 m` that defines the arctangent measurement model. The initial conditions `x̂₀ = [[0], [5]]`, `P̂₀ = diag([0.01, 1.0])`, `u₀ = −2 m/s²`, `y₁ = π/6 rad` are passed forward unchanged into Lab 3 — allowing a direct numerical comparison of EKF and UKF on the identical problem.

- 🟣 **Lab 3** inherits the entire physical setup from Lab 2 — state space, landmark geometry, initial conditions, motion model, and the first angle measurement — changing only the algorithmic approach to the non-linear measurement update. By replacing the single Jacobian evaluation at `p̌_k` with a five-point sigma cloud propagated through the exact arctangent function, the UKF demonstrates both higher-order accuracy and complete independence from analytical derivatives. The direct numerical comparison (prediction step identical, correction step differing by ~0.8%) gives students a controlled experiment isolating precisely what the sigma-point transform contributes over Jacobian linearisation.

- 🟢 **Lab 4** makes a deliberate architectural shift that reflects the structure of real robotic perception systems: estimation theory (Labs 1–3) equips the robot with the mathematics to process sensor data optimally, while geometric vision (Lab 4) provides the tools to extract metric 3D structure from images. The stereo depth pipeline implicitly contains its own estimation problem — the disparity computation via StereoSGBM minimises a global energy function over the image, closely analogous to the energy-minimisation framing introduced in Lab 1's least squares derivation.

---

## ⚙️ Shared Mathematical Framework — The Kalman Update Paradigm

### 🔢 The Unifying Gain-Update-Covariance Pattern

All three estimation labs (Labs 1, 2, 3) are unified by a single three-step update pattern. The Kalman gain formula, state correction, and covariance shrinkage appear in identical mathematical form across all three — differing only in what `x_k`, `H_k`, and `h_k` represent in each context:

```python
# Core Kalman update pattern — present in Labs 1, 2, and 3
# ---------------------------------------------------------
# Kalman gain:       (2×1) for two-param models, scales measurement trust
K_k = P_k @ H_k.T @ np.linalg.inv(H_k @ P_k @ H_k.T + R_k)

# State update:      add gain-scaled innovation to current estimate
x_k = x_k + K_k @ (y_k - h_k)

# Covariance update: posterior uncertainty always less than prior
P_k = (np.eye(n) - K_k @ H_k) @ P_k
```

The table below shows exactly how these three symbols are specialised in each lab:

| 📁 Lab | `x_k` (state/params) | `H_k` (measurement matrix) | `h_k` (predicted meas.) | `R_k` (noise) |
|:------:|:--------------------:|:--------------------------:|:-----------------------:|:--------------:|
| Lab 1 | `[R, b]` (2,) — wheel params | `[w_k, 1]` (1×2) row | `w_k·R + b` (linear) | `0.0225` scalar |
| Lab 2 | `[p, ṗ]ᵀ` (2,1) — robot state | `[S/((D−p̌)²+S²), 0]` (1×2) Jacobian | `arctan(S/(D−p̌_k))` exact | `0.01` (1,1) |
| Lab 3 | `[p, ṗ]ᵀ` (2,1) — robot state | `P_xy / P_yy` (equiv. gain) | `Σ aᵢ·ζ⁽ⁱ⁾` sigma-point mean | `0.01` + `P_yy` |

**Key insight**: Lab 1 introduces the formula; Lab 2 makes `H_k` state-dependent (recomputed every step as the linearised Jacobian); Lab 3 replaces the explicit Jacobian with a sigma-cloud-derived equivalent — but all three execute the identical gain-update-covariance cycle.

---

### 📐 Shared Landmark Geometry — Labs 2 and 3

Labs 2 and 3 use a completely identical physical problem, initial conditions, and measurement — enabling a pure algorithmic comparison between EKF and UKF:

```
Shared Setup (Labs 2 & 3)
  ─────────────────────────────────────────────────────────────────────
  State vector:      x = [p, ṗ]ᵀ        position and velocity
  Initial state:     x̂₀ = [[0], [5]]    origin, 5 m/s toward landmark
  Initial cov:       P̂₀ = diag([0.01, 1.0])
  Control:           u₀ = −2 m/s²        decelerating
  Time step:         Δt = 0.5 s

  Landmark geometry: S = 20 m (height),  D = 40 m (horizontal distance)
  Measurement:       y₁ = π/6 rad  (first camera angle observation)
  Measurement noise: R = 0.01 rad²
  Process noise:     Q = 0.1 · I₂ₓ₂

  Measurement model: h(p) = arctan( S / (D − p) )     [non-linear in p]
  ─────────────────────────────────────────────────────────────────────
```

---

### 🔁 Mathematical Components Introduced Per Lab

| Component | Introduced | Role | Used In |
|-----------|:----------:|------|:-------:|
| Batch normal equations `(HᵀH)⁻¹Hᵀy` | Lab 1 | Closed-form least squares for linear models | Lab 1 |
| Kalman gain `K = PHᵀ(HPHᵀ+R)⁻¹` | Lab 1 | Optimal measurement weighting | Labs 1, 2, 3 |
| Covariance update `P = (I − KH)P` | Lab 1 | Posterior uncertainty quantification | Labs 1, 2, 3 |
| State transition `F`, control matrix `G` | Lab 2 | Discrete-time constant-acceleration kinematics | Labs 2, 3 |
| Prediction step `x̌ = Fx̂ + Gu`, `P̌ = FP̂Fᵀ + Q` | Lab 2 | Dynamic state propagation | Labs 2, 3 |
| Measurement Jacobian `H_k = ∂h/∂x` at `x̌_k` | Lab 2 | EKF first-order linearisation of `arctan` | Lab 2 |
| `np.linalg.cholesky(P)` — covariance square root | Lab 3 | Sigma-point spread directions | Lab 3 |
| Sigma-point weights `a₀ = κ/(N+κ)`, `aᵢ = 1/(2(N+κ))` | Lab 3 | Weighted reconstruction of mean and cov | Lab 3 |
| Cross-covariance `P_xy = Σ aᵢ·(γ⁽ⁱ⁾−x̌)(ζ⁽ⁱ⁾−ŷ)ᵀ` | Lab 3 | UKF state-measurement coupling | Lab 3 |
| `cv2.calibrateCamera()` — Zhang's method | Lab 4 | Monocular intrinsic parameter recovery | Lab 4 |
| `cv2.StereoSGBM_create()` — dense disparity | Lab 4 | Semi-global block matching | Lab 4 |
| Stereo depth `Z = f·b/d`, `cv2.decomposeProjectionMatrix()` | Lab 4 | Metric 3D reconstruction from stereo pair | Lab 4 |
| `cv2.matchTemplate(TM_CCOEFF_NORMED)` | Lab 4 | Obstacle localisation via cross-correlation | Lab 4 |

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/umerahmedbaig7/Robot-Perception.git
cd Robot-Perception
```

### 2️⃣ Install Dependencies

Labs 1–3 require only NumPy and Matplotlib, available in all standard Python environments. Lab 4 additionally requires OpenCV and, for data download, `gdown`. For a clean local environment:

```bash
# Labs 1, 2, 3 — pure NumPy + Matplotlib
pip install numpy matplotlib jupyter

# Lab 4 — adds OpenCV and Colab download utility
pip install opencv-python gdown numpy matplotlib jupyter
```

> 📌 For a clean local virtual environment:
> ```bash
> python -m venv rp_env
> source rp_env/bin/activate          # Windows: rp_env\Scripts\activate
> pip install numpy matplotlib opencv-python gdown jupyter
> ```

### 3️⃣ Open in Jupyter or Google Colab

All labs are self-contained Jupyter notebooks compatible with both local Jupyter and Google Colab. For Labs 1–3, no additional data download is required — all measurement data is defined inline within the notebook. For Lab 4 on Colab:

```python
# Lab 4 — download and extract calibration + stereo data from Google Drive
!gdown 1qWYfvjRS8W54Mro743EOuY2N06OohN2x          # downloads lab5_files.rar
!unrar x /content/lab5_files.rar                    # extracts calib_images/ + stereo_set/
```

> ⚠️ Lab 4 is designed for **Google Colab** where `gdown` and `unrar` are pre-installed. For local execution, download and place the `lab5_files/` directory manually in the working directory before running any data-dependent cells.

### 4️⃣ Run Each Lab

Execute all cells sequentially in each notebook (**Cell → Run All** in Jupyter, **Runtime → Run all** in Colab):

```
# 🔵 Lab 1 — Wheel Radius Estimation  (< 1 min, any platform)
Open:  Lab_1/src/Wheel_Radius_Estimation.ipynb

# 🟠 Lab 2 — Extended Kalman Filter   (< 1 min, any platform)
Open:  Lab_2/src/Landmark_Angle_EKF.ipynb

# 🟣 Lab 3 — Unscented Kalman Filter  (< 1 min, any platform)
Open:  Lab_3/src/Landmark_Angle_UKF.ipynb

# 🟢 Lab 4 — Camera Calib + Stereo Depth  (~1 min, Colab recommended)
Open:  Lab_4/src/Camera_Calibration_Stereo_Depth.ipynb
```

| 📁 Lab | Notebook | Platform | Estimated Runtime |
|:------:|----------|:--------:|:-----------------:|
| Lab 1 | `Wheel_Radius_Estimation.ipynb` | Jupyter / Colab | < 1 min |
| Lab 2 | `Landmark_Angle_EKF.ipynb` | Jupyter / Colab | < 1 min |
| Lab 3 | `Landmark_Angle_UKF.ipynb` | Jupyter / Colab | < 1 min |
| Lab 4 | `Camera_Calibration_Stereo_Depth.ipynb` | Colab (recommended) | ~40–60 s |

> 💡 Labs 1–3 contain no external data dependencies — all sensor measurements and system parameters are hardcoded within the notebooks. They run instantly on any machine with NumPy and Matplotlib installed, including offline environments.

---

## 📊 Results at a Glance

### 🔵 Lab 1 — Wheel Radius and Bias Estimation

| Method | R̂ (m) | b̂ (m/s) | Converges To | Pass ✓ |
|:------:|:------:|:--------:|:------------:|:------:|
| Batch LS — Through Origin | **5.2247** | — | — | ✅ |
| Batch LS — Affine Model | **4.9700** | **0.0740** | — | ✅ |
| Recursive LS (5 steps) | **5.0502** | **0.0377** | Approaches batch solution | ✅ |

*Bias impact: omitting `b̂` inflates `R̂` by **0.254 m** (5.1%) — significant in precision calibration applications.*

### 🟠 Lab 2 — EKF Single-Step Results (Δt = 0.5 s)

| Quantity | Predicted (Pre-update) | Updated (Post-update) |
|:--------:|:----------------------:|:---------------------:|
| Position `p` (m) | 2.500 | **2.510** |
| Velocity `ṗ` (m/s) | 4.000 | **4.020** |
| Cov `P[0,0]` (m²) | 0.360 | **0.360** |
| Cov `P[1,1]` (m/s)² | 1.100 | **1.100** |
| Measured angle innovation | — | **0.0336 rad** |

*Kalman gain `K = [[0.40], [0.55]]` — velocity correction larger than position correction because `σ²_ṗ ≫ σ²_p`.*

### 🟣 Lab 3 — UKF Single-Step vs EKF Comparison

| Quantity | EKF (Lab 2) | UKF (Lab 3) | Δ Difference |
|:--------:|:-----------:|:-----------:|:------------:|
| Kalman gain K[0] | 0.400 | **0.397** | 0.8% |
| Kalman gain K[1] | 0.551 | **0.551** | < 0.1% |
| Updated position (m) | 2.510 | **2.513** | 3 mm |
| Updated velocity (m/s) | 4.018 | **4.019** | < 0.1% |
| Updated cov `P̂[0,0]` | 0.3600 | **0.3584** | 0.4% |

*Prediction step: identical (linear motion model). Correction step: small difference — UKF propagates 5 sigma points vs EKF's single Jacobian evaluation.*

### 🟢 Lab 4 — Calibration and Stereo Depth Results

| Part | Quantity | Value |
|:----:|----------|:-----:|
| Calibration | Focal lengths `(fx, fy)` | **379.97 px, 204.10 px** |
| Calibration | Principal point `(cx, cy)` | **163.13 px, 135.53 px** |
| Calibration | Barrel distortion `k1` | **−0.5444** |
| Calibration | Calibration images used | **19 / 19** |
| Stereo | Focal length (stereo) `f` | **640 px** |
| Stereo | Baseline `b` | **0.50 m** |
| Stereo | Disparity at obstacle | **≈ 11.1 px** |
| Stereo | **Obstacle nearest depth** | **28.764 m** |

---

## 🧰 Tech Stack

<div align="center">

| 🛠️ Tool | 🔖 Version | 🎯 Role in This Course | 🧪 Used In |
|:-------:|:---------:|:---------------------:|:----------:|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | 3.9+ | Core language — all notebooks, data pipelines, estimation algorithms | All |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | ≥ 1.21 | Matrix ops — `linalg.inv()`, `linalg.cholesky()`, `@` operator, `eye()`, `zeros()`, `arctan()`, `mgrid()` | All |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=python&logoColor=white) | ≥ 3.4 | Scatter plots, fitted lines, convergence figures, sigma-point scatter, depth maps, bounding box overlays | All |
| ![OpenCV](https://img.shields.io/badge/-OpenCV-27AE60?logo=opencv&logoColor=white) | ≥ 4.5 | `calibrateCamera`, `findChessboardCorners`, `StereoSGBM`, `decomposeProjectionMatrix`, `matchTemplate` | Lab 4 |
| ![glob](https://img.shields.io/badge/-glob-grey?logo=python&logoColor=white) | stdlib | Image path enumeration for calibration image loop | Lab 4 |
| ![gdown](https://img.shields.io/badge/-gdown-4285F4?logo=google&logoColor=white) | any | Google Drive data download for Lab 4 calibration archive (Colab only) | Lab 4 |

</div>

**No pre-built Kalman filter libraries. No estimation toolboxes. No calibration frameworks beyond the core OpenCV functions in Lab 4.** Every estimator in this course — from the `(HᵀH)⁻¹Hᵀy` batch least squares solution, to the EKF Jacobian computation and gain-update-covariance cycle, to the UKF Cholesky sigma-point generation and cross-covariance reconstruction — is derived and coded explicitly from the mathematical definition, ensuring depth of understanding at every layer of the estimation and perception stack.

---

## 👤 Author

<div align="center">

### Umer Ahmed Baig Mughal

🎓 **MSc Robotics and Artificial Intelligence** — ITMO University <br>
🏛️ *Faculty of Control Systems and Robotics* <br>
🔬 *Specialization: Machine Learning · Computer Vision · Human-Robot Interaction · Autonomous Systems · Robotic Motion Control*

[![GitHub](https://img.shields.io/badge/GitHub-umerahmedbaig7-181717?style=for-the-badge&logo=github)](https://github.com/umerahmedbaig7)

</div>

---

## 📄 License

This repository is intended for **academic and research use**. All work was developed as part of the *Robot Perception* course within the MSc Robotics and Artificial Intelligence program at ITMO University. Redistribution, modification, and use in derivative academic work are permitted with appropriate attribution to the original author.

---

<div align="center">

*Robot Perception — MSc Robotics and Artificial Intelligence | ITMO University*

⭐ *If this repository helped you understand the mathematical foundations of robotic perception — from least squares estimation to Kalman filtering to stereo vision — consider giving it a star!* ⭐

</div>
