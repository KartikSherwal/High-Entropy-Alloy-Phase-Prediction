# High-Entropy Alloy (HEA) Phase Prediction Using Classical Machine Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Library-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

An exploratory machine learning project to model and predict the crystal phases of **High-Entropy Alloys (HEAs)** from elemental compositions, physical/thermodynamic parameters, and heat treatment conditions.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset & Target Classes](#-dataset--target-classes)
3. [Feature Engineering & Preprocessing](#-feature-engineering--preprocessing)
4. [Machine Learning Methodology](#-machine-learning-methodology)
5. [Model Benchmark & Results](#-model-benchmark--results)
6. [Metallurgical Insights & Interpretability](#-metallurgical-insights--interpretability)
7. [Running the Prediction Pipeline](#-running-the-prediction-pipeline)
8. [References](#-references)

---

## 🔬 Project Overview

High-Entropy Alloys (HEAs) are multi-component alloy systems containing five or more principal elements in near-equiatomic proportions. Unlike conventional single-principal-element alloys (e.g., steel or aluminum alloys), HEAs derive their stability from high configurational entropy ($\Delta S_{\text{mix}}$).

A central metallurgical challenge in HEA design is **phase selection**: predicting whether an alloy composition will solidify into:
* A ductile **Face-Centered Cubic (FCC)** solid solution,
* A high-strength **Body-Centered Cubic (BCC)** solid solution,
* A **Dual FCC + BCC** duplex mixture, or
* A brittle **Intermetallic (IM)** compound / multiphase structure.

This project implements an end-to-end classical machine learning framework trained on experimental alloy data to accurately classify the resulting phase, eliminating time-consuming trial-and-error synthesis in the laboratory.

---

## 📊 Dataset & Target Classes

* **Source File**: `Data Set HEAs_1400.csv`
* **Total Valid Alloys**: $1,360$ samples (after removing empty formatting rows)
* **Target Variable**: `Phases` (4 mutually exclusive classes)

| Target Phase | Description | Sample Count | Class Share |
| :--- | :--- | :---: | :---: |
| **`Im`** | Intermetallic Compound / Multiphase with IM | 463 | 34.04% |
| **`BCC_SS`** | Single-Phase Body-Centered Cubic Solid Solution | 441 | 32.43% |
| **`FCC_SS`** | Single-Phase Face-Centered Cubic Solid Solution | 354 | 26.03% |
| **`FCC_PLUS_BCC`** | Dual-Phase Solid Solution (FCC + BCC Duplex) | 102 | 7.50% |

> **Note on Imbalance**: The duplex phase (`FCC_PLUS_BCC`) is a minority transition class ($7.5\%$). To avoid sampling bias, all model training and cross-validation employ **stratification**.

---

## ⚙️ Feature Engineering & Preprocessing

The model uses **38 clean predictive features** spanning chemistry, thermodynamics, and thermal processing history:

### 1. Features Retained for Training ($X \in \mathbb{R}^{1360 \times 38}$)
* **25 Elemental Mole Fractions**: `Al`, `Co`, `Cr`, `Fe`, `Ni`, `Cu`, `Mn`, `Ti`, `V`, `Nb`, `Mo`, `Zr`, `Hf`, `Ta`, `W`, `C`, `Mg`, `Zn`, `Si`, `Re`, `N`, `Sc`, `Li`, `Sn`, `Be`.
* **10 Thermodynamic & Physical Properties**:
  * $VEC$: Valence Electron Concentration (mean valence electrons per atom)
  * $\Delta H_{\text{mix}}$ (`dHmix`): Enthalpy of mixing ($\text{kJ/mol}$)
  * $\Delta S_{\text{mix}}$ (`dSmix`): Entropy of mixing ($\text{J/mol}\cdot\text{K}$)
  * $\Delta G_{\text{mix}}$ (`dGmix`): Gibbs free energy of mixing
  * $\delta$ (`Atom.Size.Diff`): Atomic size difference parameter ($\%$)
  * $\Delta \chi$ (`Elect.Diff`): Electronegativity difference parameter
  * $T_m$ (`Tm`): Calculated melting temperature ($\text{K}$)
  * $\rho_{\text{calc}}$ (`Density_calc`): Theoretical density ($\text{g/cm}^3$)
  * $\Omega$ (`n.Para`): Non-dimensional thermodynamic stability parameter ($\Omega = T_m \Delta S_{\text{mix}} / |\Delta H_{\text{mix}}|$)
  * `Num_of_Elem`: Number of constituent elements
* **3 Processing Parameters**:
  * `Route_PM`: One-hot encoded synthesis route (Powder Metallurgy vs. Arc Melting `AC`)
  * `Homogenization_Temp`: Homogenization temperature in $^\circ\text{C}$ ($0$ if unhomogenized/as-cast)
  * `Annealing_Temp`: Annealing heat treatment temperature in $^\circ\text{C}$ ($0$ if unannealed/as-cast)

### 2. Columns Dropped & Technical Justifications
* **Identifiers & Text Metadata** (`Alloy ID`, `Alloy `, `References`): Arbitrary administrative strings that provide no physical signal and would cause memorization/overfitting.
* **Target Leakage Columns** (`Microstructure_`, `Microstructure`, `Multiphase`, `IM_Structure`): These record experimental post-mortem microstructural evaluations. Using post-synthesis phase data to predict phase is circular and invalid.
* **Artifact Columns** (`Unnamed: 51`, `Unnamed: 52`): $100\%$ empty columns generated during spreadsheet export.
* **Heavily Sparse Processing Notes** (`HPR` [96% null], `Quenching` [80% null], `Hot-Cold_Working` [67% null], `Homogenization_Time` [59% null], `Annealing_Time_(min)` [56% null]): Imputing $60\text{--}96\%$ missing text entries creates severe noise; critical thermal state is already captured by temperature.

---

## 🧠 Machine Learning Methodology

1. **Stratified Train-Test Split**:
   * $80\%$ Training ($1,088$ alloys) and $20\%$ Hold-out Testing ($272$ alloys).
   * `stratify=y` guarantees identical 4-phase class proportions across both partitions.
2. **Data Leakage Isolation**:
   * `StandardScaler` was fitted strictly on `X_train` and merely transformed `X_test`.
   * Distance-based models (Logistic Regression, k-NN) received scaled features, while decision tree ensembles utilized unscaled physical features.
3. **5-Fold Stratified Cross-Validation**:
   * Evaluated all models over 5 independent validation folds to ensure that high performance is not an artifact of an isolated test split.
4. **Model Selection Rationale**:
   * Classical ML (tree ensembles) was chosen over deep learning because tabular datasets with $N \approx 1,400$ samples and heterogeneous physical features are best modeled by partition-based ensembles, avoiding neural network overfitting.

---

## 📈 Model Benchmark & Results

Seven classical machine learning algorithms were benchmarked under identical conditions:

| Algorithm | 5-Fold CV Accuracy (%) | Test Accuracy (%) | Test Precision (%) | Test Recall (%) | Weighted F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Extra Trees** | **86.86 ± 1.48** | **86.03** | **86.17** | **86.03** | **86.07** |
| 🥈 **XGBoost** | 86.12 ± 1.04 | 84.93 | 84.79 | 84.93 | 84.82 |
| 🥉 **Random Forest** | 86.40 ± 0.79 | 84.56 | 84.78 | 84.56 | 84.50 |
| **Decision Tree** | 84.93 ± 1.12 | 82.35 | 82.64 | 82.35 | 82.42 |
| **Gradient Boosting** | 85.76 ± 2.62 | 81.99 | 81.84 | 81.99 | 81.83 |
| **k-Nearest Neighbors** | 81.44 ± 2.05 | 78.68 | 78.32 | 78.68 | 78.46 |
| **Logistic Regression** | 73.90 ± 2.91 | 73.16 | 73.79 | 73.16 | 73.00 |

### Detailed Performance of the Best Model (Extra Trees)

* **Overall Test Accuracy**: **86.03%**
* **Macro F1-Score**: **83.82%**
* **Weighted F1-Score**: **86.07%**

#### Per-Class Classification Report:
```text
               precision    recall  f1-score   support

      BCC_SS      0.8889    0.9091    0.8989        88
FCC_PLUS_BCC      0.7143    0.7500    0.7317        20
      FCC_SS      0.9254    0.8732    0.8986        71
          Im      0.8191    0.8280    0.8235        93

    accuracy                          0.8603       272
   macro avg      0.8369    0.8401    0.8382       272
weighted avg      0.8617    0.8603    0.8607       272
```

---

## 🔍 Metallurgical Insights & Interpretability

Gini feature importance analysis confirms that the model's decision boundaries mirror established physical metallurgy laws:

```text
Top 10 Most Influential Features (Extra Trees):
  1. VEC                  : 8.76%  (Valence Electron Concentration)
  2. Annealing_Temp       : 7.23%  (Heat treatment kinetics)
  3. Cr                   : 5.44%  (Transition element fraction)
  4. Atom.Size.Diff (δ)   : 5.22%  (Hume-Rothery lattice distortion)
  5. Density_calc         : 5.08%  (Refractory vs light element weight)
  6. dHmix (ΔH_mix)       : 5.05%  (Enthalpy of mixing driving force)
  7. Al                   : 5.00%  (BCC stabilizer / B2 compound former)
  8. dGmix (ΔG_mix)       : 4.56%  (Gibbs free energy of formation)
  9. Homogenization_Temp  : 3.99%  (Dissolution of dendritic segregation)
 10. Co                   : 3.90%  (FCC stabilizer)
```

1. **Guo's Rule Validation ($VEC$)**:
   $VEC$ emerged as the most critical feature ($8.76\%$). The model discovered that alloys with $VEC < 6.87$ form BCC, while those with $VEC \ge 8.0$ form FCC.
2. **Hume-Rothery Criterion ($\delta$)**:
   Atomic size difference $\delta$ ($5.22\%$) acts as a threshold: when $\delta > 6.6\%$, severe lattice strain destabilizes solid solutions into intermetallics.
3. **Role of Thermal Processing (`Annealing_Temp`)**:
   Ranking second ($7.23\%$), annealing temperature dictates whether metastable solid solutions frozen during fast casting decompose into equilibrium intermetallic precipitates.
4. **Role of Aluminum (`Al`)**:
   Due to its large radius and exothermic mixing with transition metals, Aluminum strongly triggers phase transitions from FCC to BCC or B2 intermetallics.

---





## 💻 Running the Prediction Pipeline

The repository includes [**`predict.py`**](predict.py), a standalone CLI tool that loads `hea_phase_model.joblib` and evaluates custom alloy compositions.

Run the tool in your terminal:
```bash
python predict.py
```

### Main Menu Options:
1. **Interactive Manual Input**:
   * Enter only active element mole fractions (e.g. `Co:0.25, Cr:0.25, Fe:0.25, Ni:0.25`).
   * Press `[Enter]` on any thermodynamic descriptor to accept dataset median defaults.
2. **Quick Benchmark Presets**:
   * Test canonical literature benchmarks instantly:
     * **Preset 1 (Cantor Alloy `CoCrFeMnNi`)**: Predicts **`FCC_SS`**
     * **Preset 2 (Refractory HEA `TiZrNbTaMo`)**: Predicts **`BCC_SS`**
     * **Preset 3 (Al-Rich HEA `Al1.5CoCrFeNi`)**: Predicts **`Im`**
     * **Preset 4 (Duplex HEA `Al0.5CoCrFeNi`)**: Predicts **`FCC_PLUS_BCC`**
3. **Batch Prediction from CSV**:
   * Supply the path to an unlabelled CSV file; the tool automatically generates `predictions_output.csv` with class probabilities.

---

## 📚 References

1. **Guo, S., Ng, C., Lu, J., & Liu, C. T. (2011).** Effect of valence electron concentration on stability of fcc or bcc phase in high entropy alloys. *Journal of Applied Physics*, 109(10), 103505.
2. **Zhang, Y., Zuo, T. T., Tang, Z., Gao, M. C., Dahmen, K. A., Liaw, P. K., & Lu, Z. P. (2014).** Microstructures and properties of high-entropy alloys. *Progress in Materials Science*, 61, 1-93.
3. **Miracle, D. B., & Senkov, O. N. (2017).** A critical review of high entropy alloys and related concepts. *Acta Materialia*, 122, 448-511.
4. **Yang, X., & Zhang, Y. (2012).** Prediction of high-entropy alloy phases based on thermodynamic and topology parameters. *Materials Chemistry and Physics*, 132(2-3), 233-238.
