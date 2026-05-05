# Computational-Science-Group-Project-2026
Group Project based on machine learning application in computational biology using differential equation and Python programming language.

The link used for the Dataset and the Codes

https://www.kaggle.com/code/caesarmario/drug-classification-w-various-ml-models/notebook

https://github.com/JosephineTella/Drug-Classification/tree/main

https://github.com/AnjanaAbY/Drug-Classification-Model/tree/main

https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

Reference for Heart Disease Drug Classification

1. Blood Pressure (trestbps)
The Code's Logic: > 140 triggers medication (ACE Inhibitors/Beta-blockers); > 130 triggers lifestyle prevention.

The Medical Reference: 2017 AHA/ACC Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults.

The Science:

The AHA/ACC classifies systolic blood pressure of 130–139 mm Hg as "Stage 1 Hypertension." For many patients in this tier without a high baseline risk, the first line of defense is non-pharmacological interventions (Diet, Exercise, Lifestyle).

Systolic pressure of ≥ 140 mm Hg is classified as "Stage 2 Hypertension," where guidelines strongly recommend blood pressure-lowering medications, prominently including ACE inhibitors, ARBs, Calcium channel blockers, or Beta-blockers.

Citation: Whelton, P. K., et al. (2018). 2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults. Journal of the American College of Cardiology, 71(19), e127-e248.

2. Total Cholesterol (chol)
The Code's Logic: > 240 triggers Statins; > 200 triggers lifestyle prevention.

The Medical Reference: National Cholesterol Education Program (NCEP) Adult Treatment Panel III (ATP III) and updated AHA/ACC Blood Cholesterol Guidelines.

The Science:

Total cholesterol below 200 mg/dL is considered "Desirable."

Total cholesterol between 200–239 mg/dL is considered "Borderline High" (triggering the lifestyle and diet warnings in the code).

Total cholesterol ≥ 240 mg/dL is classified as "High." At this level, particularly if the patient is diagnosed with heart disease (Target = 1), Statin therapy is standard clinical practice to lower LDL (bad cholesterol) and prevent further arterial plaque buildup.

Citation: Grundy, S. M., et al. (2019). 2018 AHA/ACC/AACVPR/AAPA/ABC/ACPM/ADA/AGS/APhA/ASPC/NLA/PCNA Guideline on the Management of Blood Cholesterol. Circulation, 139(25), e1082-e1143.

3. Chest Pain / Angina (cp_1, cp_2, cp_3)
The Code's Logic: Presence of typical/atypical angina triggers Nitroglycerin or Isosorbide.

The Medical Reference: AHA/ACC Guideline for the Diagnosis and Management of Patients With Stable Ischemic Heart Disease.

The Science: Nitrates (like Nitroglycerin) are vasodilators. They relax the blood vessels, increasing blood supply and oxygen to the heart, which immediately relieves the symptoms of angina (chest pain). It is the universal first-line pharmacological treatment for acute angina episodes.

Citation: Fihn, S. D., et al. (2012). 2012 ACCF/AHA/ACP/AATS/PCNA/SCAI/STS Guideline for the Diagnosis and Management of Patients With Stable Ischemic Heart Disease. Journal of the American College of Cardiology, 60(24), e44-e164.

4. Baseline Heart Disease Prediction (Target = 1)
The Code's Logic: If predicted positive for heart disease, immediately recommend Aspirin/Antiplatelets.

The Medical Reference: AHA/ACC Guidelines for Secondary Prevention for Patients With Coronary and Other Atherosclerotic Vascular Disease.

The Science: If a patient has known atherosclerotic cardiovascular disease (which is what a "Target = 1" prediction implies), "Secondary Prevention" protocols kick in. Low-dose aspirin (75-162 mg daily) is recommended indefinitely for these patients to prevent blood clots that cause heart attacks and strokes.

**CRITICAL DISCLAIMER:** *The code provided is a simplified mathematical approximation designed for educational and conceptual data science purposes. While it is rooted in real pharmacological principles, it is a basic linear model and must never be used to make actual clinical decisions or predict real-world patient outcomes.*

The mathematical logic in the Python code you provided is based on foundational principles of **Pharmacokinetics (PK)** and **Pharmacodynamics (PD)**. Specifically, it uses a **One-Compartment First-Order Elimination Model** coupled with a simplified **Physiological Indirect Response Model**.

Here are the specific textbooks, articles, and scientific concepts that serve as the foundation for the equations used in your code.

### 1. First-Order Drug Decay (The PK Equation)
**The Code:** `dCdt = -k_elim * C` and `k_elim = np.log(2) / half_life_hours`
**The Concept:** This represents a "One-Compartment Open Model" with first-order elimination. It assumes the body acts as a single uniform container, and the rate at which the drug is cleared is directly proportional to the concentration of the drug currently in the system.

**References:**
*   **Book:** Rowland, M., & Tozer, T. N. (2010). *Clinical Pharmacokinetics and Pharmacodynamics: Concepts and Applications* (4th Ed.). Lippincott Williams & Wilkins. *(This is the gold-standard textbook for standard PK/PD modeling, specifically covering first-order kinetics and half-life calculations in early chapters).*
*   **Book:** Shargel, L., Wu-Pong, S., & Yu, A. B. (2012). *Applied Biopharmaceutics & Pharmacokinetics* (6th Ed.). McGraw-Hill Education.

### 2. Heart Rate Dynamics (The PD Equation)
**The Code:** `dHdt = r * (H_base - H) - (alpha * C)`
**The Concept:** This is a simplified, linear **Turnover Model** (or Indirect Response Model). 
*   `r * (H_base - H)` represents physiological homeostasis: the body naturally tries to return the heart rate to its baseline. 
*   `- (alpha * C)` represents the drug's inhibitory effect on that system. 
*   *Note on accuracy:* In advanced clinical pharmacology, the drug effect is rarely perfectly linear (`alpha * C`). Instead, modelers use an $E_{max}$ model (a hyperbolic curve) because a drug's effect eventually maxes out even if you add more drug. The linear version used in your code is a common first-step approximation for low drug concentrations.

**References:**
*   **Seminal Article:** Dayneka, N. L., Garg, V., & Jusko, W. J. (1993). Comparison of four basic models of indirect pharmacodynamic responses. *Journal of Pharmacokinetics and Biopharmaceutics*, 21(4), 457-478. *(This is the famous paper that introduced how to mathematically model a physiological baseline that is pushed out of balance by a drug).*
*   **Article:** Sharma, A., & Jusko, W. J. (1998). Characteristics of indirect pharmacodynamic models and applications to clinical drug responses. *British Journal of Clinical Pharmacology*, 45(3), 229-239.

### 3. Metoprolol Parameters (The Constants)
**The Code:** `half_life_hours = 4.0` and its effect as a Beta-blocker.
**The Concept:** Metoprolol is a cardioselective $\beta_1$-adrenergic receptor blocker. Its primary clinical effect is negative chronotropic (lowering the heart rate). Standard immediate-release Metoprolol has an elimination half-life of roughly 3 to 4 hours in healthy adults.

**References:**
*   **Article:** Regårdh, C. G., & Johnsson, G. (1980). Clinical pharmacokinetics of metoprolol. *Clinical Pharmacokinetics*, 5(6), 557-569. *(Details the specific 3-4 hour half-life and elimination rates).*
*   **Website:** [FDA Prescribing Information for Lopressor (metoprolol tartrate)](https://www.accessdata.fda.gov/drugsatfda_docs/label/2008/017963s062,018704s021lbl.pdf). *(Section: Clinical Pharmacology details the specific half-life and heart-rate reduction effects).*

### 4. Mathematical Implementation
**The Code:** `from scipy.integrate import odeint`
**The Concept:** The code utilizes Numerical Integration to solve the Ordinary Differential Equations. Since calculating the exact curve by hand over 24 hours is highly complex, algorithms like LSODA (which `odeint` uses under the hood) estimate the curve step-by-step.

**References:**
*   **Website:** [SciPy Documentation for `scipy.integrate.odeint`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html).
*   **Book:** Quarteroni, A., Sacco, R., & Saleri, F. (2007). *Numerical Mathematics* (2nd Ed.). Springer. *(Explains the numerical methods used to solve ODEs in computational science).*

Citation: Smith, S. C., et al. (2011). AHA/ACCF Secondary Prevention and Risk Reduction Therapy for Patients With Coronary and Other Atherosclerotic Vascular Disease: 2011 Update. Circulation, 124(22), 2458-2473.
