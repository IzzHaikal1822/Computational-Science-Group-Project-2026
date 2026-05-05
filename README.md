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

Citation: Smith, S. C., et al. (2011). AHA/ACCF Secondary Prevention and Risk Reduction Therapy for Patients With Coronary and Other Atherosclerotic Vascular Disease: 2011 Update. Circulation, 124(22), 2458-2473.
