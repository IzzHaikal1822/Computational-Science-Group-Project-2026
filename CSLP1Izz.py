# ==========================================
# 1. Importing Libraries 📚
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 2. Reading Data Set 👓
# ==========================================
# Referencing the specified dataset
df_heart = pd.read_csv(r"C:\Users\Izz Haikal\Documents\Computational Science\Year 2 Sem 2\Computational Science Laboratory\Group Project\heart.csv")

# Read the first 6 rows in the dataset
df_heart.head()

# Data type and checking null in dataset
print(df_heart.info())

# ==========================================
# 3. Initial Dataset Exploration 🔍
# ==========================================

# 3.1 Categorical Variables (Mapped to heart disease features)
df_heart.target.value_counts()
df_heart.sex.value_counts()
df_heart.cp.value_counts()
df_heart.fbs.value_counts()

# 3.2 Numerical Variables
df_heart.describe()

skewAge = df_heart.age.skew(axis = 0, skipna = True)
print('Age skewness: ', skewAge)

# Swapped Na_to_K for thalach (Maximum Heart Rate)
skewThalach = df_heart.thalach.skew(axis = 0, skipna = True)
print('Max Heart Rate (thalach) skewness: ', skewThalach)

# Using histplot as distplot is deprecated
sns.histplot(df_heart['age'], kde=True)
plt.show()

sns.histplot(df_heart['thalach'], kde=True)
plt.show()

# ==========================================
# 4. EDA 📊
# ==========================================

# 4.1 Target Distribution (Heart Disease presence)
sns.set_theme(style="darkgrid")
sns.countplot(y="target", data=df_heart, palette="flare")
plt.ylabel('Target (Heart Disease)')
plt.xlabel('Total')
plt.show()

# 4.2 Gender Distribution
sns.set_theme(style="darkgrid")
sns.countplot(x="sex", data=df_heart, palette="rocket")
plt.xlabel('Gender (0=Female, 1=Male)')
plt.ylabel('Total')
plt.show()

# 4.3 Chest Pain Type Distribution
sns.set_theme(style="darkgrid")
sns.countplot(y="cp", data=df_heart, palette="crest")
plt.ylabel('Chest Pain Type (cp)')
plt.xlabel('Total')
plt.show()

# 4.4 Fasting Blood Sugar Distribution
sns.set_theme(style="darkgrid")
sns.countplot(x="fbs", data=df_heart, palette="magma")
plt.xlabel('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)')
plt.ylabel('Total')
plt.show()

# 4.5 Gender Distribution based on Target
pd.crosstab(df_heart.sex, df_heart.target).plot(kind="bar", figsize=(12,5), color=['#003f5c','#ffa600'])
plt.title('Gender distribution based on Heart Disease')
plt.xlabel('Gender (0=Female, 1=Male)')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()

# 4.6 Chest Pain Distribution based on Target
pd.crosstab(df_heart.cp, df_heart.target).plot(kind="bar", figsize=(15,6), color=['#6929c4','#1192e8'])
plt.title('Chest Pain distribution based on Heart Disease')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()

# 4.7 Max Heart Rate Distribution based on Gender and Age
plt.scatter(x=df_heart.age[df_heart.sex==0], y=df_heart.thalach[(df_heart.sex==0)], c="Blue")
plt.scatter(x=df_heart.age[df_heart.sex==1], y=df_heart.thalach[(df_heart.sex==1)], c="Orange")
plt.legend(["Female", "Male"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate (thalach)")
plt.show()

# ==========================================
# 5. Dataset Preparation ⚙
# ==========================================

# 5.1 Data Binning
# 5.1.1 Age
bin_age = [0, 19, 29, 39, 49, 59, 69, 100]
category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
df_heart['age_binned'] = pd.cut(df_heart['age'], bins=bin_age, labels=category_age)
df_heart = df_heart.drop(['age'], axis = 1)

# 5.1.2 Max Heart Rate (thalach) instead of Na_to_K
bin_thalach = [0, 109, 139, 169, 250]
category_thalach = ['<110', '110-140', '140-170', '>170']
df_heart['thalach_binned'] = pd.cut(df_heart['thalach'], bins=bin_thalach, labels=category_thalach)
df_heart = df_heart.drop(['thalach'], axis = 1)

# 5.2 Feature Engineering (Do this BEFORE splitting)
# This ensures Train and Test have the exact same dummy columns

X = df_heart.drop(["target"], axis=1)
X = pd.get_dummies(X) 
y = df_heart["target"]

# 5.3 Splitting the dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

X_train.head()
X_test.head()

# 5.4 SMOTE Technique
from imblearn.over_sampling import SMOTE
X_train, y_train = SMOTE().fit_resample(X_train, y_train)

sns.set_theme(style="darkgrid")
sns.countplot(x=y_train, palette="mako_r")
plt.ylabel('Total')
plt.xlabel('Heart Disease Target')
plt.show()

# ==========================================
# 6. Models 🛠
# ==========================================

from sklearn.metrics import accuracy_score

# 6.1 Logistic Regression
from sklearn.linear_model import LogisticRegression
LRclassifier = LogisticRegression(solver='liblinear', max_iter=5000)
LRclassifier.fit(X_train, y_train)

y_pred = LRclassifier.predict(X_test)

print("--- Logistic Regression ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

LRAcc = accuracy_score(y_pred,y_test)
print('Logistic Regression accuracy is: {:.2f}%\n'.format(LRAcc*100))


# 6.2 K Neighbours
from sklearn.neighbors import KNeighborsClassifier
KNclassifier = KNeighborsClassifier(n_neighbors=20)
KNclassifier.fit(X_train, y_train)

y_pred = KNclassifier.predict(X_test)

print("--- K Nearest Neighbours ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

KNAcc = accuracy_score(y_pred,y_test)
print('K Neighbours accuracy is: {:.2f}%'.format(KNAcc*100))

scoreListknn = []
for i in range(1,30):
    KNclassifier = KNeighborsClassifier(n_neighbors = i)
    KNclassifier.fit(X_train, y_train)
    scoreListknn.append(KNclassifier.score(X_test, y_test))
    
plt.plot(range(1,30), scoreListknn)
plt.xticks(np.arange(1,30,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()
KNAccMax = max(scoreListknn)
print("KNN Acc Max {:.2f}%\n".format(KNAccMax*100))


# 6.3 Support Vector Machine (SVM)
from sklearn.svm import SVC
SVCclassifier = SVC(kernel='linear')
SVCclassifier.fit(X_train, y_train)

y_pred = SVCclassifier.predict(X_test)

print("--- Support Vector Machine ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

SVCAcc = accuracy_score(y_pred,y_test)
print('SVC accuracy is: {:.2f}%\n'.format(SVCAcc*100))

# 6.4.2 Gaussian NB
from sklearn.naive_bayes import GaussianNB
NBclassifier2 = GaussianNB()
NBclassifier2.fit(X_train, y_train)

y_pred = NBclassifier2.predict(X_test)

print("--- Gaussian Naive Bayes ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

NBAcc2 = accuracy_score(y_pred,y_test)
print('Gaussian Naive Bayes accuracy is: {:.2f}%\n'.format(NBAcc2*100))


# 6.5 Decision Tree
from sklearn.tree import DecisionTreeClassifier
DTclassifier = DecisionTreeClassifier(max_leaf_nodes=20)
DTclassifier.fit(X_train, y_train)

y_pred = DTclassifier.predict(X_test)

print("--- Decision Tree ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

DTAcc = accuracy_score(y_pred,y_test)
print('Decision Tree accuracy is: {:.2f}%'.format(DTAcc*100))

scoreListDT = []
for i in range(2,50):
    DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i)
    DTclassifier.fit(X_train, y_train)
    scoreListDT.append(DTclassifier.score(X_test, y_test))
    
plt.plot(range(2,50), scoreListDT)
plt.xticks(np.arange(2,50,5))
plt.xlabel("Leaf")
plt.ylabel("Score")
plt.show()
DTAccMax = max(scoreListDT)
print("DT Acc Max {:.2f}%\n".format(DTAccMax*100))


# 6.6 Random Forest
from sklearn.ensemble import RandomForestClassifier
RFclassifier = RandomForestClassifier(max_leaf_nodes=30)
RFclassifier.fit(X_train, y_train)

y_pred = RFclassifier.predict(X_test)

print("--- Random Forest ---")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

RFAcc = accuracy_score(y_pred,y_test)
print('Random Forest accuracy is: {:.2f}%'.format(RFAcc*100))

scoreListRF = []
for i in range(2,50):
    RFclassifier = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_leaf_nodes=i)
    RFclassifier.fit(X_train, y_train)
    scoreListRF.append(RFclassifier.score(X_test, y_test))
    
plt.plot(range(2,50), scoreListRF)
plt.xticks(np.arange(2,50,5))
plt.xlabel("RF Value")
plt.ylabel("Score")
plt.show()
RFAccMax = max(scoreListRF)
print("RF Acc Max {:.2f}%".format(RFAccMax*100))

# ==========================================
# 7. Treatment Recommendation System 💊
# ==========================================
import pandas as pd

# 1. Force pandas to show all columns without hiding any in the middle
pd.set_option('display.max_columns', None)

# 2. Force pandas to show the full text within each column cell
pd.set_option('display.max_colwidth', None)

# 3. (Optional) Widen the display area so it doesn't wrap to the next line as easily
pd.set_option('display.width', 1000)

def get_drug_recommendation(row, prediction):
    """
    A rule-based recommendation system mapping clinical features to general drug classes.
    """
    recommendations = []
    
    # Logic for patients predicted to HAVE heart disease (Target = 1)
    if prediction == 1:
        recommendations.append("Baseline: Aspirin/Antiplatelets")
        
        # High Blood Pressure threshold (e.g., > 140 mm Hg)
        if row['trestbps'] > 140:
            recommendations.append("Blood Pressure: ACE Inhibitors or Beta-blockers")
            
        # High Cholesterol threshold (e.g., > 240 mg/dl)
        if row['chol'] > 240:
            recommendations.append("Cholesterol: Statins")
            
        # Presence of Chest Pain (cp_1, cp_2, or cp_3 in dummified data)
        # Assuming you used pd.get_dummies, the chest pain columns might be named 'cp_1', 'cp_2', etc.
        # We check if any of the non-zero chest pain indicators are present.
        if row.get('cp_1', 0) == 1 or row.get('cp_2', 0) == 1 or row.get('cp_3', 0) == 1:
            recommendations.append("Angina: Nitroglycerin or Isosorbide")
            
    # Logic for patients predicted to NOT HAVE heart disease (Target = 0)
    else:
        # Borderline stats trigger preventive lifestyle warnings instead of heavy medication
        if row['trestbps'] > 130 or row['chol'] > 200:
            recommendations.append("Prevention: Diet, Exercise, and Lifestyle changes")
        else:
            recommendations.append("Routine annual checkup")
            
    return " | ".join(recommendations)

# 1. Ensure we are using the best model to generate predictions on the test set
# (Assuming RFclassifier is still in memory from Section 7.6)
final_predictions = RFclassifier.predict(X_test)

# 2. Create a new DataFrame to view our test patients alongside their recommendations
results_df = X_test.copy().reset_index(drop=True)
results_df['Predicted_Target'] = final_predictions
results_df['Actual_Target'] = y_test.values

# 3. Apply the recommendation function row by row
results_df['Recommended_Treatment'] = results_df.apply(
    lambda row: get_drug_recommendation(row, row['Predicted_Target']), axis=1
)

# 4. Display a clean summary of the first 10 patients
print("\n--- Patient Treatment Recommendations ---")
# Selecting specific columns for clarity
display_cols = ['trestbps', 'chol', 'Predicted_Target', 'Recommended_Treatment']
print(results_df[display_cols].head(10))

# ==========================================
# 8. Pharmacokinetics ODE Modeling 📉
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 8.1 Define the ODE model for drug concentration decay
def drug_decay_model(C, t, k):
    """
    Computes the derivative dC/dt = -k * C
    C: Current drug concentration/amount
    t: Current time
    k: Elimination rate constant
    """
    dCdt = -k * C
    return dCdt

# 8.2 Set up the parameters based on Aspirin
# Standard low-dose Aspirin is 81mg. Its half-life in the body is roughly 3 hours.
initial_dose = 81.0  
half_life_hours = 3.0  
k_elimination = np.log(2) / half_life_hours

# 8.3 Define the time array (Simulate over 24 hours)
# np.linspace(start, stop, num_points)
time_points = np.linspace(0, 24, 100)

# 8.4 Solve the ODE
# odeint takes (function, initial_condition, time_array, arguments_tuple)
concentration_over_time = odeint(drug_decay_model, initial_dose, time_points, args=(k_elimination,))

# 8.5 Visualize the ODE Results
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 5))
plt.plot(time_points, concentration_over_time, label=f'Aspirin Decay (Half-life: {half_life_hours} hrs)', color='red', linewidth=2.5)

# Add a dashed line representing a hypothetical minimum effective concentration threshold
plt.axhline(y=10, color='blue', linestyle='--', label='Minimum Effective Level')

plt.title('Pharmacokinetics: Drug Elimination Over 24 Hours (First-Order ODE)')
plt.xlabel('Time (hours)')
plt.ylabel('Drug Amount Remaining (mg)')
plt.legend()
plt.show()

# ==========================================
# 9. PK-PD ODE Modeling (Heart Rate & Beta-Blockers)
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 9.1 Define the Coupled ODE System
def beta_blocker_model(state, t, H_base, r, alpha, k_elim):
    """
    state[0] = C (Drug Concentration in mg)
    state[1] = H (Heart Rate in bpm)
    """
    C, H = state
    
    # ODE 1: Drug decay
    dCdt = -k_elim * C
    
    # ODE 2: Heart rate dynamics (Body tries to restore H_base, drug lowers it)
    dHdt = r * (H_base - H) - (alpha * C)
    
    return [dCdt, dHdt]

# 9.2 Set up Patient Parameters
# In a real pipeline, you would pull this directly from your X_test dataset:
# patient_thalach = X_test.iloc[0]['thalach'] 
patient_thalach = 170.0  # Baseline max heart rate (H_base)

# Drug Parameters (e.g., Metoprolol)
initial_dose = 50.0      # mg
half_life_hours = 4.0    # hours
k_elim = np.log(2) / half_life_hours

# Pharmacodynamic Parameters
r = 0.5       # Homeostatic recovery rate
alpha = 0.8   # Drug efficacy (bpm reduction per mg of drug)

# Initial state: Full dose in system, Heart rate currently at baseline
initial_state = [initial_dose, patient_thalach]

# 9.3 Simulate over 24 hours
time_points = np.linspace(0, 24, 200)
results = odeint(beta_blocker_model, initial_state, time_points, args=(patient_thalach, r, alpha, k_elim))

drug_concentration = results[:, 0]
heart_rate = results[:, 1]

# 9.4 Visualize the PK-PD Relationship
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Heart Rate on primary Y-axis
color = 'tab:red'
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Heart Rate (bpm)', color=color)
ax1.plot(time_points, heart_rate, color=color, linewidth=2.5, label='Patient Heart Rate')
ax1.axhline(y=patient_thalach, color='red', linestyle=':', alpha=0.5, label='Baseline Max HR')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# Create a secondary Y-axis for Drug Concentration
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Beta-Blocker Concentration (mg)', color=color)  
ax2.plot(time_points, drug_concentration, color=color, linewidth=2.5, linestyle='--', label='Drug Concentration')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title(f'PK-PD Simulation: Beta-Blocker Effect on Patient HR (Baseline: {patient_thalach} bpm)')
fig.tight_layout()  
plt.show()

# ==========================================
# 10. Advanced PK-PD: Non-Linear Emax Model (Blood Pressure) 
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 10.1 Define the Non-Linear Coupled ODE System
def emax_blood_pressure_model(state, t, P_base, r_bp, E_max, EC_50, k_elim):
    """
    state[0] = C (Drug Concentration in mg/L)
    state[1] = P (Blood Pressure in mmHg)
    """
    C, P = state
    
    # ODE 1: Pharmacokinetics (First-order drug decay)
    dCdt = -k_elim * C
    
    # ODE 2: Pharmacodynamics (Non-Linear Emax effect on Blood Pressure)
    # The drug effect cannot exceed E_max, no matter how high C gets.
    drug_effect = (E_max * C) / (EC_50 + C)
    dPdt = r_bp * (P_base - P) - drug_effect
    
    return [dCdt, dPdt]

# 10.2 Set up Patient Parameters
# In a real pipeline, pull this from your X_test dataset:
# patient_trestbps = X_test.iloc[0]['trestbps'] 
patient_trestbps = 160.0  # Baseline resting blood pressure (P_base), Stage 2 Hypertension

# Drug Parameters (e.g., standard ACE Inhibitor like Lisinopril)
initial_dose = 40.0       # mg
half_life_hours = 12.0    # ACE inhibitors generally have longer half-lives
k_elim = np.log(2) / half_life_hours

# Pharmacodynamic Parameters (Emax specific)
r_bp = 0.15      # Homeostatic recovery rate (body trying to return BP to 160)
E_max = 35.0     # The absolute maximum BP drop the drug can cause (mmHg)
EC_50 = 10.0     # The drug concentration that produces exactly 50% of E_max

# Initial state: Full dose in system, BP currently at baseline
initial_state = [initial_dose, patient_trestbps]

# 10.3 Simulate over 48 hours (longer timeframe due to 12hr half-life)
time_points = np.linspace(0, 48, 200)
results = odeint(emax_blood_pressure_model, initial_state, time_points, 
                 args=(patient_trestbps, r_bp, E_max, EC_50, k_elim))

drug_concentration = results[:, 0]
blood_pressure = results[:, 1]

# 10.4 Visualize the Non-Linear PK-PD Relationship
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Blood Pressure on primary Y-axis
color = 'tab:green'
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Systolic Blood Pressure (mmHg)', color=color)
ax1.plot(time_points, blood_pressure, color=color, linewidth=2.5, label='Patient Blood Pressure')
ax1.axhline(y=patient_trestbps, color='green', linestyle=':', alpha=0.5, label='Baseline BP')

# Mark the maximum possible drug effect (Emax) floor
max_effect_floor = patient_trestbps - E_max
ax1.axhline(y=max_effect_floor, color='gray', linestyle='-.', alpha=0.5, 
            label=f'Max Possible Effect Limit (-{E_max} mmHg)')

ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# Create a secondary Y-axis for Drug Concentration
ax2 = ax1.twinx()  
color = 'tab:purple'
ax2.set_ylabel('ACE Inhibitor Concentration (mg)', color=color)  
ax2.plot(time_points, drug_concentration, color=color, linewidth=2.5, linestyle='--', label='Drug Concentration')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title(f'Non-Linear Emax Simulation: ACE Inhibitor Effect on Blood Pressure')
fig.tight_layout()  
plt.show()
