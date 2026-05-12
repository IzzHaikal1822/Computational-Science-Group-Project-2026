This is a highly comprehensive and well-structured Python script! It successfully bridges two major pillars of Computational Science: **Data-Driven Modeling** (Machine Learning) and **Mechanistic Modeling** (Ordinary Differential Equations).

To give you the best feedback, I need to be completely transparent: **You didn't attach the specific assignment instructions or grading rubric for your group project.** However, based on the course name ("Computational Science Laboratory") and the content of your code, I can elaborate on exactly what this script does and evaluate how well it aligns with standard expectations for a project of this caliber.

Here is a detailed breakdown of your code, followed by an assessment of its likely fit for your project.

---

### **Part 1: Data Science & Machine Learning Pipeline (Sections 1-6)**

**1 & 2. Data Import & Reading**

* **What it does:** You import the standard data science stack (`pandas`, `numpy`, `matplotlib`, `seaborn`) and load a dataset named `heart.csv`. You also run basic checks (`head()`, `info()`) to understand the dataset's structure and look for missing values.

**3 & 4. Exploratory Data Analysis (EDA)**

* **What it does:** You investigate the distribution of both categorical (gender, chest pain type, fasting blood sugar) and numerical variables (age, maximum heart rate). You calculate skewness and visualize how these features relate to the presence of heart disease (the `target` variable) using bar charts and scatter plots.
* **Strengths:** Good use of `pd.crosstab` for comparing features against the target. Visualizing data before modeling is a critical best practice.

**5. Dataset Preparation**

* **What it does:** This is where you clean and prep the data for the machine learning algorithms.
* **Binning:** You convert continuous variables (`age`, `thalach`) into categorical bins (e.g., 20s, 30s).
* **Encoding:** You use `pd.get_dummies()` to convert categorical text data into a numerical format that models can understand.
* **Splitting & SMOTE:** You split the data into 70% training and 30% testing sets. Crucially, you apply **SMOTE** (Synthetic Minority Over-sampling Technique) to the training data. This is an advanced technique used to balance uneven datasets so the model doesn't become biased toward the majority class.



**6. Machine Learning Models**

* **What it does:** You train and evaluate six different classification algorithms: Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Gaussian Naive Bayes, Decision Tree, and Random Forest.
* **Strengths:** For KNN, Decision Tree, and Random Forest, you didn't just run them once; you wrote `for` loops to test different hyperparameters (like the 'K' value or the number of leaf nodes) and plotted the accuracy scores to find the absolute best configuration. You also consistently print the classification report and confusion matrix, giving a complete picture of precision and recall.

---

### **Part 2: Applied Logic & Mechanistic Modeling (Sections 7-10)**

**7. Treatment Recommendation System**

* **What it does:** You created a rule-based engine (`get_drug_recommendation`). It takes the predictive output from your best ML model (Random Forest) and combines it with the patient's raw clinical features (Blood Pressure, Cholesterol, Chest Pain) to recommend specific classes of drugs or lifestyle changes.

**8. Pharmacokinetics ODE Modeling (Simple Decay)**

* **What it does:** You shift from statistics to calculus by defining an Ordinary Differential Equation (ODE) using `scipy.integrate.odeint`. This simulates how Aspirin leaves the body over time using a first-order decay formula:

$$\frac{dC}{dt}=-k \cdot C$$


* **Why it matters:** It visualizes how quickly drug concentration drops below a "minimum effective level," informing dosing schedules.

**9. PK-PD ODE Modeling (Heart Rate & Beta-Blockers)**

* **What it does:** You level up to a system of *coupled* ODEs (Pharmacokinetics-Pharmacodynamics). One equation tracks the drug concentration decaying, while the second tracks the patient's heart rate trying to return to a baseline ($H_{base}$), offset by the drug's effect ($\alpha \cdot C$).

$$\frac{dH}{dt}=r \cdot (H_{base}-H)-\alpha \cdot C$$



**10. Advanced PK-PD: Non-Linear Emax Model (Blood Pressure)**

* **What it does:** This is the most mathematically complex part of the script. It models how an ACE Inhibitor lowers blood pressure. It uses a non-linear $E_{max}$ model, acknowledging that a drug's effect has a ceiling. No matter how high the drug concentration ($C$) gets, the blood pressure won't drop below a certain threshold:

$$\frac{dP}{dt}=r_{bp} \cdot (P_{base}-P)-\frac{E_{max} \cdot C}{EC_{50}+C}$$



---

### **Does it follow standard group project instructions?**

Without seeing your exact rubric, I can tell you that this script strongly aligns with high-level university expectations for a Computational Science project for the following reasons:

1. **Multidisciplinary Approach:** Standard computer science projects stop at Section 6. By adding the ODEs (Sections 8-10), you proved you can handle continuous mathematics/simulation, which is the defining characteristic of *Computational Science*.
2. **End-to-End Pipeline:** You take raw data, process it, train models to make predictions, and then *use* those predictions in a simulated real-world application (the recommendation system and ODE drug simulations).
3. **Data Leakage Prevention:** You correctly applied `pd.get_dummies` *before* the train-test split to ensure matching columns, but you also correctly applied SMOTE *after* the split (only to `X_train`) to prevent data leakage. This is a common pitfall that you successfully avoided.

**Minor constructive feedback for your submission:**

* **Hardcoded Paths:** In Section 2, your file path is `C:\Users\Izz Haikal\Documents\...`. If your lecturer runs this code, it will crash immediately. Change this to a relative path like `pd.read_csv("heart.csv")` and submit the CSV file in the same folder as your script.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Understanding the ODE

To understand an **Ordinary Differential Equation (ODE)**, you don't need a math degree. You just need to think about how things **change** over time.

Imagine you are driving a car.

* Your **odometer** tells you *where* you are (Total distance: 50 miles).
* Your **speedometer** tells you *how fast that distance is changing* (Speed: 60 miles per hour).

Regular algebra equations are like the odometer—they tell you how much of something you have. **Differential equations are like the speedometer—they tell you the exact rate at which something is changing at any given split second.**

In your Python code, you are using ODEs to simulate **Pharmacokinetics (PK)** (how the body breaks down a drug) and **Pharmacodynamics (PD)** (how the drug affects the body).

Let's break down the three models in your code step-by-step as if we are observing water in a bucket.


### 1. The Aspirin Model: The "Leaky Bucket" (Section 8)

In Section 8, you have a simple model for a drug decaying in the body:

$$\frac{dC}{dt} = -k \cdot C$$

* **$C$** is the amount of drug in your body (let's say, water in a bucket).
* **$t$** is time.
* **$dC/dt$** is the math way of writing "the speed at which the drug is disappearing" (the speed water leaks from the bucket).
* **$k$** is the elimination rate (how big the hole in the bucket is).

**What this ODE says:** The speed at which the drug disappears (`dC/dt`) is determined by how much drug is currently in your system (`C`).
If you take an 81mg Aspirin, the "pressure" of having all that drug in your blood makes your kidneys filter it out quickly. As the drug amount gets smaller, the speed of elimination slows down. That's why it takes a specific "half-life" to slowly clear the rest out.

### 2. The Beta-Blocker Model: The "Tug-of-War" (Section 9)

In Section 9, things get more interesting. You have a patient with a rapidly beating heart (170 bpm) taking a Beta-Blocker. You now have **two** ODEs playing tug-of-war.

**Equation 1: The Drug Decaying**


$$\frac{dC}{dt} = -k_{elim} \cdot C$$


*(This is the exact same leaky bucket as above. The beta-blocker is slowly leaving the body).*

**Equation 2: The Heart Rate Responding**


$$\frac{dH}{dt} = r(H_{base} - H) - (\alpha \cdot C)$$

This looks scary, but it's just describing a tug-of-war on the patient's heart rate ($H$):

* **The Body Pulling Up ($r(H_{base} - H)$):** Your body has a baseline heart rate it *wants* to be at (170 bpm in this sick patient). If the heart rate drops, the body panics and tries to pull it back up. The rate of recovery is $r$.
* **The Drug Pulling Down ($\alpha \cdot C$):** The drug ($C$) is forcefully pushing the heart rate down. The strength of the drug is $\alpha$.

**What this ODE says:** At any given second, the speed your heart rate changes (`dH/dt`) depends on who is winning. Right after taking the pill, the drug amount ($C$) is high, so it wins the tug-of-war and pulls the heart rate down. Hours later, as the drug leaks out of the body (Equation 1), the body starts winning again and pulls the heart rate back up to 170.

### 3. The Blood Pressure Model: The "Dimmer Switch" (Section 10)

In Section 10, you introduced the "Emax" model.

In the previous model, if you gave the patient 1000mg of a drug, the math would assume their heart rate would drop below zero and they would die. But human bodies don't work like that! We have a limited number of cellular receptors. Once all the receptors are blocked by the drug, giving *more* drug doesn't do anything extra.

To fix this, you changed the "Drug Pulling Down" part to this:


$$\text{Drug Effect} = \frac{E_{max} \cdot C}{EC_{50} + C}$$

**What this means:** * **$E_{max}$** is the absolute maximum drop in blood pressure the drug can achieve (e.g., -35 mmHg). It acts like the bottom limit on a dimmer switch.

* No matter how incredibly high the drug concentration ($C$) gets in the blood, the equation mathematically prevents the drug effect from ever pulling the blood pressure down by more than 35 points.

### Summary of what the solver (`odeint`) does:

Because these values are constantly changing every millisecond (the drug drops, which changes the heart rate, which changes the pull of the body), you can't just use standard algebra. The Python tool `odeint` acts as a time-traveler. It looks at the starting values, calculates the "speedometer" for 1 second, steps forward, recalculates the new speed for the next second, and does this hundreds of times to draw the final curves you see on your graphs!

To help you visualize how these variables interact dynamically, you can explore the simulation below. Try changing the dose or the drug's half-life to see how the mathematical "tug-of-war" plays out over time.
