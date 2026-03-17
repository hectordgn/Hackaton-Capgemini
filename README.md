# HR Analytics and Turnover Prediction Project

## 1. Objectives
This project provides a decision-support tool for Human Resources management, focusing on anticipating employee resignations. The technical and strategic objectives are:
* Prediction: Proactively identify employees with a high risk of leaving using a Machine Learning model.
* Cybersecurity & Compliance: Ensure personal data security and GDPR compliance through pseudonymization and cryptographic hashing of identifiers before any analysis.
* Frugality (Green IT): Favor a lightweight architecture and a resource-efficient Machine Learning model to minimize the carbon footprint compared to massive AI models.

## 2. Scope
In Scope:
* Processing, cleaning, and anonymization of the provided HR dataset.
* Training a predictive model (binary classification) based on socio-professional, performance, and satisfaction variables.
* Creation of an interactive dashboard to visualize risks and explain the key factors influencing the algorithm's decisions (Explainability).
* Code security management (environment variables, input validation).


## 3. Persona
This solution was designed for the following user profile:
* Target Role: Human Resources Director / HR Manager.
* Context: Mid-sized company facing a problematic turnover rate, particularly regarding strategic profiles.
* Needs: A visual tool, accessible without Data Science skills, allowing a shift from a reactive posture (noting departures) to a proactive posture (targeted retention actions).
* Constraints: Limited IT infrastructure budget (justifying the frugal approach) and manipulation of highly confidential data (justifying the cybersecurity approach).

## 4. Architecture

Our solution follows a clear and modular pipeline:

            ┌───────────────┐
            │   Raw Data    │
            └──────┬────────┘
                   ↓
        ┌──────────────────────────┐
        │ Data Cleaning & Processing│
        │ (No Data Leakage)        │
        └──────┬───────────────────┘
               ↓
        ┌──────────────────────────┐
        │ Feature Engineering      │
        │ (No Sensitive Data)      │
        └──────┬───────────────────┘
               ↓
        ┌──────────────────────────┐
        │ ML Pipeline              │
        │ (Frugal AI - Scikit)     │
        └──────┬───────────────────┘
               ↓
        ┌──────────────────────────┐
        │ Risk Prediction (%)      │
        └──────┬───────────────────┘
               ↓
        ┌──────────────────────────┐
        │ Explainability           │
        │ (Top 5 Drivers)          │
        └──────┬───────────────────┘
               ↓
        ┌────────────────────────────────────┐
        │ Dashboard + Excel Output           │
        │ -> HR Decisions & Risk Mitigation  │
        └──────┬─────────────────────────────┘


## 5. Project Structure

The repository is organized exactly as follows:

├── .gitignore                     # Prevents sensitive data and virtual environments from leaking
├── app.py                         # (3) Streamlit App: The main interactive HR dashboard (UI)
├── Audit_Attrition_Complet.xlsx   # Output: Specific factors for explainability (Excel format)
├── common.py                      # Shared helper functions and configurations
├── Hash_delete.py                 # (1) Security Script: Anonymizes data and hashes IDs
├── HR_Analytics.csv               # Safe, anonymized dataset used for training the model
├── hr_clean_notebook_new.ipynb    # Jupyter Notebook: Initial data exploration and prototyping
├── HR_Keys.csv                    # SECRET mapping table (Must never be shared or committed)
├── HRDataset_v14.csv              # Raw initial dataset (Sensitive data)
├── MODEL CARD.docx                # Documentation: Risks, compliance, and model limitations
├── README.md                      # Project documentation and instructions
└── train_model.py                 # (2) ML Pipeline: Trains the frugal model and generates predictions

## 6. Démo

Raw Results

<img width="2385" height="1101" alt="image" src="https://github.com/user-attachments/assets/ab46f50b-99ea-42df-8146-6e5570a52496" />

 ## 7.DashBoard

<img width="1888" height="871" alt="Screenshot 2026-03-17 120141" src="https://github.com/user-attachments/assets/fb6fac95-ce5e-417f-99d3-42e34841fd03" />
<img width="1903" height="913" alt="Screenshot 2026-03-17 120129" src="https://github.com/user-attachments/assets/96c4f66f-8049-4fc1-bb97-1ffba327e0b5" />
<img width="1904" height="1012" alt="Screenshot 2026-03-17 120107" src="https://github.com/user-attachments/assets/15a74acd-0c46-407a-a747-3c41fd6abdb3" />

