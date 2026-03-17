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

## 6. Démo

<img width="2385" height="1101" alt="image" src="https://github.com/user-attachments/assets/ab46f50b-99ea-42df-8146-6e5570a52496" />

