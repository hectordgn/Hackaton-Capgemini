import pandas as pd
import hashlib

df = pd.read_csv("HRDataset_v14.csv")

SEL_SECRET = "Explainability"

def hacher_avec_sel(valeur):
    chaine = str(valeur) + SEL_SECRET
    return hashlib.sha256(chaine.encode('utf-8')).hexdigest()[:10]


df['Hash_ID'] = df['EmpID'].apply(hacher_avec_sel)

# FICHIER 1 : HR_KEYS (Le fichier secret de la DRH)
cols_keys = ['EmpID', 'Employee_Name']
if 'Email' in df.columns:
    cols_keys.append('Email')
    
cols_keys.append('Hash_ID') 

df_keys = df[cols_keys]
df_keys.to_csv("HR_Keys.csv", index=False)
print("🔒 Fichier secret généré : 'HR_Keys.csv'")



# FICHIER 2 : HR_ANALYTICS (Pour le modèle de Machine Learning)
df_analytics = df.copy()

df_analytics['EmpID'] = df_analytics['Hash_ID']

colonnes_analytics = [
    'EmpID','MarriedID','MaritalStatusID','GenderID','EmpStatusID','DeptID',
    'PerfScoreID','FromDiversityJobFairID','Salary','Termd','PositionID','Position',
    'State','Sex','MaritalDesc','CitizenDesc','HispanicLatino','RaceDesc','DateofHire',
    'DateofTermination','TermReason','EmploymentStatus','Department','ManagerName',
    'ManagerID','RecruitmentSource','PerformanceScore','EngagementSurvey',
    'EmpSatisfaction','SpecialProjectsCount','LastPerformanceReview_Date',
    'DaysLateLast30','Absences'
]

colonnes_presentes = [col for col in colonnes_analytics if col in df_analytics.columns]
df_analytics = df_analytics[colonnes_presentes]

df_analytics.to_csv("HR_Analytics.csv", index=False)
print(" Dataset ML généré avec succès : 'HR_Analytics.csv'")