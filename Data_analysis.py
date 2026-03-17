import pandas as pd

# 1. Charger les données avec le bon nom de fichier !
df = pd.read_csv("HRDataset_v14.csv")

# 2. Aperçu général
print("--- LES 5 PREMIÈRES LIGNES ---")
print(df.head()) # Remplacé display par print

print("\n--- STRUCTURE DU DATASET ---")
df.info() 

# 3. Statistiques descriptives sur les colonnes numériques
print("\n--- STATISTIQUES GLOBALES ---")
print(df.describe()) # Remplacé display par print

# 4. Focus sur la cible du Hackathon : Les démissions
print("\n--- ANALYSE DU TURNOVER ---")
print("Répartition des employés (1 = Terminé/Démission, 0 = Toujours en poste) :")
print(df['Termd'].value_counts(normalize=True) * 100) 

print("\nTop 5 des raisons de départ :")
print(df['TermReason'].value_counts().head(50))

# 5. Focus sur les données sensibles (pour la Cybersécurité)
print("\n--- DONNÉES SENSIBLES ---")
print("Répartition par sexe :")
print(df['Sex'].value_counts())
print("\nRépartition par ethnie (RaceDesc) :")
print(df['RaceDesc'].value_counts())

# 6. Vérifier les valeurs manquantes
print("\n--- VALEURS MANQUANTES ---")
print(df.isnull().sum()[df.isnull().sum() > 0])