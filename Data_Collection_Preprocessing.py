import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Load the dataset
# Assume the dataset has columns: 'compound_smiles', 'target_sequence', 'interaction'
data = pd.read_csv('drug_target_interactions.csv')

# Function to compute molecular descriptors
def compute_molecular_descriptors(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    descriptors = {
        'MolWt': Descriptors.MolWt(molecule),
        'MolLogP': Descriptors.MolLogP(molecule),
        'NumHDonors': Descriptors.NumHDonors(molecule),
        'NumHAcceptors': Descriptors.NumHAcceptors(molecule)
    }
    return descriptors

# Function to compute protein descriptors
def compute_protein_descriptors(sequence):
    analysis = ProteinAnalysis(sequence)
    descriptors = {
        'MolecularWeight': analysis.molecular_weight(),
        'Aromaticity': analysis.aromaticity(),
        'IsoelectricPoint': analysis.isoelectric_point(),
        'InstabilityIndex': analysis.instability_index()
    }
    return descriptors

# Compute descriptors for all compounds and proteins
compound_descriptors = data['compound_smiles'].apply(compute_molecular_descriptors)
protein_descriptors = data['target_sequence'].apply(compute_protein_descriptors)

# Convert descriptors to DataFrame
compound_df = pd.DataFrame(compound_descriptors.tolist())
protein_df = pd.DataFrame(protein_descriptors.tolist())

# Combine descriptors with interaction label
features = pd.concat([compound_df, protein_df], axis=1)
target = data['interaction']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
