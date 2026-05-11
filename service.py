import bentoml
import numpy as np
import torch

from model.model import PXR_MLP
from rdkit import Chem
from rdkit.Chem import AllChem

@bentoml.service(resources={"cpu": 1})
class PXRService:
    def __init__(self):
        self.model = PXR_MLP.load_from_checkpoint("model/best_model.ckpt")
        self.model.eval()

    @bentoml.api
    def predict(self, smiles: str) -> float:

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=8192)
        input_vector = np.array(fp)
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(input_tensor).item()
        
        return float(prediction)