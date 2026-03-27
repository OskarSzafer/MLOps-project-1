import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

class PXRDataset(Dataset):
    def __init__(self, smiles_list, targets):
        self.X = []
        self.y = targets
        
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048*4)
        
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = mfpgen.GetFingerprint(mol)
                self.X.append(list(fp))
            else:
                self.X.append([0] * (2048*4))
                
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PXRDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32, num_workers: int = 7):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        df = pd.read_csv(self.data_path, sep='\t')
        
        smiles = df['SMILES'].values
        targets = df['pEC50'].values

        X_temp_smi, X_test_smi, y_temp, y_test = train_test_split(
            smiles, targets, test_size=0.2, random_state=1
        )

        X_train_smi, X_val_smi, y_train, y_val = train_test_split(
            X_temp_smi, y_temp, test_size=0.2, random_state=42
        )

        if stage == 'fit' or stage is None:
            self.train_dataset = PXRDataset(X_train_smi, y_train)
            self.val_dataset = PXRDataset(X_val_smi, y_val)
            
        if stage == 'test' or stage is None:
            self.test_dataset = PXRDataset(X_test_smi, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)