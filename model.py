import torch
import torch.nn as nn
import pytorch_lightning as pl

class PXR_MLP(pl.LightningModule):
    def __init__(self, input_dim=2048*4, hidden_dim=256, learning_rate=0.005, dropout_rate=0.4, weight_decay=1e-3):
        super().__init__()
        self.save_hyperparameters() 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1) 
        )
        self.criterion = nn.MSELoss() 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log('loss/train', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        val_loss = self.criterion(preds, y)
        self.log('loss/val', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        test_loss = self.criterion(preds, y)
        self.log('loss/test', test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)