import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb

from data import PXRDataModule
from model import PXR_MLP

def main():
    wandb_logger = WandbLogger(
        project="QSAR-MLP-MLOPS",
        entity="genai-mcda-put",
        name="PXR-MLP-baseline",
        group="baseline"
    )

    data_module = PXRDataModule(data_path="data/PXR_outlier_removed.smi", batch_size=32)
    model = PXR_MLP(hidden_dim=256, learning_rate=0.005, dropout_rate=0.4, weight_decay=1e-3)

    early_stop = EarlyStopping(monitor="loss/val", patience=10, mode="min")
    
    # Zapisywanie najlepszego modelu do konkretnego folderu
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/baseline",
        filename="best_model",
        monitor="loss/val",
        mode="min",
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=90,
        logger=wandb_logger,
        callbacks=[early_stop, checkpoint_callback],
        log_every_n_steps=5
    )

    print("Rozpoczynam trenowanie modelu bazowego (Baseline)...")
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    wandb.finish()

if __name__ == "__main__":
    main()