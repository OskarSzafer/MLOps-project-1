import logging
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from data import PXRDataModule
from model import PXR_MLP

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    data_module = PXRDataModule(data_path="data/PXR_outlier_removed.smi", batch_size=32)
    model = PXR_MLP(
        hidden_dim=hidden_dim, 
        learning_rate=learning_rate, 
        dropout_rate=dropout_rate, 
        weight_decay=weight_decay
    )

    early_stop = EarlyStopping(monitor="loss/val", patience=6, mode="min")

    wandb_logger = WandbLogger(
        project="QSAR-MLP-MLOPS",
        entity="genai-mcda-put",
        name=f"trial_{trial.number}",
        group="optuna_trials",
        reinit=True
        )

    trainer = pl.Trainer(
        max_epochs=40,
        logger=wandb_logger,
        callbacks=[early_stop],
        enable_progress_bar=False,
        enable_model_summary=False
    )

    trainer.fit(model, datamodule=data_module)
    
    best_score = early_stop.best_score.item()
    
    wandb.finish()
    
    return best_score

def train_best_model(best_params):
    print("\nTrenowanie finalnego, zoptymalizowanego modelu...")
    
    wandb_logger = WandbLogger(
        project="QSAR-MLP-MLOPS",
        entity="genai-mcda-put",
        name="PXR-MLP-optimized",
        group="optimized"
    )

    data_module = PXRDataModule(data_path="data/PXR_outlier_removed.smi", batch_size=32)
    model = PXR_MLP(**best_params)

    early_stop = EarlyStopping(monitor="loss/val", patience=12, mode="min")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/optimized",
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

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    wandb.finish()

if __name__ == "__main__":
    print("Rozpoczynam optymalizację hiperparametrów (Optuna)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("\nNajlepsze znalezione hiperparametry:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    train_best_model(study.best_params)