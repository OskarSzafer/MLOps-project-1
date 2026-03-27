import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from data import PXRDataModule
from model import PXR_MLP

def get_predictions(model, dataloader):
    model.eval()
    device = model.device
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            preds = model(x)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy()) 
            
    return np.array(y_true).flatten(), np.array(y_pred).flatten()

def evaluate_and_plot(model_path, mode_name):
    print(f"\n--- Ewaluacja modelu: {mode_name.upper()} ---")
    
    model = PXR_MLP.load_from_checkpoint(model_path)
    
    data_module = PXRDataModule(data_path="data/PXR_outlier_removed.smi", batch_size=64, num_workers=7)
    data_module.setup() 

    y_train_true, y_train_pred = get_predictions(model, data_module.train_dataloader())
    y_test_true, y_test_pred = get_predictions(model, data_module.test_dataloader())

    r2_train = r2_score(y_train_true, y_train_pred)
    r2_test = r2_score(y_test_true, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test_true, y_test_pred))

    plt.figure(figsize=(10, 10))
    
    plt.scatter(y_train_true, y_train_pred, alpha=0.3, color='#27ae60', edgecolor='none', s=50,
                label=f'Training Data ($R^2 = {r2_train:.3f}$ | RMSE = {rmse_train:.3f})')
    plt.scatter(y_test_true, y_test_pred, alpha=0.7, color='#2c3e50', edgecolor='w', s=80,
                label=f'Test Data ($R^2 = {r2_test:.3f}$ | RMSE = {rmse_test:.3f})')

    all_obs = np.concatenate([y_train_true, y_test_true])
    all_pred = np.concatenate([y_train_pred, y_test_pred])
    min_val, max_val = min(all_obs.min(), all_pred.min()) - 0.5, max(all_obs.max(), all_pred.max()) + 0.5

    plt.plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=2, label='Perfect Fit')

    m, b = np.polyfit(y_test_true, y_test_pred, 1)
    plt.plot(y_test_true, m * y_test_true + b, color='#3498db', alpha=0.5, linewidth=2, label=f'Test Trend (Slope={m:.2f})')

    plt.xlabel('Experimental pEC50', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted pEC50', fontsize=14, fontweight='bold')
    
    title_suffix = "Optimized (Optuna)" if mode_name == "optimized" else "Baseline"
    plt.title(f'QSAR Model Performance: MLP {title_suffix} (Train vs Test)', fontsize=16)
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9)

    hparams = model.hparams
    best_params_str = "\n".join([f"{k}: {v if not isinstance(v, float) else f'{v:.5f}'}" 
                                 for k, v in hparams.items() if k not in ['model', 'criterion']])
    
    plt.gcf().text(0.92, 0.5, f"Model Hyperparameters:\n\n{best_params_str}", fontsize=10, verticalalignment='center')

    plt.tight_layout()
    output_filename = f'pxr_mlp_{mode_name}_performance_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Zapisano wykres: {output_filename}")
    plt.close()

def main():
    # Słownik ścieżek do sprawdzenia
    models_to_evaluate = {
        "baseline": "models/baseline/best_model.ckpt",
        "optimized": "models/optimized/best_model.ckpt"
    }

    found_any = False
    for mode, path in models_to_evaluate.items():
        if os.path.exists(path):
            evaluate_and_plot(path, mode)
            found_any = True
        else:
            print(f"Pominięto '{mode}': brak pliku {path}")

    if not found_any:
        print("\nNie znaleziono żadnych modeli. Uruchom najpierw `python train.py` lub `python tune.py`.")

if __name__ == "__main__":
    main()