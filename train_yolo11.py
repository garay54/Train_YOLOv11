#!/usr/bin/env python3
"""
train_yolo11_optuna_plot.py — Optimización automática de hiperparámetros para YOLOv11-seg con Optuna + gráficos.

Requisitos:
    pip install ultralytics optuna pyyaml matplotlib
"""

import os
import yaml
import optuna
import matplotlib.pyplot as plt
import torch
import gc
from ultralytics import YOLO


# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
DATASET = "data.yaml"               # Ruta al YAML del dataset
BASE_MODEL = "yolo11s.pt"           # Modelo base: "yolo11s.pt" (bbox) o "yolo11s-seg.pt" (seg)
PROJECT = "runs/optuna_search"      # Carpeta de resultados
DEVICE = "0"                        # GPU o "cpu"
EPOCHS = 100                        # Épocas por prueba
WORKERS = 8                         # Hilos del dataloader
N_TRIALS = 20                       # Número de combinaciones a probar


# ======================================================
# FUNCIÓN OBJETIVO PARA OPTUNA
# ======================================================
def objective(trial):
    """
    Entrena YOLOv11 con hiperparámetros propuestos y devuelve el mAP50-95.
    """

    # Limpiar memoria GPU antes de cada trial
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    try:
        # Hiperparámetros a optimizar
        imgsz = trial.suggest_categorical("imgsz", [416, 512, 640])
        batch = trial.suggest_categorical("batch", [8, 16, 32])
        lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
        momentum = trial.suggest_float("momentum", 0.7, 0.99)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        exp_name = f"trial_{trial.number}_img{imgsz}_b{batch}_{optimizer}"
        model = YOLO(BASE_MODEL)

        results = model.train(
            data=DATASET,
            epochs=EPOCHS,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            optimizer=optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT,
            name=exp_name,
            verbose=False
        )

        # Leer métricas del archivo de resultados
        metrics_path = os.path.join(results.save_dir, "results.yaml")
        if not os.path.exists(metrics_path):
            print(f"⚠️ No se encontró results.yaml para trial {trial.number}")
            return 0.0

        with open(metrics_path, "r") as f:
            data = yaml.safe_load(f)

        mAP = data.get("metrics/mAP50-95(B)", 0)
        recall = data.get("metrics/recall(B)", 0)
        precision = data.get("metrics/precision(B)", 0)

        trial.set_user_attr("recall", recall)
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("path", results.save_dir)

        print(f"✅ Trial {trial.number} completado → mAP50-95={mAP:.4f}, Recall={recall:.4f}")
        
        # Limpiar memoria después del entrenamiento
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return mAP

    except RuntimeError as e:
        # Capturar errores de memoria GPU y otros errores de runtime
        error_msg = str(e)
        # Obtener parámetros de manera segura
        batch_val = trial.params.get("batch", "N/A")
        imgsz_val = trial.params.get("imgsz", "N/A")
        
        if "CUDA" in error_msg or "cuDNN" in error_msg or "out of memory" in error_msg.lower():
            print(f"❌ Trial {trial.number} falló por falta de memoria GPU (batch={batch_val}, imgsz={imgsz_val})")
            print(f"   Error: {error_msg[:100]}...")
            trial.set_user_attr("error", "GPU_MEMORY_ERROR")
            trial.set_user_attr("error_msg", error_msg[:200])
        else:
            print(f"❌ Trial {trial.number} falló con error: {error_msg[:100]}...")
            trial.set_user_attr("error", "RUNTIME_ERROR")
            trial.set_user_attr("error_msg", error_msg[:200])
        
        # Limpiar memoria después del error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Lanzar excepción para que Optuna la maneje y continúe con el siguiente trial
        raise optuna.TrialPruned(f"Trial falló: {error_msg[:100]}")
    
    except Exception as e:
        # Capturar cualquier otro error inesperado
        error_msg = str(e)
        print(f"❌ Trial {trial.number} falló con error inesperado: {error_msg[:100]}...")
        trial.set_user_attr("error", "UNEXPECTED_ERROR")
        trial.set_user_attr("error_msg", error_msg[:200])
        
        # Limpiar memoria después del error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        raise optuna.TrialPruned(f"Trial falló: {error_msg[:100]}")


# ======================================================
# FUNCIÓN PRINCIPAL
# ======================================================
def main():
    os.makedirs(PROJECT, exist_ok=True)
    print("\n🚀 Iniciando optimización automática de hiperparámetros con Optuna...")
    print(f"📂 Resultados en: {PROJECT}\n")

    study = optuna.create_study(
        study_name="yolo11_hyperopt",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()  # Pruner para detener trials que no mejoran
    )

    # Optimizar con manejo de errores
    try:
        study.optimize(objective, n_trials=N_TRIALS, catch=(RuntimeError, optuna.TrialPruned))
    except KeyboardInterrupt:
        print("\n⚠️ Optimización interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la optimización: {e}")

    # ======================================================
    # RESUMEN FINAL
    # ======================================================
    print("\n🏁 Optimización completada.")
    
    # Contar trials exitosos vs fallidos
    successful_trials = [t for t in study.trials if t.value is not None]
    failed_trials = [t for t in study.trials if t.value is None]
    print(f"📊 Trials exitosos: {len(successful_trials)}/{len(study.trials)}")
    if failed_trials:
        print(f"⚠️ Trials fallidos: {len(failed_trials)}")
        gpu_errors = [t for t in failed_trials if t.user_attrs.get("error") == "GPU_MEMORY_ERROR"]
        if gpu_errors:
            print(f"   - Errores de memoria GPU: {len(gpu_errors)}")
    
    if study.best_value is not None:
        print(f"🔝 Mejor mAP50-95: {study.best_value:.4f}")
        print("📊 Mejores hiperparámetros encontrados:")
        for k, v in study.best_params.items():
            print(f"   {k}: {v}")
    else:
        print("❌ No se completó ningún trial exitosamente.")
        return

    resumen_path = os.path.join(PROJECT, "optuna_best.yaml")
    with open(resumen_path, "w") as f:
        yaml.safe_dump({
            "best_mAP": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number
        }, f)
    print(f"\n📁 Resumen guardado en: {resumen_path}")

    # ======================================================
    # GRÁFICO DE RESULTADOS
    # ======================================================
    print("\n📊 Generando gráfico de evolución...")

    trial_numbers = [t.number for t in study.trials if t.value is not None]
    mAP_values = [t.value for t in study.trials if t.value is not None]
    recall_values = [t.user_attrs.get("recall", 0) for t in study.trials if t.value is not None]

    plt.figure(figsize=(10, 6))
    plt.plot(trial_numbers, mAP_values, "o-", label="mAP50-95", linewidth=2)
    plt.plot(trial_numbers, recall_values, "s--", label="Recall", linewidth=2, alpha=0.8)
    plt.xlabel("Trial")
    plt.ylabel("Métrica")
    plt.title("Evolución de mAP y Recall por trial (Optuna YOLOv11)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plot_path = os.path.join(PROJECT, "optuna_results_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"📈 Gráfico guardado en: {plot_path}\n")

    # ======================================================
    # TOP 5 CONFIGURACIONES
    # ======================================================
    print("🏆 Top 5 configuraciones:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]
    for t in trials_sorted:
        print(f"Trial {t.number} → mAP={t.value:.4f}, Recall={t.user_attrs.get('recall', 0):.4f}")
        print(f"   Params: {t.params}")
        print(f"   Path: {t.user_attrs.get('path', '-')}\n")


if __name__ == "__main__":
    main()
