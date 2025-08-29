#!/usr/bin/env python3
import json
import os
import numpy as np

# === CONFIGURAZIONE ===
DATASET_FILENAME = "gpt-3.5-turbo.json"   # Nome del file JSON nella stessa cartella di questo script
# DATASET_FILENAME = "human.json" 
TEST_RATIO = 0.2                       # Percentuale di dati per il test set (0.1 = 10%)
SEED = 42                              # Seed per shuffle riproducibile
CONCAT_PROMPT_SOLUTION = True          # True = concatena [prompt, solution] con \n\n
# ======================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, DATASET_FILENAME)

    # Carico dataset originale
    with open(dataset_path, "r") as f:
        data_raw = json.load(f)

    # Se è un dataset "Code" ([prompt, solution])
    if CONCAT_PROMPT_SOLUTION and isinstance(data_raw[0], list) and len(data_raw[0]) == 2:
        data = [rec[0].rstrip() + "\n\n" + rec[1].lstrip() for rec in data_raw]
    else:
        data = data_raw

    # Shuffle riproducibile
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(data))
    rng.shuffle(idx)

    # Split
    split_point = int(round(len(data) * (1 - TEST_RATIO)))
    train_idx = idx[:split_point]
    test_idx  = idx[split_point:]

    train_data = [data[i] for i in train_idx]
    test_data  = [data[i] for i in test_idx]

    # Path output
    base, ext = os.path.splitext(dataset_path)
    train_path = f"{base}_train{ext}"
    test_path  = f"{base}_test{ext}"

    # Salvataggio
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"[OK] Train salvato in: {train_path} ({len(train_data)} esempi)")
    print(f"[OK] Test salvato in:  {test_path} ({len(test_data)} esempi)")

if __name__ == "__main__":
    main()
