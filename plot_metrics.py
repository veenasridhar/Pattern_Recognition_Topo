import matplotlib.pyplot as plt
import wandb
import numpy as np

ENTITY  = "veenasridhar96-university-of-stuttgart"
PROJECT = "Masterthesis Curvess"
X_KEY   = "epoch"
METRIC_GROUPS = {
    "cldice": {
        "title": "clDice across 5 folds", "ylabel": "clDice",
        "model":    {"Train": "Train Cldice DL", "Val": "Val Cldice DL", "Test": "Test Cldice DL"},
        "baseline": {"Train": "Train_Img_Processing_Cldice",
                     "Val":   "Val_Img_Proccessing_Cldice",
                     "Test":  "Test_Img_Processing_Cldice"},
    },
}
 
def fetch_fold_series(metric, entity=ENTITY, project=PROJECT,
                      x_key=X_KEY, name_filter=None):
    """Return {fold: (x, y)} for one metric, keeping the newest run per fold."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")          # newest first by default
    series = {}
    for r in runs:
        if name_filter and name_filter not in r.name:
            continue
        fold = r.config.get("fold")
        if fold is None or fold in series:          # keep newest run per fold
            continue
        df = r.history(keys=[x_key, metric])        # full history for these keys
        if df is None or df.empty:
            continue
        df = df.dropna(subset=[metric]).sort_values(x_key)
        if df.empty:
            continue
        series[fold] = (df[x_key].to_numpy(), df[metric].to_numpy())
    if not series:
        raise ValueError(f"No runs with metric '{metric}' found in {project}")
    return series