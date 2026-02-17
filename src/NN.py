# ===========================
# 1) ADAPTERS (feature parsing)
# ===========================
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import ast
import sys

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---- Base adapter ----
class FeatureAdapter(ABC):
    """
    Transforms a flat numpy row (F,) or matrix (N,F) into tensors that a model expects.
    Must also report num_actions.
    """
    @abstractmethod
    def build_from_header(self, feature_names: list[str], meta: dict): ...
    @abstractmethod
    def transform(self, X_np: np.ndarray) -> dict[str, torch.Tensor]: ...
    @abstractmethod
    def num_actions(self) -> int: ...

# ---- Adapter A: Assign (ctx + per-phys + mask) based on your current header ----
class PerPhyAdapter(FeatureAdapter):
    """
    Expects header with:
      Context: ["t_current","remaining_release","priority_is_p1"]
      For p in [0..P-1]:
        ["phys{p}_is_preferred","phys{p}_cap_remaining_norm","phys{p}_load_pressure"]
      Masks (repeated per-phys): for p in [0..P-1]: ["mask_phys{p}","mask_reject"]
    Produces:
      ctx: (N,3)
      phys: (N,P,3)
      mask: (N,P+1) with 1=invalid, 0=valid
    """
    def __init__(self):
        self._built = False

    def build_from_header(self, feature_names: list[str], meta: dict):
        P = int(meta["num_class"] - 1)
        self.P = P
        name_to_idx = {n:i for i,n in enumerate(feature_names)}

        # context indices
        ctx_names = ["t_current", "remaining_release", "priority_is_p1"]
        self.ctx_idx = [name_to_idx[n] for n in ctx_names]

        # per-phys feature indices
        self.phys_idx = []
        for p in range(P):
            self.phys_idx.append([
                name_to_idx[f"phys{p}_is_preferred"],
                name_to_idx[f"phys{p}_cap_remaining_norm"],
                #name_to_idx[f"phys{p}_load_pressure"],
            ])

        # masks: take all mask_phys{p}, and just one mask_reject
        self.mask_idx_phys = [name_to_idx[f"mask_phys{p}"] for p in range(P)]
        # some datasets repeat mask_reject; grab the first
        self.mask_idx_reject = name_to_idx.get("mask_reject", None)
        if self.mask_idx_reject is None:
            # fallback: after last mask_phys
            self.mask_idx_reject = max(self.mask_idx_phys) + 1

        self._built = True

    def transform(self, X_np: np.ndarray) -> dict[str, torch.Tensor]:
        assert self._built, "Adapter not built. Call build_from_header first."
        if X_np.ndim == 1:
            X_np = X_np[None, :]
        #N = X_np.shape[0]; P = self.P

        ctx = X_np[:, self.ctx_idx]                                    # (N,3)
        phys = np.concatenate([X_np[:, cols][:,None,:] for cols in self.phys_idx], axis=1)  # (N,P,3)
        mask_phys = X_np[:, self.mask_idx_phys].astype(np.float32)     # (N,P)
        mask_rej  = X_np[:, [self.mask_idx_reject]].astype(np.float32) # (N,1)
        mask = np.concatenate([mask_phys, mask_rej], axis=1)           # (N,P+1)

        return {
            "ctx":  torch.tensor(ctx,  dtype=torch.float32),
            "phys": torch.tensor(phys, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }

    def num_actions(self) -> int:
        return self.P + 1

# ---- Adapter B: Plain flat classifier (no structure) ----
class FlatAdapter(FeatureAdapter):
    """Treats the whole row as a single flat input: x: (N,F)"""
    def build_from_header(self, feature_names: list[str], meta: dict):
        self.F = len(feature_names)
        P = int(meta["num_class"] - 1)
        self.P = P
        name_to_idx = {n:i for i,n in enumerate(feature_names)}
        self.mask_idx_phys = [name_to_idx[f"mask_phys{p}"] for p in range(P)]
        # some datasets repeat mask_reject; grab the first
        self.mask_idx_reject = name_to_idx.get("mask_reject", None)
        if self.mask_idx_reject is None:
            # fallback: after last mask_phys
            self.mask_idx_reject = max(self.mask_idx_phys) + 1

    def transform(self, X_np: np.ndarray) -> dict[str, torch.Tensor]:
        if X_np.ndim == 1:
            X_np = X_np[None, :]
        mask_phys = X_np[:, self.mask_idx_phys].astype(np.float32)     # (N,P)
        mask_rej  = X_np[:, [self.mask_idx_reject]].astype(np.float32) # (N,1)
        mask = np.concatenate([mask_phys, mask_rej], axis=1)   
        return {"x": torch.tensor(X_np, dtype=torch.float32),
                "mask": torch.tensor(mask, dtype=torch.float32)}

    def num_actions(self) -> int:
        return self.P + 1

# ======================================
# 2) MODELS (architecture is swappable)
# ======================================

# ---- Model A: Shared per-phys scorer + reject head (masked softmax) ----
class SharedPhysNet(nn.Module):
    """
    Inputs via PerPhyAdapter:
      - ctx:  (B, c_dim)
      - phys: (B, P, p_dim)
      - mask: (B, P+1)  [1=invalid]
    """
    def __init__(self, p_dim: int, c_dim: int, hidden: int = 64):
        super().__init__()
        self.ctx = nn.Sequential(
            nn.Linear(c_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.phys = nn.Sequential(
            nn.Linear(p_dim + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.rej  = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ctx  = batch["ctx"]    # (B,c)
        phys = batch["phys"]   # (B,P,p_dim)
        mask = batch.get("mask", None)  # (B,P+1) float32 {0,1}

        B, P, Dp = phys.shape
        c = self.ctx(ctx)                           # (B,h)
        cP = c.unsqueeze(1).expand(-1, P, -1)       # (B,P,h)
        logits_p = self.phys(torch.cat([phys, cP], dim=-1)).squeeze(-1)  # (B,P)
        logit_rej = self.rej(c).squeeze(-1).unsqueeze(1)                 # (B,1)
        logits = torch.cat([logits_p, logit_rej], dim=1)                 # (B,P+1)
        if mask is not None:
            logits = logits.masked_fill(mask.bool(), -1e9)
        return logits

# ---- Model B: Simple flat MLP classifier (no mask) ----
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]
        mask = batch.get("mask", None)  # (B,P+1) float32 {0,1}
        logits = self.net(x)
        logits = logits.masked_fill(mask.bool(), -1e9)
        return logits

# ======================================
# 3) LOSS (handles sample weights cleanly)
# ======================================
class SampleWeightedCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none")
    def forward(self, logits, target, weight=None):
        loss = self.ce(logits, target)  # (B,)
        #print("NNLoss",  loss)
        if weight is not None:
            loss = loss * weight.view(-1)
        return loss.mean()
    
class SampleWeightedSoftCELoss(nn.Module):
    """Soft-label cross-entropy: L = - sum_k p_k * log q_k"""
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
    def forward(self, logits: torch.Tensor, target_probs: torch.Tensor, weight: torch.Tensor):
        log_q = F.log_softmax(logits, dim=1)                  # (B,K)
        p = target_probs.clamp_min(self.eps)                  # (B,K)
        loss = -(p * log_q).sum(dim=1)                        # (B,)
        if weight is not None:
            loss = loss * weight.view(-1)
        return loss.mean()


# ======================================
# 4) REGISTRY (choose adapter+model from config)
# ======================================
ADAPTERS = {
    "per_phy": PerPhyAdapter,   # ctx+per-phys+mask (your current setup)
    "flat":    FlatAdapter,         # plain flat rows
}


# ======================================
# 5) TRAINER (generic, model-agnostic)
# ======================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
import numpy as np
import copy
import tqdm
import torch.optim as optim
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)


def _parse_soft_targets_column(col: pd.Series, num_classes: int) -> np.ndarray:
    out = []
    for v in col.astype(str).values:
        try:
            arr = np.array(ast.literal_eval(v), dtype=np.float32)
        except Exception:
            arr = np.full(num_classes, 1.0/num_classes, dtype=np.float32)
        if arr.shape[0] != num_classes:
            tmp = np.zeros(num_classes, dtype=np.float32)
            m = min(num_classes, arr.shape[0])
            tmp[:m] = arr[:m]
            arr = tmp
        s = float(arr.sum())
        arr = arr / s if s > 0 else np.full(num_classes, 1.0/num_classes, dtype=np.float32)
        out.append(arr)
    return np.vstack(out)

class GenericTrainer(nn.Module):
    def __init__(self, adapter, model, num_actions, lr=1e-3, batch_size=256, epochs=50,
                 use_weight=False, target_mode: str = "hard", dist_col: str = "expert_distribution"):
        super().__init__()
        
        self.adapter = adapter
        self.model = model
        self.num_actions = num_actions
        self.loss_fn_hard = SampleWeightedCELoss()
        self.loss_fn_soft = SampleWeightedSoftCELoss()
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_weight = use_weight
        self.target_mode = target_mode
        self.dist_col = dist_col
        self.rng = np.random.default_rng(42)
        self.train_losses = pd.DataFrame( columns=["losses"])
        

    def _to_batch(self, X_np: np.ndarray, y_np: np.ndarray , w_np: np.ndarray ):
        batch = self.adapter.transform(X_np)
        target = None if y_np is None else torch.tensor(y_np, dtype=torch.long)
        weight = None if w_np is None else torch.tensor(w_np, dtype=torch.float32)
        return batch, target, weight

    def fit_eval(self, df: pd.DataFrame, feature_names: list[str], target_col: str, weight_col: str =None):
        X = df[feature_names].values
        w = df[weight_col].values if (weight_col is not None) else np.ones(len(df))
    
        if self.target_mode == "soft":
            Ysoft = _parse_soft_targets_column(df[self.dist_col], self.num_actions)     # (N,K)
            y_hard = Ysoft.argmax(axis=1).astype(int)                                   # for metrics only
            Xtr, Xte, ytr_h, yte_h, Ytr_s, Yte_s, wtr, wte = train_test_split(
                X, y_hard, Ysoft, w, test_size=0.2, random_state=8
            )
        else:
            y_hard = df[target_col].values.astype(int)
            Xtr, Xte, ytr_h, yte_h, wtr, wte = train_test_split(X, y_hard, w, test_size=0.2, random_state=8)
            Ytr_s = Yte_s = None
    
        train_losses, best_val = [], float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        Ntr = Xtr.shape[0]
        
        idxs = np.arange(0, Ntr, self.batch_size)
        #print(Ntr, idxs)
        for epoch in range(self.epochs):
            self.model.train()
            self.rng.shuffle(idxs)
            running = 0.0
    
            for start in idxs:
                sl = slice(start, min(start + self.batch_size, Ntr))
                batch = self.adapter.transform(Xtr[sl].astype(np.float32))
                logits = self.model(batch)
    
                if self.target_mode == "soft":
                    y_s = torch.tensor(Ytr_s[sl], dtype=torch.float32)
                    wt  = torch.tensor(wtr[sl], dtype=torch.float32) if self.use_weight else None
                    loss = self.loss_fn_soft(logits, y_s, wt)
                else:
                    y_h = torch.tensor(ytr_h[sl], dtype=torch.long)
                    wt  = torch.tensor(wtr[sl], dtype=torch.float32) if self.use_weight else None
                    loss = self.loss_fn_hard(logits, y_h, wt)
    
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                running += loss.item() * (sl.stop - sl.start)
    
            # validation
            self.model.eval()
            with torch.no_grad():
                bte = self.adapter.transform(Xte.astype(np.float32))
                v_logits = self.model(bte)
                if self.target_mode == "soft":
                    y_s = torch.tensor(Yte_s, dtype=torch.float32)
                    wt  = torch.tensor(wte, dtype=torch.float32) if self.use_weight else None
                    val_loss = self.loss_fn_soft(v_logits, y_s, wt).item()
                else:
                    y_h = torch.tensor(yte_h, dtype=torch.long)
                    wt  = torch.tensor(wte, dtype=torch.float32) if self.use_weight else None
                    val_loss = self.loss_fn_hard(v_logits, y_h, wt).item()
                    
    
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
    
            train_losses.append(running / len(Xtr))
            if (epoch) % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {train_losses[-1]:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
    
        # restore best
        self.model.load_state_dict(best_state)
        self.train_losses = pd.concat([self.train_losses, pd.DataFrame(train_losses, columns=["losses"])])
    
        # metrics (use hard labels for interpretability)
        self.model.eval()
        with torch.no_grad():
            ball = self.adapter.transform(X.astype(np.float32))
            logits_all = self.model(ball)
            probs = F.softmax(logits_all, dim=1).cpu().numpy()
            y_pred = probs.argmax(axis=1)
    
        for i in range(probs.shape[1]):
            df[f"predict_prob_{i}"] = probs[:, i]
        df["predicted_classes"] = y_pred

        if self.target_mode == "soft":
            y_for_report = Ysoft.argmax(axis=1).astype(int)
        else:
            y_for_report = y_hard
        print("Classification Report")
        print(classification_report(y_for_report, y_pred))
    
        stats = {
            "precision_micro": precision_score(y_for_report, y_pred, average="micro"),
            "precision_macro": precision_score(y_for_report, y_pred, average="macro"),
            "recall_micro": recall_score(y_for_report, y_pred, average="micro"),
            "recall_macro": recall_score(y_for_report, y_pred, average="macro"),
            "accuracy": accuracy_score(y_for_report, y_pred),
            "data": len(df),
        }
        return df, stats


    def predict_proba(self, x_row: np.ndarray) -> np.ndarray:
        self.model.eval()
        batch = self.adapter.transform(x_row.astype(np.float32))
        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs
    
    # ------------------------------
# Permutation Feature Importance
# ------------------------------
    @torch.no_grad()
    def _predict_logits_batched(self, X_np: np.ndarray, batch_size: int = 4096) -> torch.Tensor:
        """Return logits for a whole matrix X_np in batches (adapter-agnostic)."""
        self.model.eval()
        outs = []
        N = X_np.shape[0]
        for start in range(0, N, batch_size):
            sl = slice(start, min(start + batch_size, N))
            batch = self.adapter.transform(X_np[sl].astype(np.float32))
            logits = self.model(batch)
            outs.append(logits.cpu())
        return torch.cat(outs, dim=0)
    
    @torch.no_grad()
    def _metric_score(self, y_true: np.ndarray, logits: torch.Tensor, metric: str = "accuracy") -> float:
        """Compute a scalar score from logits for a chosen metric."""
        if metric == "accuracy":
            y_pred = logits.softmax(dim=1).argmax(dim=1).cpu().numpy()
            return float((y_pred == y_true).mean())
        elif metric == "loss":
            # negative CE (higher is better) so we can do baseline - permuted for importance
            y_t = torch.tensor(y_true, dtype=torch.long)
            ce = nn.CrossEntropyLoss(reduction="mean")(logits, y_t)
            return float(-ce.item())
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'accuracy' or 'loss'.")
    
    def permutation_importance(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        target_col: str,
        *,
        repeats: int = 5,
        metric: str = "accuracy",
        exclude_prefixes: tuple[str, ...] = ("mask_",),   # avoid permuting masks by default
        batch_size: int = 4096,
        random_state: int = 123
    ) -> pd.DataFrame:
        """
        Compute permutation importance over the provided dataframe.
    
        Importance = baseline_score - permuted_score  (higher => more important).
    
        Args:
          df               : DataFrame with features + target
          feature_names    : columns to use as features
          target_col       : label column (int class id)
          repeats          : times to permute each feature
          metric           : 'accuracy' or 'loss' (loss is returned as negative CE)
          exclude_prefixes : feature name prefixes to skip (e.g., masks)
          batch_size       : eval batch size
          random_state     : RNG seed for permutations
    
        Returns:
          DataFrame with columns:
            ['feature','importance_mean','importance_std','baseline','permuted_mean','permuted_std','n_repeats']
        """
        # pick only a copy of features and labels to avoid mutating df
        X = df[feature_names].to_numpy(copy=True)
        y = df[target_col].to_numpy(dtype=int)
    
        # filter features to permute (skip excluded prefixes)
        keep_idx = []
        kept_names = []
        for j, name in enumerate(feature_names):
            if not any(name.startswith(pref) for pref in exclude_prefixes):
                keep_idx.append(j)
                kept_names.append(name)
    
        # baseline score
        base_logits = self._predict_logits_batched(X, batch_size=batch_size)
        baseline = self._metric_score(y, base_logits, metric=metric)
    
        rng = np.random.default_rng(random_state)
        imp_means = []
        imp_stds = []
        perm_means = []
        perm_stds = []
    
        # pre-allocated buffer for permutation
        X_perm = X.copy()
    
        for col_idx in keep_idx:
            scores = []
            for _ in range(repeats):
                # permute column col_idx
                rng.shuffle(X_perm[:, col_idx])
                logits_perm = self._predict_logits_batched(X_perm, batch_size=batch_size)
                score_p = self._metric_score(y, logits_perm, metric=metric)
                scores.append(score_p)
                # restore original column
                X_perm[:, col_idx] = X[:, col_idx]
    
            scores = np.asarray(scores, dtype=float)
            perm_means.append(scores.mean())
            perm_stds.append(scores.std(ddof=1) if repeats > 1 else 0.0)
            # importance = baseline - permuted
            imp = baseline - scores
            imp_means.append(imp.mean())
            imp_stds.append(imp.std(ddof=1) if repeats > 1 else 0.0)
    
        out = pd.DataFrame({
            "feature": kept_names,
            "importance_mean": imp_means,
            "importance_std": imp_stds,
            "baseline": baseline,
            "permuted_mean": perm_means,
            "permuted_std": perm_stds,
            "n_repeats": repeats,
            "metric": metric,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    
        return out


# ======================================
# 6) HIGH-LEVEL WRAPPER (like your SupervisedNN)
# ======================================
class SupervisedNN:
    """
    Configuration-driven, swappable adapter + model.

    config example:
      {
        "adapter": "assign_flat",
        "model": "shared_phys",

        "train": true,
        "num_actions": 5,        # only needed by FlatAdapter
        "hidden": 64,
        "epochs": 50,
        "batch_size": 256,
        "lr": 1e-3,
        "use_weight": false
      }
    """
    def __init__(self, feature_names, config, folder=None, df=pd.DataFrame()):
        self.feature_names = feature_names
        self.target = config.get("target")
        meta = {"num_class": config.get("num_class", 1)}
        set_seed(10)
        self.model_stats = pd.DataFrame()
        self.featur_imp = pd.DataFrame()
        self.state_counter = []
        self.fit_count = 0
        self.use_weight = config.get("use_weight", False)
        self.num_class = config["num_class"]
        self.num_epochs = config["epochs"]
        self.target_mode = config["target_mode"]
        self.batch_size = 64
        self.chooser = np.random.default_rng(10)

        # Action selection strategy
        self.choose_action = (
            self.random_action if config.get("action_type") == "random" else self.max_prob_action
        )
        # build adapter
        AdapterCls = ADAPTERS[config["adapter"]]
        self.adapter = AdapterCls()
        self.adapter.build_from_header(feature_names, meta)
        self.fit_count = 0

        if config.get("train", False):
            self.df = df
            if config["adapter"] == "per_phy":
                # dims are fixed by PerPhyAdapter (3 ctx, 3 per-phys). change here if your schema changes.
                self.model = SharedPhysNet(p_dim=2, c_dim=3, hidden=config.get("hidden", 64))
            elif config["adapter"] == "flat":
                self.model = MLPClassifier(input_dim=len(self.feature_names), num_classes=self.num_class,
                                           hidden=config.get("hidden", 128),
                                           depth=config.get("depth", 2))
            else:
                raise ValueError("Unknown model: " + config["model"])
                
            self.trainer = GenericTrainer(
                adapter=self.adapter,
                model=self.model,
                num_actions=config.get("num_class", 1),
                lr=config.get("lr", 1e-3),
                batch_size=config.get("batch_size", 256),
                epochs=config.get("epochs", 50),
                use_weight=config.get("use_weight", False),
                target_mode=self.target_mode,                 # <— soft labels
                dist_col="expert_distribution",     # <— your column
            )

            if not self.df.empty:
                self.trainer.fit_eval(self.df, self.feature_names, self.target)
        else:
            self.df = pd.DataFrame()
            self._load_model(folder, config)

        if not self.df.empty:
            self.save_model(folder, "-1")
            self.save_data(folder, empty=True)


    def update_model(self, df: pd.DataFrame, folder, n, save_data=True):
        if self.fit_count > 0:
            self.df = pd.read_csv(f"{folder}All_data.csv", index_col=0)
            print(f"Read Data after {self.fit_count} iterations")
            value_counts = self.df["iteration"].value_counts().to_dict()
            print("All Data Size =", len(self.df))

            # Drop iterations with too few samples
            self.df = self.df[self.df["iteration"].map(value_counts.get) > 20]
        else:
            self.df = pd.DataFrame()

        self.df = pd.concat([self.df, df])
        print("New Data Size =", len(self.df))

        data, stats = self.trainer.fit_eval(self.df, self.feature_names, self.target)
        self.model_stats = pd.concat([self.model_stats, pd.DataFrame([stats])])
        if save_data:
            self.save_data(data, folder)
        self.fit_count += 1
        self.save_model(folder, n)
        self.df = pd.DataFrame()

        pd.DataFrame(self.state_counter).to_csv(f"{folder}state_counter.csv")
        
        # inside class SupervisedNN
    def feature_importance(
        self,
        df: pd.DataFrame,
        *,
        repeats: int = 5,
        metric: str = "accuracy",
        exclude_prefixes: tuple[str, ...] = ("mask_",),
        batch_size: int = 4096,
        random_state: int = 123,
        save_to: str  = None
    ) -> pd.DataFrame:
        """
        Convenience wrapper to compute & (optionally) save permutation importance.
        """
        imp_df = self.trainer.permutation_importance(
            df=df,
            feature_names=self.feature_names,
            target_col=self.target,
            repeats=repeats,
            metric=metric,
            exclude_prefixes=exclude_prefixes,
            batch_size=batch_size,
            random_state=random_state,
        )
        self.feature_importances_ = imp_df
        if save_to:
            os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
            imp_df.to_csv(save_to, index=False)
        return imp_df
    
    def get_action(self, x:np.ndarray):
        x_np = np.asarray(x, dtype=np.float32)
        probs = self.trainer.predict_proba(x_np)
        return self.choose_action(probs), probs

    def max_prob_action(self, prob):
        return np.argmax(prob)

    def random_action(self, prob):
        return self.chooser.choice(range(self.num_class), p=prob)
    
    def save_data(self, data, folder):
        data.reset_index(drop=True).to_csv(f"{folder}All_data.csv")
        self.model_stats.to_csv(f"{folder}Model_Stats.csv")

    def save_model(self, folder, n):
        torch.save(self.model.state_dict(), f"{folder}NN_{n}.pth")
        self.trainer.train_losses.to_csv(f"{folder}NN_Losses.csv")
        print(f"Model Saved - {n} at {folder} with name NN_{n}.pth.")

    def _load_model(self, folder, config):
        n = config["model"]
        if config["adapter"] == "per_phy":
            # dims are fixed by PerPhyAdapter (3 ctx, 3 per-phys). change here if your schema changes.
            self.model = SharedPhysNet(p_dim=2, c_dim=3, hidden=config.get("hidden", 64))
        elif config["adapter"] == "flat":
            self.model = MLPClassifier(input_dim=len(self.feature_names), num_classes=self.num_class,
                                       hidden=config.get("hidden", 128),
                                       depth=config.get("depth", 2))
        print("Reading Model")
        self.model.load_state_dict(torch.load(f"{folder}NN_{n}.pth"))
        self.trainer = GenericTrainer(
                                        adapter=self.adapter,
                                        model=self.model,
                                        num_actions= config.get("num_class", 1),
                                        lr=config.get("lr", 1e-3),
                                        batch_size=config.get("batch_size", 256),
                                        epochs=config.get("epochs", 50),
                                        use_weight=config.get("use_weight", False),
                                    )
        self.fit_count = n + 1