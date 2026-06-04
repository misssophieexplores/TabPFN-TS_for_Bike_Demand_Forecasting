"""
Forecasting evaluation metrics.
"""
import numpy as np
import warnings
from typing import Dict, Optional
from pathlib import Path
import pandas as pd

class MetricsCalculator:
    """Calculate forecasting performance metrics"""
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error.
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        float
            MAE value
        """
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error.
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        float
            RMSE value
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def mase(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_train: np.ndarray, 
        seasonal_period: int = 24
    ) -> float:
        """
        Mean Absolute Scaled Error.
        
        Scales forecast error by naive seasonal forecast error on training set.
        Values < 1 indicate better than naive seasonal forecast.
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
        y_train : np.ndarray
            Training set values (for scaling)
        seasonal_period : int
            Seasonal period (24 for hourly data with daily seasonality)
            
        Returns:
        --------
        float
            MASE value
        """
        # Calculate naive seasonal forecast error on training set
        naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
        mae_naive = np.mean(naive_errors)
        
        # Calculate forecast error
        mae_forecast = np.mean(np.abs(y_true - y_pred))
        
        # Return scaled error
        if mae_naive == 0:
            return np.inf
        return float(mae_forecast / mae_naive)
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error.
        
        Handles zero values better than MAPE.
        Returns percentage in range [0, 100].
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        float
            sMAPE value (percentage)
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        smape_values = np.where(
            mask,
            np.abs(y_true - y_pred) / denominator,
            0
        )
        
        return float(100 * np.mean(smape_values))

    @staticmethod
    def win_rate(
        errors_j: np.ndarray,
        errors_k: np.ndarray,
    ) -> float:
        """
        Pairwise Win Rate of model j against model k across tasks.

        Represents the fraction of tasks where model j achieves lower error
        than model k. Ties count as half-wins for each model.

        Equation (6) from fev-bench (Shchur et al., 2026):
            W_jk = (1/R) * sum_r [ 1(E_rj < E_rk) + 0.5 * 1(E_rj == E_rk) ]

        Parameters:
        -----------
        errors_j : np.ndarray, shape (R,)
            Per-task errors for model j (e.g. MASE averaged over eval windows).
        errors_k : np.ndarray, shape (R,)
            Per-task errors for model k.

        Returns:
        --------
        float
            Win rate in [0, 1]. Values above 0.5 indicate model j outperforms
            model k on the majority of tasks.
        """
        if errors_j.shape != errors_k.shape:
            raise ValueError(
                f"errors_j and errors_k must have the same shape, "
                f"got {errors_j.shape} vs {errors_k.shape}"
            )
        wins = np.sum(errors_j < errors_k) + 0.5 * np.sum(errors_j == errors_k)
        return float(wins / len(errors_j))

    @staticmethod
    def skill_score(
        errors_j: np.ndarray,
        errors_baseline: np.ndarray,
        clip_lower: float = 1e-2,
        clip_upper: float = 1e2,
    ) -> float:
        """
        Pairwise Skill Score of model j relative to a baseline model.

        Quantifies the average error reduction of model j over the baseline
        using a geometrically-aggregated relative error, clipped to avoid
        distortion from extreme values.

        Equation (7) from fev-bench (Shchur et al., 2026):
            S_j = 1 - geomean( clip(E_rj / E_r_baseline; lower, upper) )

        A score of 0 means parity with the baseline; positive values indicate
        improvement; negative values indicate underperformance.

        Parameters:
        -----------
        errors_j : np.ndarray, shape (R,)
            Per-task errors for model j.
        errors_baseline : np.ndarray, shape (R,)
            Per-task errors for the baseline model (e.g. Seasonal Naive).
        clip_lower : float
            Lower clipping bound for relative errors (default: 1e-2).
        clip_upper : float
            Upper clipping bound for relative errors (default: 1e2).

        Returns:
        --------
        float
            Skill score. Positive = better than baseline, negative = worse.
        """
        if errors_j.shape != errors_baseline.shape:
            raise ValueError(
                f"errors_j and errors_baseline must have the same shape, "
                f"got {errors_j.shape} vs {errors_baseline.shape}"
            )

        # Zero baseline errors are real (e.g. Seasonal Naive predicts a
        # zero-demand task perfectly). Rather than dropping these tasks
        # (which would bias the score by removing the baseline's strongest
        # cases), handle them at the limits the clip bounds already encode:
        #   baseline == 0, model  > 0  -> ratio +inf -> clip_upper (model maximally worse)
        #   baseline == 0, model == 0  -> ratio 1.0 (both perfect -> parity)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_errors = errors_j / errors_baseline
        relative_errors = np.where(
            (errors_baseline == 0) & (errors_j == 0), 1.0, relative_errors
        )
        clipped = np.clip(relative_errors, clip_lower, clip_upper)
        geom_mean = float(np.exp(np.mean(np.log(clipped))))
        return float(1.0 - geom_mean)

    @staticmethod
    def win_rate_with_ci(
        errors_j: np.ndarray,
        errors_k: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Pairwise Win Rate with bootstrapped confidence interval.

        Uses paired bootstrap over tasks (Efron, 1992) as described in
        Section 4.3 of fev-bench (Shchur et al., 2026). Draws R tasks with
        replacement B times to estimate the sampling distribution of W_jk.

        Parameters:
        -----------
        errors_j : np.ndarray, shape (R,)
            Per-task errors for model j.
        errors_k : np.ndarray, shape (R,)
            Per-task errors for model k.
        n_bootstrap : int
            Number of bootstrap samples (default: 1000).
        confidence : float
            Confidence level for the interval (default: 0.95).
        seed : int, optional
            Random seed for reproducibility.

        Returns:
        --------
        Dict[str, float]
            {
              'win_rate': point estimate,
              'ci_lower': lower bound,
              'ci_upper': upper bound,
            }
        """
        rng = np.random.default_rng(seed)
        R = len(errors_j)
        point_estimate = MetricsCalculator.win_rate(errors_j, errors_k)

        bootstrap_stats = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, R, size=R)
            bootstrap_stats.append(
                MetricsCalculator.win_rate(errors_j[idx], errors_k[idx])
            )

        alpha = 1.0 - confidence
        ci_lower = float(np.quantile(bootstrap_stats, alpha / 2))
        ci_upper = float(np.quantile(bootstrap_stats, 1 - alpha / 2))

        return {
            'win_rate': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

    @staticmethod
    def skill_score_with_ci(
        errors_j: np.ndarray,
        errors_baseline: np.ndarray,
        clip_lower: float = 1e-2,
        clip_upper: float = 1e2,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Pairwise Skill Score with bootstrapped confidence interval.

        Uses paired bootstrap over tasks (Efron, 1992) as described in
        Section 4.3 of fev-bench (Shchur et al., 2026).

        Parameters:
        -----------
        errors_j : np.ndarray, shape (R,)
            Per-task errors for model j.
        errors_baseline : np.ndarray, shape (R,)
            Per-task errors for the baseline model.
        clip_lower : float
            Lower clipping bound for relative errors (default: 1e-2).
        clip_upper : float
            Upper clipping bound for relative errors (default: 1e2).
        n_bootstrap : int
            Number of bootstrap samples (default: 1000).
        confidence : float
            Confidence level for the interval (default: 0.95).
        seed : int, optional
            Random seed for reproducibility.

        Returns:
        --------
        Dict[str, float]
            {
              'skill_score': point estimate,
              'ci_lower': lower bound,
              'ci_upper': upper bound,
            }
        """
        rng = np.random.default_rng(seed)
        R = len(errors_j)
        point_estimate = MetricsCalculator.skill_score(
            errors_j, errors_baseline, clip_lower, clip_upper
        )

        bootstrap_stats = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, R, size=R)
            bootstrap_stats.append(
                MetricsCalculator.skill_score(
                    errors_j[idx], errors_baseline[idx], clip_lower, clip_upper
                )
            )

        alpha = 1.0 - confidence
        ci_lower = float(np.quantile(bootstrap_stats, alpha / 2))
        ci_upper = float(np.quantile(bootstrap_stats, 1 - alpha / 2))

        return {
            'skill_score': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

    @classmethod
    def calculate_all(
        cls, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all per-series metrics at once.

        Note: win_rate and skill_score operate across tasks (error vectors),
        not on individual series. Use them separately after collecting
        per-task errors across models.
        
        Parameters:
        -----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
        y_train : np.ndarray
            Training set values (for MASE calculation)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with all metric values
        """
        return {
            'MAE': cls.mae(y_true, y_pred),
            'RMSE': cls.rmse(y_true, y_pred),
            'MASE': cls.mase(y_true, y_pred, y_train),
            'sMAPE': cls.smape(y_true, y_pred)
        }

    @staticmethod
    def compute_comparative_metrics(
        results_df,
        baseline_model: str = "Seasonal_Naive",
        error_column: str = "MASE_mean",
        task_cols=("dataset", "horizon", "weather_scenario"),
        seed: int = 42,
        error_log_path=None,
    ):
        import pandas as pd

        task_cols = list(task_cols)

        if baseline_model not in results_df['model'].values:
            raise ValueError(
                f"Baseline model '{baseline_model}' not found in results_df. "
                f"Available models: {sorted(results_df['model'].unique().tolist())}"
            )

        # One error per (task, model). A task is (dataset, horizon,
        # weather_scenario): the aggregated grain, with folds already
        # summarized into MASE_mean. Bootstrap resamples these tasks.
        # The aggregated file has exactly one row per (task, model), so
        # aggfunc='mean' is just a guard and never pools across scenarios.
        pivot = results_df.pivot_table(
            index=task_cols,
            columns='model',
            values=error_column,
            aggfunc='mean',
        )

        if baseline_model not in pivot.columns:
            raise ValueError(f"Baseline '{baseline_model}' missing after pivot.")

        # Fail loudly if the baseline is missing any cell within the
        # scenario(s) it was actually run in. (It is legitimately NaN in
        # scenarios it never ran, e.g. 'degraded', so restrict the check to
        # the scenarios where the baseline has at least one result.)
        if "weather_scenario" in task_cols:
            scen_level = task_cols.index("weather_scenario")
            baseline_col = pivot[baseline_model]
            baseline_scenarios = {
                idx[scen_level]
                for idx in baseline_col.dropna().index
            }
            in_baseline_scen = baseline_col.index.get_level_values(
                scen_level
            ).isin(baseline_scenarios)
            missing = baseline_col[in_baseline_scen & baseline_col.isna()]
            if len(missing) > 0:
                raise ValueError(
                    f"Baseline '{baseline_model}' is missing "
                    f"{len(missing)} cell(s) within its own scenario(s) "
                    f"{sorted(baseline_scenarios)}. These comparisons would "
                    f"silently drop. Missing tasks:\n{missing.index.tolist()}"
                )

        baseline_errors = pivot[baseline_model].values
        rows = []

        for model_name in pivot.columns:
            if model_name == baseline_model:
                continue

            model_errors = pivot[model_name].values
            # Both non-NaN -> compared only on shared tasks. Because the
            # baseline ran clean_only, its column is NaN on degraded tasks,
            # so every baseline comparison is automatically clean-weather only.
            valid = ~(np.isnan(model_errors) | np.isnan(baseline_errors))

            if valid.sum() < 2:
                msg = (
                    f"[WARN] Skipping {model_name} vs {baseline_model}: "
                    f"only {valid.sum()} shared tasks, need at least 2."
                )
                if error_log_path is not None:
                    with open(error_log_path, "a") as f:
                        f.write(f"\n{msg}\n")
                continue

            errs_j = model_errors[valid]
            errs_b = baseline_errors[valid]

            wr = MetricsCalculator.win_rate_with_ci(errs_j, errs_b, seed=seed)
            ss = MetricsCalculator.skill_score_with_ci(errs_j, errs_b, seed=seed)

            rows.append({
                'model': model_name,
                'baseline': baseline_model,
                'n_tasks': int(valid.sum()),
                'win_rate': wr['win_rate'],
                'win_rate_ci_lower': wr['ci_lower'],
                'win_rate_ci_upper': wr['ci_upper'],
                'skill_score': ss['skill_score'],
                'skill_score_ci_lower': ss['ci_lower'],
                'skill_score_ci_upper': ss['ci_upper'],
            })

        return pd.DataFrame(rows)

    @staticmethod
    def compute_and_save_comparative_metrics(
        results_csv_path,
        output_dir,
        version,
        baseline_model="Seasonal_Naive",
        error_column="MASE_mean",
    ):
        import pandas as pd
        if isinstance(results_csv_path, pd.DataFrame):
            df = results_csv_path
        else:
            df = pd.read_csv(results_csv_path)
        # Do NOT group by dataset: that would shrink each comparison to
        # (horizon x scenario) tasks per dataset. Dataset is a task key,
        # so all datasets are pooled into one comparison.
        combined = MetricsCalculator.compute_comparative_metrics(
            df, baseline_model=baseline_model, error_column=error_column
        )
        out = Path(output_dir) / f"comparative_metrics_{version}.csv"
        combined.to_csv(out, index=False)
        return combined

## NOT USED YET
@staticmethod
def compute_pairwise_win_rate_matrix(
    results_df: pd.DataFrame,
    error_column: str = "MASE_mean",
    task_keys: tuple = ("horizon", "weather_scenario"),
) -> pd.DataFrame:
    """
    Full pairwise win-rate matrix W where W[j, k] is the fraction of
    tasks on which model j beats model k (half-credit for ties).

    Diagonal is set to 0.5 (tie vs. self). Off-diagonal cells use only
    tasks where both models j and k have a non-NaN error.

    Parameters
    ----------
    results_df : pd.DataFrame
        Aggregated results, one row per (model, *task_keys).
    error_column : str
        Column to use as the per-task error (default: 'MASE_mean').
    task_keys : tuple of str
        Columns that jointly identify a task.

    Returns
    -------
    pd.DataFrame
        Square DataFrame indexed and columned by model name, values in [0, 1].
    """
    pivot = results_df.pivot_table(
        index=list(task_keys),
        columns="model",
        values=error_column,
    )
    models = pivot.columns.tolist()
    mat = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)

    for j in models:
        for k in models:
            if j == k:
                mat.loc[j, k] = 0.5
                continue
            ej = pivot[j]
            ek = pivot[k]
            valid = ej.notna() & ek.notna()
            if valid.sum() < 1:
                continue
            mat.loc[j, k] = MetricsCalculator.win_rate(
                ej[valid].values, ek[valid].values
            )
    return mat