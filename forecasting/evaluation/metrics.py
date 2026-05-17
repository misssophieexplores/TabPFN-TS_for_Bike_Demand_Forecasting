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
        if np.any(errors_baseline == 0):
            raise ValueError(
                "errors_baseline contains zeros; cannot compute relative error."
            )

        relative_errors = errors_j / errors_baseline
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
        seed: int = 42,
        error_log_path = None,
    ):
        """
        Compute win_rate and skill_score for each model vs. a baseline,
        with bootstrapped 95% CIs. Operates on MASE_mean as the per-task error.

        A "task" is a unique (horizon, weather_scenario) combination. Only tasks
        where both the candidate model and the baseline have results are used.

        Parameters:
        -----------
        results_df : pd.DataFrame
            Aggregated results with one row per (model, horizon, weather_scenario),
            as produced by ForecastingExperiment.run_all_experiments().
        baseline_model : str
            Model name to use as reference (default: 'SeasonalNaive').
        seed : int
            Random seed for bootstrap reproducibility (default: 42).

        Returns:
        --------
        pd.DataFrame
            One row per non-baseline model with columns:
            model, baseline, n_tasks,
            win_rate, win_rate_ci_lower, win_rate_ci_upper,
            skill_score, skill_score_ci_lower, skill_score_ci_upper.
        """
        import pandas as pd

        if baseline_model not in results_df['model'].values:
            raise ValueError(
                f"Baseline model '{baseline_model}' not found in results_df. "
                f"Available models: {sorted(results_df['model'].unique().tolist())}"
            )

        # Pivot to (horizon, weather_scenario) x model matrix of MASE_mean values
        pivot = results_df.pivot_table(
            index=['horizon', 'weather_scenario'],
            columns='model',
            values='MASE_mean',
        )

        baseline_errors = pivot[baseline_model].values
        rows = []

        for model_name in pivot.columns:
            if model_name == baseline_model:
                continue

            model_errors = pivot[model_name].values

            # Only compare on tasks where both model and baseline have a result
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
    def compute_and_save_comparative_metrics(results_csv_path, output_dir, version, baseline_model="Seasonal_Naive"):
        import pandas as pd
        df = pd.read_csv(results_csv_path)
        results = []
        for dataset, group in df.groupby('dataset'):
            comp = MetricsCalculator.compute_comparative_metrics(group, baseline_model=baseline_model)
            comp['dataset'] = dataset
            results.append(comp)
        combined = pd.concat(results, ignore_index=True)
        out = Path(output_dir) / f"comparative_metrics_{version}.csv"
        combined.to_csv(out, index=False)
        return combined