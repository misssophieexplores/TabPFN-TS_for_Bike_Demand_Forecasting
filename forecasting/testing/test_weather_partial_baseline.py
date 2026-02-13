import sys
sys.path.insert(0, '..')

from config import ForecastConfig
from models.statistical import SeasonalNaiveForecaster
from models.ml_models import XGBoostForecaster
from run_experiments import ForecastingExperiment, load_and_prepare_data

config = ForecastConfig()
config.n_folds = 5  # Quick test
config.horizons = [24, 168]  # Only 2 horizons

df, dataset_name = load_and_prepare_data("../../data/SeoulBikeData.csv")
config.dataset_name = dataset_name

models = [
    SeasonalNaiveForecaster(seasonal_period=24),
    XGBoostForecaster(n_lags=24)
]

scenarios = ["clean_only", "degraded"]

exp = ForecastingExperiment(config, "weather_partial_test")
results = exp.run_all_experiments(models, df, scenarios, verbose=True)
exp.save_results(results)
exp.finish()

print("\n✓ LEVEL 3 PASSED")