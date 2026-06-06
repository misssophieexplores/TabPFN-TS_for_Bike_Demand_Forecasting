import numpy as np
import timesfm

from dotenv import load_dotenv
load_dotenv()

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
)
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=168,
    )
)

point_forecast, quantile_forecast = model.forecast(
    horizon=168,
    inputs=[
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
    ],
)

print("point_forecast shape:", point_forecast.shape)
print("quantile_forecast shape:", quantile_forecast.shape)