# Open Meteo:
@software{Zippenfenig_Open-Meteo,
  author = {Zippenfenig, Patrick},
  doi = {10.5281/zenodo.7970649},
  licence = {CC-BY-4.0},
  title = {Open-Meteo.com Weather API},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International},
  url = {https://open-meteo.com/}
}

## Missing Data: 
@software{VisualCrossing_Weather_API,
  author = {{Visual Crossing Corporation}},
  title = {Visual Crossing Weather API},
  year = {2026},
  url = {https://www.visualcrossing.com/},
  note = {Weather data for 2015-2017}
}

## KMA ASOS hourly station observations

- **Source:** Korea Meteorological Administration (KMA), Automated Synoptic Observing System (ASOS)
- **Planned station:** Seoul 108
- **Official service:** https://www.data.go.kr/data/15139432/openapi.do
- **Cost:** Free
- **Licence:** Korea Open Government License, Type 1 (attribution required)
- **Official licence text:** `공공저작물 : 출처표시 (제 1유형)`
- **Academic/publication use:** Permitted with source attribution.

The official data.go.kr catalogue describes the ASOS service as free and lists the permitted-use scope as Korea Open Government License Type 1, which requires attribution.

## Copernicus Climate Change Service / ERA5 and ERA5-Land

- **Source:** Copernicus Climate Change Service (C3S), operated by ECMWF
- **Official licence:** https://apps.ecmwf.int/datasets/licences/copernicus/
- **Licence:** Free of charge, worldwide, non-exclusive, royalty-free and perpetual; attribution required.
- **Official licence text:** `Access to Copernicus Products is given for any purpose in so far as it is lawful`, including reproduction, distribution, communication to the public, adaptation, modification and combination with other data.
- **Academic/publication use:** Permitted. Publications using modified/adapted Copernicus products must provide the required Copernicus attribution and disclaimer.

Suggested attribution from the licence for modified products:
`Contains modified Copernicus Climate Change Service information [Year].`

The licence also requires a statement that neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data.

## NOAA / NCEI Integrated Surface Database (ISD)

- **Source:** NOAA National Centers for Environmental Information (NCEI)
- **Dataset used:** Integrated Surface Database (ISD) station observations
- **Official open-data policy:** https://www.ncei.noaa.gov/archive
- **Policy document:** https://www.ncei.noaa.gov/sites/default/files/2023-12/NCEI%20PD-10-2-02%20-%20Open%20Data%20Policy%20Signed.pdf
- **Licence/status:** NOAA/federal environmental data are in the public domain in the United States unless explicitly exempt; NCEI works to apply CC0 for international users.
- **Official policy text:** `Environmental data and information produced by NOAA or any Federal agency are available fully and openly to data users` and `These data are in the public domain in the United States.`
- **Academic/publication use:** Permitted.

## v7 weather columns (2 Oct 2026)

- **Precipitation (all cities):** total precipitation including melted snow, mm per hour.
  - Seoul `precipitation_mm` = `Rainfall` (KMA ASOS 강수량, station 108). KMA provides it as 3-hour totals at 00, 03, …, 21 h from November to March and hourly from April to October. Each winter total is spread evenly over its three hours (the value at hour t covers the hours labelled t−2, t−1 and t); totals are kept. `Rainfall` itself is unchanged.
  - London/Washington `precipitation_mm` = `rainfall_mm` + `snowfall_cm` / 0.7 (Open-Meteo `rain` and `snowfall`; 0.7 cm of snow per mm of water). This equals Open-Meteo's `precipitation` (checked: identical within 0.1 mm in 100 % of hours of the download).
- **Snow depth (all cities):** snow lying on the ground, cm.
  - Seoul `snow_depth_cm` = `Snowfall` (KMA ASOS 적설: snow depth, despite the column name; KMA has no hourly snowfall field). `Snowfall` itself is unchanged.
  - London/Washington `snow_depth_cm` = Open-Meteo Historical Weather API `snow_depth` (`best_match` = ERA5-Land; metres × 100; 1 cm steps), downloaded 2 Oct 2026 for 51.479, −0.449 (Heathrow; grid cell 51.4938, −0.4891) and 38.8483, −77.0342 (Reagan National; grid cell 38.8401, −77.0902). Files: `open_meteo/open-meteo-51.49N0.49W24m.csv`, `open_meteo/open-meteo-38.84N77.09W3m.csv`.
- **Time alignment (London/Washington):** the Open-Meteo columns (temperature, humidity, wind, dew point, solar radiation, rain, snowfall) were stored on standard time all year (London UTC+0, Washington UTC−5), while the bike counts follow the local clock with daylight saving. They were re-aligned to the clock on 2 Oct 2026 (during daylight saving each value moves down one row; outside it nothing changes). Visibility (Visual Crossing) already followed the clock. London `t2` and `weather_code` (original Kaggle columns, not used) were not changed.
- **Column names:** all three cities use `precipitation_mm` and `snow_depth_cm`.
- **No longer used:** London/Washington `rainfall_mm`, `snowfall_cm` (kept for provenance; re-aligned as well); Seoul `Rainfall`, `Snowfall` (original UCI columns).
- **Script:** `build_weather_v7.py` (provenance record; not part of the pipeline).
