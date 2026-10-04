# Command-line scripts named test_*.py, not pytest tests (run them directly,
# see their docstrings): test_max_degradation.py parses its arguments on
# import, test_weather_single_model.py's test function takes a config.
collect_ignore = ["test_max_degradation.py", "test_weather_single_model.py"]
