# models/

Trained model artifacts.

## Files

| File | Description |
|---|---|
| `model.pkl` | Serialized sklearn Pipeline (preprocessor + Random Forest classifier) |

## Bundle contents

`model.pkl` is a joblib-serialized dict with keys:

| Key | Type | Description |
|---|---|---|
| `pipeline` | `sklearn.Pipeline` | Full preprocessing + classifier pipeline |
| `numeric_features` | list | `['price_level', 'review_count_log']` |
| `categorical_features` | list | `['neighborhood', 'primary_type']` |
| `target` | str | `'high_rated'` |
| `threshold` | float | `4.5` (rating cutoff for high-rated label) |
| `neighborhoods` | list | All valid neighborhood values |
| `place_types` | list | All valid `primary_type` values |

## Reproduce

```bash
python scripts/train_model.py
```
