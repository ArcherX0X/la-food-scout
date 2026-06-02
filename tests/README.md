# tests/

Automated API tests using pytest and FastAPI's `TestClient`.

## Files

| File | Description |
|---|---|
| `test_api.py` | 7 tests covering all endpoints and validation edge cases |

## Run

```bash
uv pip install pytest httpx
pytest tests/ -v
```

## What's tested

| Test | What it checks |
|---|---|
| `test_health` | `/health` returns 200 |
| `test_ready` | `/ready` returns 200 with model loaded |
| `test_meta_returns_lists` | `/meta` returns non-empty neighborhood and type lists |
| `test_predict_valid` | `/predict` returns a probability between 0 and 1 |
| `test_predict_missing_field` | Missing required field returns 422 |
| `test_predict_invalid_price_level` | Price level > 4 returns 422 |
| `test_predict_negative_reviews` | Negative review count returns 422 |
