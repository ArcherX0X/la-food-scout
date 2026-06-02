# data/

Contains the raw dataset collected from the Google Places API.

## Files

| File | Description |
|---|---|
| `la_restaurants.csv` | 2,443 restaurants across 20 LA neighborhoods, collected May 2026 |

## Schema

| Column | Type | Description |
|---|---|---|
| `name` | string | Restaurant name |
| `rating` | float | Google Places rating (1.0–5.0) |
| `review_count` | float | Number of user reviews |
| `price_level` | float | Price tier: 1=Budget, 2=Moderate, 3=Expensive, 4=Very Expensive |
| `primary_type` | string | Google Places category (e.g. `korean_restaurant`, `cafe`) |
| `business_status` | string | `OPERATIONAL`, `CLOSED_TEMPORARILY`, etc. |
| `lat` | float | Latitude |
| `lon` | float | Longitude |
| `neighborhood` | string | Search neighborhood (one of 20 LA areas) |
| `search_type` | string | Place type used in the search query |

## Collection method

See `scripts/collect.py`. Requires a `GOOGLE_PLACES_API_KEY` in `.env`.

Each neighborhood was searched from 5 grid points (center + N/S/E/W offsets) across 4 place types (restaurant, cafe, bakery, bar) to maximize coverage and avoid the 20-result-per-call cap.
