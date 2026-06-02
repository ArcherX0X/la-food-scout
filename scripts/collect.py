import os
import pathlib
import time
import requests
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")
API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# 20 neighborhoods with 4 search points each (center + N/S/E/W offsets)
# covers more of each area and avoids the 20-result cap per call
OFFSET = 0.012  # ~1.3km

LA_NEIGHBORHOODS = [
    {"name": "Koreatown",       "lat": 34.0615, "lon": -118.3033},
    {"name": "Westwood",        "lat": 34.0623, "lon": -118.4454},
    {"name": "Silver Lake",     "lat": 34.0875, "lon": -118.2700},
    {"name": "Santa Monica",    "lat": 34.0195, "lon": -118.4912},
    {"name": "Downtown LA",     "lat": 34.0407, "lon": -118.2468},
    {"name": "Venice",          "lat": 33.9850, "lon": -118.4695},
    {"name": "Hollywood",       "lat": 34.0928, "lon": -118.3287},
    {"name": "Culver City",     "lat": 34.0211, "lon": -118.3965},
    {"name": "Pasadena",        "lat": 34.1478, "lon": -118.1445},
    {"name": "Mid-Wilshire",    "lat": 34.0622, "lon": -118.3406},
    {"name": "Echo Park",       "lat": 34.0782, "lon": -118.2606},
    {"name": "Los Feliz",       "lat": 34.1076, "lon": -118.2898},
    {"name": "Fairfax",         "lat": 34.0775, "lon": -118.3614},
    {"name": "Studio City",     "lat": 34.1397, "lon": -118.3870},
    {"name": "Chinatown",       "lat": 34.0637, "lon": -118.2370},
    {"name": "Little Tokyo",    "lat": 34.0489, "lon": -118.2390},
    {"name": "Brentwood",       "lat": 34.0496, "lon": -118.4759},
    {"name": "Monterey Park",   "lat": 34.0625, "lon": -118.1228},
    {"name": "Long Beach",      "lat": 33.7701, "lon": -118.1937},
    {"name": "Burbank",         "lat": 34.1808, "lon": -118.3090},
]

PLACE_TYPES = ["restaurant", "cafe", "bakery", "bar"]

PRICE_MAP = {
    "PRICE_LEVEL_FREE":           0,
    "PRICE_LEVEL_INEXPENSIVE":    1,
    "PRICE_LEVEL_MODERATE":       2,
    "PRICE_LEVEL_EXPENSIVE":      3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}


def search_nearby(lat, lon, place_type, radius=1500):
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.rating,places.userRatingCount,"
            "places.priceLevel,places.primaryType,places.location,places.businessStatus"
        ),
    }
    body = {
        "includedTypes": [place_type],
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": radius,
            }
        },
    }
    r = requests.post(url, headers=headers, json=body)
    if r.status_code != 200:
        return []
    return r.json().get("places", [])


def sub_points(lat, lon):
    """Return center + 4 offset points to cover more of a neighborhood."""
    return [
        (lat, lon),
        (lat + OFFSET, lon),
        (lat - OFFSET, lon),
        (lat, lon + OFFSET),
        (lat, lon - OFFSET),
    ]


def parse_place(p, neighborhood, search_type):
    return {
        "name":            p.get("displayName", {}).get("text", ""),
        "rating":          p.get("rating"),
        "review_count":    p.get("userRatingCount"),
        "price_level":     PRICE_MAP.get(p.get("priceLevel", ""), None),
        "primary_type":    p.get("primaryType", ""),
        "business_status": p.get("businessStatus", ""),
        "lat":             p.get("location", {}).get("latitude"),
        "lon":             p.get("location", {}).get("longitude"),
        "neighborhood":    neighborhood,
        "search_type":     search_type,
    }


def main():
    total_requests = len(LA_NEIGHBORHOODS) * len(sub_points(0, 0)) * len(PLACE_TYPES)
    est_cost = total_requests * 0.032
    print(f"Planned: {total_requests} API requests — estimated cost: ${est_cost:.2f}")
    print(f"(Safe: $300 free credit available)\n")

    results = []
    seen_ids = set()

    for hood in LA_NEIGHBORHOODS:
        for lat, lon in sub_points(hood["lat"], hood["lon"]):
            for ptype in PLACE_TYPES:
                places = search_nearby(lat, lon, ptype)
                for p in places:
                    pid = p.get("id", "")
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    results.append(parse_place(p, hood["name"], ptype))
                time.sleep(0.1)

        print(f"  {hood['name']}: {len(results)} unique places so far")

    df = pd.DataFrame(results)
    df = df.dropna(subset=["rating"])
    out_path = REPO_ROOT / "data" / "la_restaurants.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nDone — saved {len(df)} rated restaurants to {out_path}")
    print(df["rating"].describe().round(2))
    print("\nNeighborhood counts:")
    print(df["neighborhood"].value_counts().to_string())


if __name__ == "__main__":
    main()
