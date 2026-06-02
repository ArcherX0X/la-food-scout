import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="LA Food Scout", page_icon="🍜", layout="centered")

st.title("🍜 LA Food Scout")
st.caption("Will this restaurant be highly rated? Find out before you go.")


@st.cache_data(ttl=300)
def fetch_meta() -> dict:
    try:
        r = requests.get(f"{API_URL}/meta", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"neighborhoods": [], "place_types": []}


meta = fetch_meta()

if not meta["neighborhoods"]:
    st.error("Could not connect to the prediction API. Make sure it is running.")
    st.stop()

st.subheader("Restaurant Details")

col1, col2 = st.columns(2)

with col1:
    neighborhood = st.selectbox("Neighborhood", meta["neighborhoods"])
    price_level = st.selectbox(
        "Price Level",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "$ Budget", 2: "$$ Moderate", 3: "$$$ Expensive", 4: "$$$$ Very Expensive"}[x],
    )

with col2:
    place_type = st.selectbox("Place Type", meta["place_types"])
    review_count = st.number_input("Number of Reviews", min_value=0, value=500, step=50)

if st.button("Predict Rating", type="primary", use_container_width=True):
    payload = {
        "neighborhood": neighborhood,
        "primary_type": place_type,
        "price_level": price_level,
        "review_count": review_count,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()

        prob = result["high_rated_probability"]
        is_high = result["predicted_high_rated"]

        st.divider()
        if is_high:
            st.success(f"**Likely High-Rated ⭐** — {prob*100:.0f}% confidence")
        else:
            st.warning(f"**Probably Not High-Rated** — {prob*100:.0f}% chance of ≥4.5★")

        st.progress(prob, text=f"Probability of ≥4.5★ rating: {prob*100:.1f}%")

    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Is it running?")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption(
    "Model: Gradient Boosting Classifier trained on 2,400+ LA restaurants "
    "from the Google Places API. Features: neighborhood, place type, price level, review count."
)
