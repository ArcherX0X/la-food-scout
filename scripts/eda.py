import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

REPO_ROOT = pathlib.Path(__file__).parent.parent
df = pd.read_csv(REPO_ROOT / "data" / "la_restaurants.csv")
df["high_rated"] = (df["rating"] >= 4.5).astype(int)

fig = plt.figure(figsize=(16, 10))
fig.suptitle("LA Food Scout — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Rating distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df["rating"], bins=20, color="#FF6B6B", edgecolor="white")
ax1.axvline(4.5, color="black", linestyle="--", linewidth=1.5, label="4.5 threshold")
ax1.set_title("Rating Distribution")
ax1.set_xlabel("Rating")
ax1.set_ylabel("Count")
ax1.legend()

# 2. Price level vs avg rating
ax2 = fig.add_subplot(gs[0, 1])
price_labels = {1: "Budget", 2: "Moderate", 3: "Expensive", 4: "Very Exp."}
price_avg = df.dropna(subset=["price_level"]).groupby("price_level")["rating"].mean()
x_labels = [price_labels.get(int(k), str(k)) for k in price_avg.index]
bars = ax2.bar(
    range(len(x_labels)),
    price_avg.values,
    color=["#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"][:len(x_labels)]
)
ax2.set_xticks(range(len(x_labels)))
ax2.set_xticklabels(x_labels)
ax2.set_title("Avg Rating by Price Level")
ax2.set_xlabel("Price Level")
ax2.set_ylabel("Avg Rating")
ax2.set_ylim(4.0, 4.8)
for bar, val in zip(bars, price_avg.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.2f}", ha="center", va="bottom", fontsize=9)

# 3. Review count vs rating
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(np.log1p(df["review_count"]), df["rating"],
            alpha=0.3, color="#A29BFE", s=20)
ax3.set_title("Log(Review Count) vs Rating")
ax3.set_xlabel("log(Review Count + 1)")
ax3.set_ylabel("Rating")

# 4. Neighborhood avg rating
ax4 = fig.add_subplot(gs[1, 0])
hood_avg = df.groupby("neighborhood")["rating"].mean().sort_values(ascending=True)
ax4.barh(hood_avg.index, hood_avg.values, color="#FD79A8")
ax4.set_title("Avg Rating by Neighborhood")
ax4.set_xlabel("Avg Rating")
ax4.set_xlim(4.0, 4.7)
for i, (hood, val) in enumerate(hood_avg.items()):
    ax4.text(val + 0.005, i, f"{val:.2f}", va="center", fontsize=8)

# 5. Top cuisine types
ax5 = fig.add_subplot(gs[1, 1])
top_types = df["primary_type"].value_counts().head(8)
ax5.barh(top_types.index[::-1], top_types.values[::-1], color="#55EFC4")
ax5.set_title("Top 8 Place Types")
ax5.set_xlabel("Count")

# 6. High-rated % by neighborhood
ax6 = fig.add_subplot(gs[1, 2])
high_pct = df.groupby("neighborhood")["high_rated"].mean().sort_values(ascending=True) * 100
ax6.barh(high_pct.index, high_pct.values, color="#FDCB6E")
ax6.set_title("% High-Rated (≥4.5★) by Neighborhood")
ax6.set_xlabel("% of Restaurants")
for i, (hood, val) in enumerate(high_pct.items()):
    ax6.text(val + 0.5, i, f"{val:.0f}%", va="center", fontsize=8)

out = REPO_ROOT / "notebooks" / "eda_charts.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")

# Print key stats for slides
print(f"\nTotal restaurants: {len(df)}")
print(f"High-rated (>=4.5): {df['high_rated'].sum()} ({df['high_rated'].mean()*100:.1f}%)")
print(f"Neighborhoods: {df['neighborhood'].nunique()}")
print(f"Price level coverage: {df['price_level'].notna().sum()} have price data")
print(f"\nRating breakdown:")
for cutoff in [3.0, 3.5, 4.0, 4.5, 5.0]:
    pct = (df["rating"] >= cutoff).mean() * 100
    print(f"  >= {cutoff}: {pct:.1f}%")
