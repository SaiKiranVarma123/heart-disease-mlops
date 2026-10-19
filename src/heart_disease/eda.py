import matplotlib.pyplot as plt
import seaborn as sns
from .config import CLEAN_DATA, FIGURE_DIR, RAW_DATA
from .data import load_raw

def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DATA.parent.mkdir(parents=True, exist_ok=True)
    df = load_raw(RAW_DATA)
    df.to_csv(CLEAN_DATA, index=False)
    df.hist(figsize=(14, 10), bins=20); plt.tight_layout(); plt.savefig(FIGURE_DIR / "feature_histograms.png", dpi=160); plt.close()
    plt.figure(figsize=(11, 8)); sns.heatmap(df.corr(numeric_only=True), cmap="vlag", center=0); plt.tight_layout(); plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=160); plt.close()
    plt.figure(figsize=(6, 4)); sns.countplot(data=df, x="target", hue="target", legend=False); plt.title("Class balance"); plt.tight_layout(); plt.savefig(FIGURE_DIR / "class_balance.png", dpi=160); plt.close()
    print(f"Cleaned {len(df)} rows; figures saved to {FIGURE_DIR}")

if __name__ == "__main__": main()
