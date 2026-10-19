from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://www.archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DEST = Path("data/raw/processed.cleveland.data")


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:  # nosec B310: trusted UCI HTTPS host
        payload = response.read()
    data = payload
    if len(data) < 10_000 or b"," not in data:
        raise RuntimeError("Downloaded file failed basic validation")
    DEST.write_bytes(data)
    print(f"Saved {len(data):,} bytes to {DEST}")


if __name__ == "__main__":
    main()
