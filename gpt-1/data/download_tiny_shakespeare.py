"""Download the Tiny Shakespeare text shared by all three language versions."""

from pathlib import Path
from urllib.request import urlopen


DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATASET_PATH = Path(__file__).with_name("input.txt")


def main() -> None:
    with urlopen(DATASET_URL) as response:
        text = response.read()

    DATASET_PATH.write_bytes(text)
    print(f"Downloaded {len(text):,} bytes to {DATASET_PATH}")


if __name__ == "__main__":
    main()
