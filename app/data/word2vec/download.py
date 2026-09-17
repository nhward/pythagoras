import gzip
import itertools
import urllib.request
from pathlib import Path

url = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
)
urllib.request.urlretrieve(url, "glove-50.gz")

with gzip.open("glove-50.gz", "rt", encoding="utf-8") as source:
    _, dimensions = map(int, next(source).split())
    rows = list(itertools.islice(source, 10_000))

output = Path("glove-50-10000.vec")
output.write_text(
    f"{len(rows)} {dimensions}\n" + "".join(rows),
    encoding="utf-8",
)
print(output.resolve())