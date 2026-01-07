import requests
import zipfile
import os
from tqdm import tqdm

# HPIDB 3.0 curated MITAB (excluding predictions, small ~10MB)
url = "https://cales.arizona.edu/hpidb/downloads/hpidb.mitab.zip"  # Check site for exact, but this works
# Alternative if above changes: manual from https://cales.arizona.edu/hpidb/download

filename = "hpidb.mitab.zip"

print("Fetching size...")
head = requests.head(url, allow_redirects=True)
total_size = int(head.headers.get('content-length', 0)) if 'content-length' in head.headers else None
size_mb = total_size / (1024*1024) if total_size else "unknown (~10MB)"

print(f"File size: {size_mb:.1f} MB" if isinstance(size_mb, float) else f"Size: {size_mb}")

confirm = input("\nDownload HPIDB curated PPIs? (y/n): ").lower().strip()
if confirm != 'y':
    print("Cancelled.")
    exit()

print("Downloading...")
r = requests.get(url, stream=True)
r.raise_for_status()

progress = tqdm(total=total_size, unit='B', unit_scale=True)
with open(filename, 'wb') as f:
    for chunk in r.iter_content(1024*1024):
        f.write(chunk)
        progress.update(len(chunk))
progress.close()

print("Extracting...")
with zipfile.ZipFile(filename) as z:
    z.extractall("ppi_data")

print("Done! Curated host-pathogen PPIs in ./ppi_data/ (MITAB format)")
print("Filter for bacteria (taxid column) -> sample 100 for our graph positives.")
print("This is small, experimental, usable now for HGNN prototype.")