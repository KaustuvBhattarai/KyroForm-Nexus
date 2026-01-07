import requests
import zipfile
import os
from tqdm import tqdm

url = "https://zenodo.org/records/14780446/files/Healty_Bac_predictions.zip?download=1"
filename = "Healty_Bac_predictions.zip"
extract_dir = "gut_host_ppi_predictions"

print("gettin file size")
head = requests.head(url, allow_redirects=True)
total_size = int(head.headers.get('content-length', 0)) if 'content-length' in head.headers else None
size_mb = total_size / (1024**2) if total_size else "~419"

print(f"File size: {size_mb:.1f} MB")

confirm = input("\nProceed downloadin? (y/n): ").strip().lower()
if confirm != 'y':
    print("Cancelled.")
    exit()

print("Downloading")
r = requests.get(url, stream=True)
r.raise_for_status()

progress = tqdm(total=total_size, unit='B', unit_scale=True, desc="Download")
with open(filename, 'wb') as f:
    for chunk in r.iter_content(1024*1024):
        f.write(chunk)
        progress.update(len(chunk))
progress.close()
print("Finished")

print("Now Extracting")
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(filename) as z:
    z.extractall(extract_dir)
print(f"Done and Files are in ./{extract_dir}/ ig")
print("fuck fuck fuck UwU")