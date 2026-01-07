import requests
import zipfile
import io
import pandas as pd

url = "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.zip"

print("DL full IntAct dataset")
response = requests.get(url)
if response.status_code == 200:
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Usually one big file: intact.txt there is
        file_name = [f for f in z.namelist() if f.endswith('.txt')][0]
        with z.open(file_name) as f:
            df = pd.read_csv(f, sep='\t', header=None, low_memory=False)
    
    # Standard MITAB columns (simplified)
    # Column 9: Taxid interactor A, Column 10: Taxid interactor B
    df.columns = [f"col{i}" for i in range(len(df.columns))]
    
    # Filter for interactions where one is human (taxid:9606) and other is bacteria (taxid starts with 'taxid:2') idk if there's anything elso relevant add later
    human_tax = "taxid:9606"
    bacteria_prefix = "taxid:2"
    
    mask = (
        ((df['col9'] == human_tax) & (df['col10'].str.startswith(bacteria_prefix))) |
        ((df['col10'] == human_tax) & (df['col9'].str.startswith(bacteria_prefix)))
    )
    
    filtered = df[mask]
    
    filtered.to_csv("human_bacteria_ppis_intact.txt", sep='\t', index=False)
    print(f"Don and Saved {len(filtered)} human bacteria PPIs to human_bacteria_ppis_intact.txt")
else:
    print("Download faileddddd")