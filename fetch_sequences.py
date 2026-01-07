import requests
import time
import os

protein_list_path = '/home/satkrit/Downloads/Kyroform_RV/ppi_data/unique_proteins_for_sequences.txt'
output_fasta = '/home/satkrit/Downloads/Kyroform_RV/ppi_data/all_proteins.fasta'

with open(protein_list_path, 'r') as f:
    proteins = [line.strip() for line in f if line.strip()]

print(f"Fetching FASTA for {len(proteins)} proteins via new UniProt REST API...")

base_url = "https://rest.uniprot.org/uniprotkb/accessions"

with open(output_fasta, 'w') as out:
    for i in range(0, len(proteins), 200):  # Batch 200 (safe limit)
        batch = proteins[i:i+200]
        params = {
            'accessions': ','.join(batch),
            'format': 'fasta'
        }
        print(f"Batch {i//200 + 1}/{ (len(proteins)-1)//200 + 1 } ({len(batch)} proteins)...")
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            out.write(response.text)
        else:
            print(f"Error batch {i//200 + 1}: {response.status_code} - {response.text}")
        
        time.sleep(1)  # Polite delay

print(f"\nFASTA saved to {output_fasta}")
print(f"Size: {os.path.getsize(output_fasta) / (1024**2):.1f} MB")