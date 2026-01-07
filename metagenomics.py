from Bio import Entrez

Entrez.email = "metalvolvox@example.com"

search = Entrez.esearch(
    db="sra",
    term="SRP235678",
    retmax=100
)
record = Entrez.read(search)
search.close()

sra_ids = record["IdList"]
print(f"Retrieved {len(sra_ids)} SRA IDs")
