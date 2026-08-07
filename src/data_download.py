"""
Downloads and preprocesses DNA sequences from direct bulk file downloads.
"""

import argparse
import gzip
import hashlib
import io
import random
import tarfile
import urllib.request
from pathlib import Path

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

# Download URLs

# CARD latest data tarball (resistance genes and drug class metadata)
CARD_URL = "https://card.mcmaster.ca/latest/data"

# Ensembl Bacteria FTP — E. coli K-12 MG1655 CDS FASTA
ECOLI_CDS_URL = (
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria/current/fasta/"
    "bacteria_0_collection/escherichia_coli_str_k_12_substr_mg1655_gca_000005845/"
    "cds/Escherichia_coli_str_k_12_substr_mg1655_gca_000005845.ASM584v2.cds.all.fa.gz"
)

# Ensembl Bacteria FTP — Staphylococcus aureus N315 CDS FASTA
# Phylogenetically distant from E. coli — adds diversity to non-resistant class
STAPH_CDS_URL = (
    "https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria/current/fasta/"
    "bacteria_0_collection/staphylococcus_aureus_subsp_aureus_n315_gca_000009645/"
    "cds/Staphylococcus_aureus_subsp_aureus_n315_gca_000009645.ASM964v1.cds.all.fa.gz"
)


# CARD drug class -> project class mapping

CARD_CLASS_MAP = {
    "beta_lactam": [
        "beta-lactam antibiotic",
        "cephalosporin",
        "carbapenem",
        "monobactam",
        "penam",
        "penem",
        "cephamycin",
        "oxacillin",
    ],
    "tetracycline": [
        "tetracycline antibiotic",
        "glycylcycline",
    ],
    "aminoglycoside": [
        "aminoglycoside antibiotic",
        "aminocyclitol antibiotic",
    ],
    "macrolide": [
        "macrolide antibiotic",
        "lincosamide antibiotic",
        "streptogramin antibiotic",
        "oxazolidinone antibiotic",
        "pleuromutilin antibiotic",
    ],
    "fluoroquinolone": [
        "fluoroquinolone antibiotic",
        "quinolone antibiotic",
    ],
}


# Quality filters

MIN_LEN = 300
MAX_LEN = 5000
MAX_N_FRACTION = 0.05


def is_valid_sequence(seq: str) -> bool:
    """Basic quality filter: length, ambiguous bases, valid alphabet."""
    seq = seq.upper()
    if len(seq) < MIN_LEN or len(seq) > MAX_LEN:
        return False
    if seq.count("N") / len(seq) > MAX_N_FRACTION:
        return False
    if not all(b in "ATGCN" for b in seq):
        return False
    return True


# Download helper

def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """
    Download a file to dest with a progress bar.
    Skips if the file already exists.
    """
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name
    print(f"\n  Downloading: {label}")
    print(f"  URL: {url}")

    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            with open(dest, "wb") as f:
                with tqdm(total=total or None, unit="B", unit_scale=True,
                          desc=f"    {dest.name}", leave=False) as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed for {url}:\n  {e}")

    print(f"  Saved -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# CARD parsing

def build_accession_lookup(tar: tarfile.TarFile) -> dict[str, str]:
    """
    Parse aro_categories_index.tsv from the CARD tarball and build a
    dict mapping DNA accession -> project class name.

    TSV columns (tab-separated):
      Protein Accession | DNA Accession | AMR Gene Family | Drug Class | Resistance Mechanism

    """
    lookup: dict[str, str] = {}

    member = tar.extractfile("./aro_categories_index.tsv")
    text = io.TextIOWrapper(member, encoding="utf-8", errors="replace")

    for i, line in enumerate(text):
        if i == 0:
            continue  # skip header
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue

        dna_accession = parts[1].strip()
        drug_classes_raw = parts[3].strip().lower()

        # drug_classes_raw may be semicolon-separated
        cls = None
        for drug_class in drug_classes_raw.split(";"):
            drug_class = drug_class.strip()
            for project_class, keywords in CARD_CLASS_MAP.items():
                if any(kw in drug_class for kw in keywords):
                    cls = project_class
                    break
            if cls:
                break

        if cls and dna_accession:
            # Store both with and without version suffix 
            lookup[dna_accession] = cls
            lookup[dna_accession.split(".")[0]] = cls

    return lookup


def extract_accession(record_id: str) -> str:
    """
    Extract the bare accession from a CARD FASTA record ID.

    """
    parts = record_id.split("|")
    # Second field after 'gb' is the accession
    if len(parts) >= 2:
        return parts[1].strip()
    return record_id.split(".")[0]


def parse_card_fasta(card_tarball: Path) -> dict[str, list[SeqRecord]]:
    """
    Open the CARD tarball, build an accession -> class lookup from
    aro_categories_index.tsv, then classify sequences from ALL
    nucleotide FASTA files in the tarball.

    Returns dict: class_name -> list of SeqRecord
    """
    print(f"\n  Parsing CARD tarball: {card_tarball.name}")
    buckets: dict[str, list[SeqRecord]] = {k: [] for k in CARD_CLASS_MAP}
    n_total = 0
    n_classified = 0

    with tarfile.open(card_tarball, "r:bz2") as tar:
        # Step 1: build accession -> class lookup
        print("  Building accession lookup from aro_categories_index.tsv ...")
        accession_map = build_accession_lookup(tar)
        print(f"  Lookup entries: {len(accession_map)}")

        # Step 2: find ALL nucleotide FASTA files
        fasta_names = [
            name for name in tar.getnames()
            if "nucleotide_fasta" in name and name.endswith(".fasta")
        ]
        print(f"  Found {len(fasta_names)} nucleotide FASTA files")

        # Step 3: parse each one
        for fasta_name in fasta_names:
            print(f"  Parsing: {fasta_name}")
            member = tar.extractfile(fasta_name)
            text_wrapper = io.TextIOWrapper(member, encoding="utf-8", errors="replace")

            for record in SeqIO.parse(text_wrapper, "fasta"):
                n_total += 1
                seq = str(record.seq).upper()
                if not is_valid_sequence(seq):
                    continue

                accession = extract_accession(record.id)
                cls = accession_map.get(accession) or accession_map.get(accession.split(".")[0])

                if cls:
                    record.description = f"{cls} | {record.description}"
                    buckets[cls].append(record)
                    n_classified += 1

    # Deduplicate within each class by sequence hash
    for cls in buckets:
        seen: set[str] = set()
        unique = []
        for r in buckets[cls]:
            h = hashlib.md5(str(r.seq).upper().encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(r)
        buckets[cls] = unique

    print(f"  Total records: {n_total} | Classified: {n_classified}")
    for cls, recs in buckets.items():
        print(f"    {cls:<25} {len(recs):>5}")

    return buckets



# Ensembl Bacteria parsing

def parse_ensembl_cds(gz_path: Path, source_label: str) -> list[SeqRecord]:
    """
    Parse a gzipped Ensembl Bacteria CDS FASTA.
    Accepts all valid coding sequences — no gene name filter.
    All CDS from E. coli K-12 and S. aureus are non-resistant by definition.
    Quality filters (length, valid DNA alphabet) still apply.
    """
    print(f"\n  Parsing: {gz_path.name}")
    records = []

    with gzip.open(gz_path, "rt", errors="replace") as f:
        for record in SeqIO.parse(f, "fasta"):
            seq = str(record.seq).upper()
            # Skip protein sequences that occasionally appear in CDS files
            if not all(b in "ATGCN" for b in seq):
                continue
            if not is_valid_sequence(seq):
                continue
            record.description = f"non_resistant | {source_label} | {record.description}"
            records.append(record)

    print(f"  Found {len(records)} non-resistant CDS sequences")
    return records


# Main steps

def download_all(raw_dir: Path):
    """Download all source files to raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("STEP 1: Downloading source files")
    print("=" * 60)

    download_file(CARD_URL,      raw_dir / "card_data.tar.bz2", "CARD bulk data")
    download_file(ECOLI_CDS_URL, raw_dir / "ecoli_cds.fa.gz",  "E. coli K-12 CDS (Ensembl)")
    download_file(STAPH_CDS_URL, raw_dir / "staph_cds.fa.gz",  "S. aureus N315 CDS (Ensembl)")


def parse_and_split(raw_dir: Path, n_per_class: int, seed: int = 42):
    """
    Parse downloaded files and write two FASTA files:
      - resistant.fasta    all CARD resistance genes merged into one class
      - non_resistant.fasta  E. coli K-12 + S. aureus housekeeping genes
    Both capped at n_per_class sequences.
    """
    random.seed(seed)
    print("\n" + "=" * 60)
    print("STEP 2: Parsing and splitting into binary FASTA files")
    print("=" * 60)

    # Resistant class: merge all CARD classes
    resistant_path = raw_dir / "resistant.fasta"
    if resistant_path.exists():
        print(f"  [SKIP] resistant.fasta already exists")
    else:
        card_tarball = raw_dir / "card_data.tar.bz2"
        if not card_tarball.exists():
            raise FileNotFoundError(
                f"{card_tarball} not found. Run with --download first."
            )

        card_buckets = parse_card_fasta(card_tarball)

        # Merge all resistance classes into one pool
        all_resistant: list[SeqRecord] = []
        for cls, records in card_buckets.items():
            for r in records:
                r.description = f"resistant | {r.description}"
                all_resistant.append(r)

        # Deduplicate by sequence hash across all classes
        seen: set[str] = set()
        unique_resistant: list[SeqRecord] = []
        for r in all_resistant:
            h = hashlib.md5(str(r.seq).upper().encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique_resistant.append(r)

        random.shuffle(unique_resistant)
        selected = unique_resistant[:n_per_class]
        SeqIO.write(selected, resistant_path, "fasta")
        print(f"\n  Wrote {len(selected):>5} sequences -> resistant.fasta")
        print(f"  (merged from {sum(len(v) for v in card_buckets.values())} total CARD sequences)")

    # Non-resistant class from Ensembl Bacteria
    nr_path = raw_dir / "non_resistant.fasta"
    if nr_path.exists():
        print(f"  [SKIP] non_resistant.fasta already exists")
    else:
        all_housekeeping: list[SeqRecord] = []

        ecoli_gz = raw_dir / "ecoli_cds.fa.gz"
        if not ecoli_gz.exists():
            raise FileNotFoundError(
                f"{ecoli_gz} not found. Run with --download first."
            )
        all_housekeeping.extend(parse_ensembl_cds(ecoli_gz, "E.coli_K12"))

        staph_gz = raw_dir / "staph_cds.fa.gz"
        if staph_gz.exists():
            all_housekeeping.extend(parse_ensembl_cds(staph_gz, "S.aureus_N315"))
        else:
            print(f"  WARNING: {staph_gz.name} not found, skipping")

        if not all_housekeeping:
            raise RuntimeError(
                "No housekeeping sequences found. "
                "Check that ecoli_cds.fa.gz and staph_cds.fa.gz downloaded correctly."
            )

        # Deduplicate by sequence hash
        seen2: set[str] = set()
        unique_nr: list[SeqRecord] = []
        for r in all_housekeeping:
            h = hashlib.md5(str(r.seq).upper().encode()).hexdigest()
            if h not in seen2:
                seen2.add(h)
                unique_nr.append(r)

        random.shuffle(unique_nr)
        selected_nr = unique_nr[:n_per_class]
        SeqIO.write(selected_nr, nr_path, "fasta")
        print(f"  Wrote {len(selected_nr):>5} sequences -> non_resistant.fasta")

    print_summary(raw_dir)


def print_summary(out_dir: Path):
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    total = 0
    for fasta in sorted(out_dir.glob("*.fasta")):
        n = sum(1 for _ in SeqIO.parse(fasta, "fasta"))
        total += n
        print(f"  {fasta.stem:<25} {n:>5} sequences")
    print(f"  {'TOTAL':<25} {total:>5} sequences")
    print("=" * 60)


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Prepare DNA sequences for CGR project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/data_download.py --download --output_dir data/raw
  python src/data_download.py --parse   --output_dir data/raw --n_per_class 1667
  python src/data_download.py --download --parse --output_dir data/raw
        """
    )
    parser.add_argument("--output_dir", default="data/raw",
                        help="Directory for downloads and output FASTA files (default: data/raw)")
    parser.add_argument("--download", action="store_true",
                        help="Download source files from CARD and Ensembl Bacteria")
    parser.add_argument("--parse", action="store_true",
                        help="Parse downloaded files into per-class FASTA files")
    parser.add_argument("--n_per_class", type=int, default=1667,
                        help="Max sequences per class (default: 1667 -> ~10k total)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    args = parser.parse_args()

    if not args.download and not args.parse:
        parser.print_help()
        print("\nHint: use --download to fetch files, --parse to process them, or both together.")
        return

    raw_dir = Path(args.output_dir)

    if args.download:
        download_all(raw_dir)

    if args.parse:
        parse_and_split(raw_dir, args.n_per_class, args.seed)


if __name__ == "__main__":
    main()
