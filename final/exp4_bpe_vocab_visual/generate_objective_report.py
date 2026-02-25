
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rdkit import Chem
import collections
import re

# --- Configuration ---
PROJECT_ROOT = Path('/home/gzy/py/tokenizerGraph')
sys.path.append(str(PROJECT_ROOT))
CODEBOOK_PATH = PROJECT_ROOT / 'model/bpe/zinc/smiles/multi_100/bpe_codebook.pkl'
REPORT_PATH = PROJECT_ROOT / 'final/exp4_bpe_vocab_visual/vocab_analysis_report.md'

def load_bpe_codebook(path: Path) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)

class BpeDecoder:
    def __init__(self, merge_rules: List[Tuple[int, int, int]]):
        self.id_to_pair = {new_id: (r1, r2) for r1, r2, new_id in merge_rules}
        all_ids = set()
        new_ids = {rule[2] for rule in merge_rules}
        for r1, r2, _ in merge_rules:
            all_ids.add(r1)
            all_ids.add(r2)
        self.base_token_ids = all_ids - new_ids
        self.all_token_ids = all_ids.union(new_ids)

    def decode_token_to_ids(self, token_id: int) -> List[int]:
        if token_id in self.base_token_ids:
            return [token_id]
        if token_id not in self.id_to_pair:
            return []
        r1, r2 = self.id_to_pair[token_id]
        return self.decode_token_to_ids(r1) + self.decode_token_to_ids(r2)

    def decode_to_string(self, token_id: int) -> str:
        base_ids = self.decode_token_to_ids(token_id)
        try:
            chars = [chr(i) for i in base_ids]
            return "".join(chars)
        except ValueError:
            return ""

def count_heavy_atoms(smiles: str) -> int:
    # Regex for robust heavy atom counting (C, c, N, n, O, o, S, s, F, Cl, Br, I, P, etc.)
    pattern = r"(\[[^\]]+\]|Cl|Br|Si|Se|Na|Li|K|Mg|Ca|Fe|Zn|Cu|Mn|Co|Ni|Cd|Hg|[BCNOPSFIbcnopsfi])"
    return len(re.findall(pattern, smiles))

def analyze_atom_types(smiles: str) -> List[str]:
    types = []
    if 'N' in smiles or 'n' in smiles: types.append('Nitrogen')
    if 'O' in smiles or 'o' in smiles: types.append('Oxygen')
    if 'S' in smiles or 's' in smiles: types.append('Sulfur')
    if 'F' in smiles: types.append('Fluorine')
    if 'Cl' in smiles: types.append('Chlorine')
    if 'Br' in smiles: types.append('Bromine')
    # Check for purely Carbon/Hydrocarbon (ignoring H in SMILES usually)
    # Heuristic: if no N, O, S, F, Cl, Br, P, etc.
    has_hetero = re.search(r'[NOPSFIbcnopsfi]|Cl|Br', smiles)
    if not has_hetero:
        types.append('Hydrocarbon_Only')
    return types

def analyze_rings(smiles: str) -> int:
    # Heuristic: count pairs of digits
    # Find all digits
    digits = re.findall(r'\d', smiles)
    # Simple approximation: count digits / 2. (Not perfect for fused rings sharing numbers, but good proxy for complexity)
    # Better: use RDKit if possible.
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol:
        try:
            return mol.GetRingInfo().NumRings()
        except:
            pass
    # Fallback to digit counting pairs
    return len(digits) // 2

def generate_objective_report():
    codebook = load_bpe_codebook(CODEBOOK_PATH)
    merge_rules = codebook.get('merge_rules')
    decoder = BpeDecoder(merge_rules)
    
    # Data containers
    all_tokens = []
    size_dist = collections.defaultdict(int)
    atom_type_dist = collections.defaultdict(int)
    ring_count_dist = collections.defaultdict(int)
    rank_data = {} # token_id -> rank (0 is first merge)
    
    # Map merge rank
    for i, rule in enumerate(merge_rules):
        rank_data[rule[2]] = i
        
    # Analyze merged tokens
    for token_id in decoder.all_token_ids:
        if token_id in decoder.base_token_ids:
            rank = -1 # Atomic/Base
            type_ = 'Atomic'
        else:
            rank = rank_data.get(token_id, 999999)
            type_ = 'Merged'
            
        s = decoder.decode_to_string(token_id)
        size = count_heavy_atoms(s)
        
        # Objective stats
        size_dist[size] += 1
        
        # Atom types
        a_types = analyze_atom_types(s)
        for at in a_types:
            atom_type_dist[at] += 1
            
        # Ring count
        n_rings = analyze_rings(s)
        ring_count_dist[n_rings] += 1
        
        all_tokens.append({
            'id': token_id,
            'smiles': s,
            'size': size,
            'rank': rank,
            'rings': n_rings,
            'atom_types': a_types
        })
        
    # Sort by rank for high-freq analysis
    # Filter only merged tokens for rank analysis
    merged_tokens = [t for t in all_tokens if t['rank'] != -1]
    merged_tokens.sort(key=lambda x: x['rank'])
    
    # --- Generate Report ---
    with open(REPORT_PATH, 'w') as f:
        f.write("# Objective Analysis of BPE Vocabulary (ZINC)\n\n")
        
        # Section 1: Overall Size Distribution
        f.write("## 1. Token Size Distribution (Heavy Atoms)\n")
        f.write("| Size (Atoms) | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        total = len(all_tokens)
        for size in sorted(size_dist.keys()):
            count = size_dist[size]
            pct = (count / total) * 100
            f.write(f"| {size} | {count} | {pct:.2f}% |\n")
            
        # Section 2: Compositional Analysis
        f.write("\n## 2. Elemental Composition (Token Count by Atom Type)\n")
        f.write("*Note: Categories are not mutually exclusive.*\n\n")
        f.write("| Atom Type Present | Token Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for at, count in sorted(atom_type_dist.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100
            f.write(f"| {at} | {count} | {pct:.2f}% |\n")
            
        # Section 3: Topological Complexity (Ring Counts)
        f.write("\n## 3. Topological Complexity (Ring Content)\n")
        f.write("| Number of Rings | Token Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for n_r in sorted(ring_count_dist.keys()):
            count = ring_count_dist[n_r]
            pct = (count / total) * 100
            f.write(f"| {n_r} | {count} | {pct:.2f}% |\n")
            
        # Section 4: Top 50 Frequent Merged Tokens (Rank 0-49)
        f.write("\n## 4. Top 50 Highest Priority Merged Tokens (Rank 0-49)\n")
        f.write("Objective observation of the first 50 substructures learned by the algorithm.\n\n")
        f.write("| Rank | SMILES | Size | Rings | Composition |\n")
        f.write("|---|---|---|---|---|\n")
        for t in merged_tokens[:50]:
            comp_str = ", ".join(t['atom_types']) if t['atom_types'] else "C/H only"
            f.write(f"| {t['rank']} | `{t['smiles']}` | {t['size']} | {t['rings']} | {comp_str} |\n")
            
        # Section 5: Specific Functional Group Detection (Objective Search)
        f.write("\n## 5. Rank of Specific Chemical Substructures\n")
        f.write("First occurrence rank of common SMARTS patterns (exact string match):\n\n")
        
        targets = {
            'Carbonyl (C=O)': ['C=O'],
            'Amide (NC=O)': ['NC=O', 'C(=O)N'],
            'Benzene (c1ccccc1)': ['c1ccccc1', 'c1ccccc1', 'c2ccccc2'],
            'Carboxyl (C(=O)[O-])': ['C(=O)[O-]', 'C(=O)O'],
            'Sulfonyl (S(=O)(=O))': ['S(=O)(=O)'],
            'Nitro ([N+](=O)[O-])': ['[N+](=O)[O-]'],
            'Methoxy (CO)': ['CO', 'OC'],
            'Cyano (C#N)': ['C#N']
        }
        
        f.write("| Substructure | First Rank | SMILES |\n")
        f.write("|---|---|---|\n")
        
        for name, patterns in targets.items():
            found = False
            for t in merged_tokens:
                if t['smiles'] in patterns:
                    f.write(f"| {name} | {t['rank']} | `{t['smiles']}` |\n")
                    found = True
                    break
            if not found:
                f.write(f"| {name} | Not Found (Exact Match) | - |\n")

    print(f"Report generated at {REPORT_PATH}")

if __name__ == "__main__":
    generate_objective_report()





