
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rdkit import Chem
import csv

# --- Configuration ---
PROJECT_ROOT = Path('/home/gzy/py/tokenizerGraph')
sys.path.append(str(PROJECT_ROOT))
CODEBOOK_PATH = PROJECT_ROOT / 'model/bpe/zinc/smiles/multi_100/bpe_codebook.pkl'

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

def find_domain_tokens():
    codebook = load_bpe_codebook(CODEBOOK_PATH)
    merge_rules = codebook.get('merge_rules')
    if not merge_rules:
        return

    decoder = BpeDecoder(merge_rules)
    
    # Target Patterns (SMARTS or Canonical SMILES)
    # Benzene: c1ccccc1
    # Carboxyl: C(=O)O or C(=O)[O-]
    # Amide: C(=O)N
    # Methyl: C (usually too simple, but maybe -CH3 in context)
    # Nitro: [N+](=O)[O-]
    # Sulfonyl: S(=O)(=O)
    
    targets = {
        'Benzene': ['c1ccccc1', 'C1=CC=CC=C1'],
        'Carboxyl': ['C(=O)O', 'C(O)=O', 'OC=O', 'C(=O)[O-]'],
        'Amide': ['C(=O)N', 'NC=O', 'NC(=O)'],
        'Nitro': ['[N+](=O)[O-]', 'N(=O)=O'],
        'Sulfonyl': ['S(=O)(=O)', 'O=S=O'],
        'Carbonyl': ['C=O'],
        'Cyano': ['C#N'],
        'Methoxy': ['COC', 'CO'], # Context dependent
        'Fluorobenzene fragment': ['c1ccc(F)cc1']
    }
    
    print(f"Total rules: {len(merge_rules)}")
    
    found_tokens = []
    
    # Iterate through all merged tokens
    # High rank = appears LATER in merge_rules list? 
    # Usually merge rules are ordered by frequency (most frequent pairs merged first).
    # WAIT. In standard BPE, the first merge is the MOST frequent pair.
    # So index 0 (rank 0) is the most frequent/important substructure found first.
    # Index 2999 (rank 2999) is the least frequent of the top 3000 merges.
    # Let's verify this assumption. "High-frequency tokens" usually means tokens formed early (low index) or tokens that appear often in the dataset.
    # If a token is formed at step 0, it means its components (atomic) were the most frequent pair.
    # Let's assume merge_rules[i] is the i-th merge.
    
    for i, rule in enumerate(merge_rules):
        token_id = rule[2]
        s = decoder.decode_to_string(token_id)
        
        # Canonicalize if possible
        mol = Chem.MolFromSmiles(s, sanitize=False)
        canonical_s = ""
        if mol:
            try:
                canonical_s = Chem.MolToSmiles(mol)
            except:
                pass
                
        match_type = None
        
        # Check targets
        for name, patterns in targets.items():
            # Check exact string or canonical match
            if s in patterns or (canonical_s and canonical_s in patterns):
                match_type = name
                break
            # Check if it's a simple substructure (optional, exact match is better for "token IS functional group")
        
        if match_type:
            found_tokens.append({
                'rank': i, # 0-indexed, 0 is first merge
                'token_id': token_id,
                'smiles': s,
                'type': match_type
            })
            
    # Sort by rank (early merges first)
    found_tokens.sort(key=lambda x: x['rank'])
    
    print("\n--- Domain Alignment Candidates (First 30 found) ---")
    print(f"{'Rank':<6} | {'ID':<6} | {'Type':<15} | {'SMILES'}")
    for t in found_tokens[:30]:
        print(f"{t['rank']:<6} | {t['token_id']:<6} | {t['type']:<15} | {t['smiles']}")

    # Also check specific benzene and carboxyl counts/ranks
    benzene = [t for t in found_tokens if t['type'] == 'Benzene']
    carboxyl = [t for t in found_tokens if t['type'] == 'Carboxyl']
    
    print(f"\nBenzene tokens found: {len(benzene)}")
    if benzene:
        print(f"Earliest Benzene Rank: {benzene[0]['rank']}")
        
    print(f"Carboxyl tokens found: {len(carboxyl)}")
    if carboxyl:
        print(f"Earliest Carboxyl Rank: {carboxyl[0]['rank']}")

if __name__ == "__main__":
    find_domain_tokens()





