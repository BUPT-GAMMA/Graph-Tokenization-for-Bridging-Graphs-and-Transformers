
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np
import collections

# --- Configuration ---
PROJECT_ROOT = Path('/home/gzy/py/tokenizerGraph')
sys.path.append(str(PROJECT_ROOT))
CODEBOOK_PATH = PROJECT_ROOT / 'model/bpe/zinc/smiles/multi_100/bpe_codebook.pkl'

def load_bpe_codebook(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Error: BPE codebook file not found: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ BPE codebook loaded: {path}")
    return data

class BpeDecoder:
    def __init__(self, merge_rules: List[Tuple[int, int, int]]):
        self._build_decoding_maps(merge_rules)

    def _build_decoding_maps(self, merge_rules: List[Tuple[int, int, int]]):
        self.id_to_pair: Dict[int, Tuple[int, int]] = {new_id: (r1, r2) for r1, r2, new_id in merge_rules}
        all_ids = set()
        new_ids = {rule[2] for rule in merge_rules}
        for r1, r2, _ in merge_rules:
            all_ids.add(r1)
            all_ids.add(r2)
        
        self.base_token_ids = all_ids - new_ids
        self.all_token_ids = all_ids.union(new_ids)
        print(f"🔍 Decoder initialized: {len(self.base_token_ids)} base tokens, {len(merge_rules)} merge rules.")

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
        except ValueError as e:
            return ""

import re

def count_atoms(smiles: str) -> int:
    if not smiles:
        return 0
    
    # Robust counting using Regex for SMILES atoms
    # Matches:
    # 1. Atoms in brackets: [NH+], [O-], [C@H] -> Count as 1
    # 2. Two-letter atoms: Cl, Br, Si, Se, Na, Li, K, Mg, Ca, Fe, Zn, Cu, Mn, Co, Ni, Cd, Hg
    # 3. One-letter atoms: C, N, O, S, P, F, I, B, c, n, o, s, p (aromatic or aliphatic)
    
    # Note: 'Sc' is Scandium, but 'S' 'c' is Sulfur and aromatic carbon. 
    # Standard SMILES parsers handle this greedily. 
    # ZINC mainly organic: C, N, O, S, F, P, Cl, Br, I.
    
    pattern = r"(\[[^\]]+\]|Cl|Br|Si|Se|Na|Li|K|Mg|Ca|Fe|Zn|Cu|Mn|Co|Ni|Cd|Hg|[BCNOPSFIbcnopsfi])"
    
    # Find all matches
    atoms = re.findall(pattern, smiles)
    return len(atoms)

def analyze_vocabulary():
    codebook = load_bpe_codebook(CODEBOOK_PATH)
    merge_rules = codebook.get('merge_rules')
    if not merge_rules:
        print("Error: merge_rules not found.")
        return

    decoder = BpeDecoder(merge_rules)
    
    # Collect all tokens
    all_tokens = []
    
    # Add base tokens
    for tid in decoder.base_token_ids:
        s = decoder.decode_to_string(tid)
        all_tokens.append({'id': tid, 'smiles': s, 'type': 'base'})
        
    # Add merged tokens
    for r1, r2, tid in merge_rules:
        s = decoder.decode_to_string(tid)
        all_tokens.append({'id': tid, 'smiles': s, 'type': 'merged'})
        
    print(f"Total tokens to analyze: {len(all_tokens)}")
    
    atom_counts = []
    
    # Detailed analysis for Table
    # Categories:
    # 1: Single Node / Edge (We will count single atoms as 1. Non-atoms are 0, effectively ignored in Node Count or counted as specialized?)
    # The table asks for "Raw Node Num".
    # I will assume 0-atom tokens (syntax) are not "patterns" in the sense of subgraphs, or they are "atomic" in a different sense.
    # But the prompt says "atomic tokens representing single nodes/edges".
    # Let's stick to Node Count.
    
    count_stats = collections.defaultdict(int)
    
    # Alignment Check (Benzene, Carboxyl)
    benzene_patterns = ['c1ccccc1', 'c1ccccc1', 'C1=CC=CC=C1'] # Canonical forms might vary
    carboxyl_patterns = ['C(=O)O', 'C(=O)[O-]', 'C(O)=O'] 
    
    found_benzene = 0
    found_carboxyl = 0
    
    for t in all_tokens:
        s = t['smiles']
        num_atoms = count_atoms(s)
        atom_counts.append(num_atoms)
        count_stats[num_atoms] += 1
        
        # Simple string matching for alignment (approximate)
        # In reality, we should use substructure match, but let's see if we find exact matches or close ones.
        # Since we have SMARTS/SMILES, we can try exact string match or canonical check.
        
        # Canonical check for domain alignment
        if num_atoms >= 6:
             try:
                 mol = Chem.MolFromSmiles(s)
                 if mol:
                     can = Chem.MolToSmiles(mol)
                     # Benzene check
                     if can == 'c1ccccc1':
                         found_benzene += 1
             except:
                 pass
        if num_atoms >= 2: # Carboxyl is C, O, O -> 3 atoms
             try:
                 mol = Chem.MolFromSmiles(s)
                 if mol:
                     # Check for carboxyl group substructure
                     # This is harder without exact definition of "token corresponds to". 
                     # If the token IS the group.
                     can = Chem.MolToSmiles(mol)
                     if can in ['OC=O', 'O=CO', '[O-]C=O']: 
                         found_carboxyl += 1
             except:
                 pass

    total_vocab = len(all_tokens)
    
    # Calculate percentages for the table
    # Range: 2-3, 3-10, 10+
    # Note: Ranges overlap? "2~3" and "3~10". Usually means [2, 3] and (3, 10] or [3, 10]?
    # "2~3" likely means 2 or 3.
    # "3~10" likely means 4 to 10? Or 3 to 10?
    # Given "2~3", I will assume it covers 2 and 3.
    # Then "3~10" is ambiguous. I will assume it means >3 and <=10 (i.e., 4,5,6,7,8,9,10).
    # Or maybe "3~10" includes 3? If so, 3 is double counted.
    # Standard interpretation: 2-3 (Small), 4-10 (Medium), >10 (Large).
    # Or maybe 2-3 means 2 <= x <= 3. And 3-10 means 3 < x <= 10?
    # Let's assume partitions: [2, 3], [4, 10], [11, inf).
    # Also "Single nodes/edges": [0, 1].
    
    c_0_1 = 0
    c_2_3 = 0
    c_4_10 = 0 # Adjusted to 4-10 to avoid overlap with 2-3
    c_10_plus = 0 # > 10
    
    # Wait, the table headers are "Raw Node Num": "2 ~ 3", "3 ~ 10", "10+".
    # This notation usually implies overlaps or vague boundaries. 
    # But percentages must sum to... well, maybe not 100% if "Atomic" is excluded from the table?
    # Table R1 caption: "Distribution of pattern lengths...".
    # Text says: "approximately XX% ... single nodes/edges, YY% ... small motifs (2-3 nodes), and even ZZ% ... larger subgraph patterns."
    # It seems "YY%" corresponds to "2~3". "ZZ%" corresponds to "larger" (maybe 3+? or 10+?).
    # The table has 3 columns for ranges.
    # I will categorize as:
    # Size 1 (and 0?): Atomic
    # Size 2-3: Small
    # Size 4-10: Medium (Corresponds to 3~10 column, assuming 3 is covered by 2-3? or maybe 3 is in both? No, that's bad stats. I'll put 3 in 2-3).
    # Size >10: Large
    
    for n in atom_counts:
        if n <= 1:
            c_0_1 += 1
        elif 2 <= n <= 3:
            c_2_3 += 1
        elif 4 <= n <= 10:
            c_4_10 += 1
        else:
            c_10_plus += 1
            
    p_0_1 = (c_0_1 / total_vocab) * 100
    p_2_3 = (c_2_3 / total_vocab) * 100
    p_4_10 = (c_4_10 / total_vocab) * 100
    p_10_plus = (c_10_plus / total_vocab) * 100
    
    print("-" * 30)
    print("Statistics:")
    print(f"Total Vocab Size: {total_vocab}")
    print(f"Size <= 1 (Atomic): {c_0_1} ({p_0_1:.2f}%)")
    print(f"Size 2-3 (Small): {c_2_3} ({p_2_3:.2f}%)")
    print(f"Size 4-10 (Medium): {c_4_10} ({p_4_10:.2f}%)")
    print(f"Size 10+ (Large): {c_10_plus} ({p_10_plus:.2f}%)")
    print(f"Sum percentages: {p_0_1 + p_2_3 + p_4_10 + p_10_plus:.2f}%")
    
    print(f"\nDomain Alignment Checks (Preliminary):")
    print(f"Exact Benzene (c1ccccc1) tokens: {found_benzene}")
    print(f"Exact Carboxyl (OC=O) tokens: {found_carboxyl}")
    
    # Output for LaTeX
    print("\n--- LaTeX Table Data ---")
    print(f"Percentage of Vocabulary & {p_2_3:.1f}\\% & {p_4_10:.1f}\\% & {p_10_plus:.1f}\\% \\\\")
    
    print("\n--- Text Data ---")
    print(f"Atomic (XX%): {p_0_1:.1f}%")
    print(f"Small (YY%): {p_2_3:.1f}%")
    print(f"Larger (ZZ% - using 10+? Or sum of 4-10 and 10+?): Text says 'ZZ% correspond to larger subgraph patterns'.") 
    # If the text has 3 placeholders (XX, YY, ZZ) and describes "single", "2-3", "larger".
    # "Larger" could mean >3 (i.e. 4-10 + 10+).
    # Let's calculate that too.
    p_larger = p_4_10 + p_10_plus
    print(f"Larger (>3) (ZZ% candidate): {p_larger:.1f}%")
    
    # Histogram Data
    # We can save the counts to a csv or just print them for the user to plot or I can generate a plot image.
    # The user said: "need to draw token size distribution histogram".
    # I will generate a PNG histogram.
    
    plt.figure(figsize=(10, 6))
    plt.hist(atom_counts, bins=range(0, max(atom_counts)+2), align='left', rwidth=0.8)
    plt.title('BPE Token Size Distribution (Number of Heavy Atoms)')
    plt.xlabel('Number of Heavy Atoms')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5)
    
    hist_path = PROJECT_ROOT / 'final/exp4_bpe_vocab_visual/token_size_distribution.png'
    plt.savefig(hist_path)
    print(f"\nHistogram saved to {hist_path}")
    
    # Save raw data for histogram
    hist_data_path = PROJECT_ROOT / 'final/exp4_bpe_vocab_visual/token_size_data.csv'
    with open(hist_data_path, 'w') as f:
        f.write("size,count\n")
        for s, c in sorted(count_stats.items()):
            f.write(f"{s},{c}\n")
    print(f"Histogram data saved to {hist_data_path}")

if __name__ == "__main__":
    analyze_vocabulary()

