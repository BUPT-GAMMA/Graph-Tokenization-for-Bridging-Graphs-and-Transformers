"""
BPE compression for integer token sequences.

This file serves as the Python backend fallback for BPEEngine.
The C++ backend (cpp_bpe_backend.py) is recommended for better performance.

Optimizations:
- Incremental frequency updates: avoids recomputing all pair frequencies, ~4x speedup
- Integrated merge: single pass for sequence merge and frequency update
- In-place modification via del

Kept as a fallback/compatibility option; primary path is the C++ backend.
"""

from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)

def get_stats(ids, counts=None):
        """
        Given a list of integers, return a dictionary of counts of consecutive pairs
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]): # iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
  
class StandardBPECompressor:
    """
    BPE compressor for integer token sequences.
    All inputs must be List[List[int]] or List[int].
    
    Key optimizations:
    - Incremental pair frequency maintenance (~4x speedup)
    - Integrated merge + frequency update in single pass
    - In-place sequence modification
    """
    def __init__(self, num_merges: int = 10000, min_frequency: int = 20, debug: bool = False):
        self.num_merges = num_merges
        self.min_frequency = min_frequency
        self.debug = debug
        self.token_to_id: Dict[Any, int] = {}
        self.id_to_token: Dict[int, Any] = {}
        self.next_id = 0
        self.merge_rules: List[Tuple[int, int, int]] = []
        self.training_stats = {}
        # Maintain pair frequency table incrementally
        self.pair_freqs = defaultdict(int)

    def _check_token_list(self, seq):
        if not isinstance(seq, list):
            raise TypeError("BPE input must be a list (token id sequence), not a string.")
        for token in seq:
            if not isinstance(token, int):
                raise TypeError("BPE only supports list of int token IDs. Convert strings to token ID lists first.")

    def _check_token_sequences(self, seqs):
        if not isinstance(seqs, list):
            raise TypeError("BPE input must be list[list[int]], not a string.")
        for seq in seqs:
            self._check_token_list(seq)

    def train(self, token_sequences: List[List[int]]) -> Dict[str, Any]:
        """
        Train BPE model. token_sequences must be List[List[int]].
        
        Args:
            token_sequences: Token sequence list to train on
            
        Returns:
            Dict[str, Any]: Training statistics
        """
        self._check_token_sequences(token_sequences)
        
        if not token_sequences:
            raise ValueError("Training sequences empty")
        
        if self.debug:
            logger.debug(f"Training started, {len(token_sequences)} sequences")
        
        logger.info(f"BPE config: num_merges={self.num_merges}, min_frequency={self.min_frequency}")
        self._build_base_vocab(token_sequences)
        
        if len(self.token_to_id) < 2:
            raise ValueError("Base vocabulary too small for merging")
        logger.info(f"Base vocab size: {len(self.token_to_id)}")
        
        if self.debug:
            self._print_vocab()
        
        logger.info("Converting to ID sequences")
        id_sequences = self._convert_to_id_sequences(token_sequences)
        
        logger.info("Initializing pair frequencies")
        self._initialize_pair_frequencies(id_sequences)
        
        logger.info("Running optimized BPE merges")
        merge_count = 0
        for merge_iter in range(self.num_merges):
            # Use maintained frequency table
            valid_pairs = {pair: freq for pair, freq in self.pair_freqs.items() if freq >= self.min_frequency}
            
            if not valid_pairs:  # Normal termination: no pairs meet frequency threshold
                logger.info(
                    f"BPE stopped: no pairs above threshold [{self.min_frequency}], "
                    f"pairs={len(self.pair_freqs)}, vocab={len(self.token_to_id)}, merges={merge_count}"
                )
                break
            
            try:
                best_pair = max(valid_pairs, key=valid_pairs.get)
                best_freq = valid_pairs[best_pair]
            except ValueError as e:
                raise ValueError(f"Failed to select best pair: {str(e)}")
            
            new_id = self.next_id
            self.next_id += 1
            self.merge_rules.append((best_pair[0], best_pair[1], new_id))
            merged_token = (self.id_to_token[best_pair[0]], self.id_to_token[best_pair[1]])
            self.token_to_id[merged_token] = new_id
            self.id_to_token[new_id] = merged_token
            # Integrated merge and frequency update
            self._optimized_merge_and_update(id_sequences, best_pair, new_id)
            merge_count += 1
            
            if self.debug and (merge_iter + 1) % 50 == 0:
                logger.debug(f"Merge progress {merge_iter + 1}/{self.num_merges}, vocab: {len(self.token_to_id)}, best_freq: {best_freq}")
            if self.debug and merge_iter < 5:
                left_token = self.id_to_token[best_pair[0]]
                right_token = self.id_to_token[best_pair[1]]
                logger.debug(f"Merge {merge_iter}: {left_token} + {right_token} -> ID{new_id} (freq:{best_freq})")
        
        if merge_count == 0:
            raise ValueError("No merges performed")
        
        logger.info(f"BPE training done: {merge_count} merges, vocab={len(self.token_to_id)}, min_freq={self.min_frequency}")
        
        self._print_final_codebook()
        # Build fast lookup for minBPE-style encoding (pair -> rank, pair -> new_id)
        self._minbpe_pair_to_rank = { (l, r): idx for idx, (l, r, nid) in enumerate(self.merge_rules) }
        self._minbpe_pair_to_newid = { (l, r): nid for (l, r, nid) in self.merge_rules }
        return {
            'num_merges_performed': merge_count,
            'final_vocab_size': len(self.token_to_id),
            'merge_rules_count': len(self.merge_rules)
        }

    def _encode(self, token_sequence: List[int]) -> List[int]:
        """
        Encode token sequence to ID list. Applies all merge rules sequentially.
        
        Args:
            token_sequence: Token sequence (List[int])
            
        Returns:
            List[int]: Encoded ID sequence
        """
        self._check_token_list(token_sequence)
        id_seq = []
        for token in token_sequence:
            if token in self.token_to_id:
                id_seq.append(self.token_to_id[token])
            else:
                raise ValueError(f"Unknown token: {token}")
        for left_id, right_id, merged_id in self.merge_rules:
            id_seq = self._apply_single_merge(id_seq, left_id, right_id, merged_id)
        return id_seq

    def _encode_minbpe_style(self, token_sequence: List[int]) -> List[int]:
        """
        minBPE-style encode: each round picks the lowest-rank mergeable pair
        from the current sequence and applies a non-overlapping replace.
        """
        self._check_token_list(token_sequence)
        # Base ID sequence
        id_seq: List[int] = []
        for token in token_sequence:
            if token in self.token_to_id:
                id_seq.append(self.token_to_id[token])
            else:
                raise ValueError(f"Unknown token: {token}")
        while len(id_seq) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(id_seq)
            pair = min(stats, key=lambda p: self._minbpe_pair_to_rank.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self._minbpe_pair_to_newid:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self._minbpe_pair_to_newid[pair]
            id_seq = merge(id_seq, pair, idx)
        return id_seq
        # Alternative: exact minBPE logic (uses only adjacent pair keys, no freq)
        # while len(id_seq) >= 2:
        #     best_pair = None
        #     best_rank = float("inf")
        #     for pair in zip(id_seq, id_seq[1:]):
        #         rank = self._minbpe_pair_to_rank.get(pair)
        #         if rank is not None and rank < best_rank:
        #             best_rank = rank
        #             best_pair = pair
        #     if best_pair is None:
        #         break
        #     new_id = self._minbpe_pair_to_newid[best_pair]
        #     id_seq = self._apply_single_merge(id_seq, best_pair[0], best_pair[1], new_id)
        # return id_seq

    def encode(self, token_sequence: List[int]) -> List[int]:
        if len(token_sequence) < 150:
          return self._encode_minbpe_style(token_sequence)
        else:
          return self._encode(token_sequence)

    # def encode_minbpe_all(self, sequences: List[List[int]]) -> List[List[int]]:
    #     return [self.encode_minbpe_style(seq) for seq in sequences]

    def decode(self, id_sequence: List[int]) -> List[int]:
        """
        Decode ID list back to token sequence.
        
        Args:
            id_sequence: ID sequence to decode
            
        Returns:
            List[int]: Decoded token sequence
        """
        self._check_token_list(id_sequence)
        result = []
        for id_val in id_sequence:
            if id_val in self.id_to_token:
                token = self.id_to_token[id_val]
                expanded = self._expand_token(token)
                result.extend(expanded)
            else:
                raise ValueError(f"Unknown ID: {id_val}")
        return result

    def calculate_compression_stats(self, original_sequences: List[List[int]]) -> Dict[str, Any]:
        """Compute compression statistics."""
        self._check_token_sequences(original_sequences)
        if self.debug:
            logger.debug("Computing compression stats")
        total_original_tokens = sum(len(seq) for seq in original_sequences)
        total_compressed_tokens = 0
        correct_decoding = 0
        for i, seq in enumerate(original_sequences):
            encoded = self.encode(seq)
            decoded = self.decode(encoded)
            total_compressed_tokens += len(encoded)
            if decoded == seq:
                correct_decoding += 1
            if self.debug and (i + 1) % 1000 == 0:
                current_ratio = total_compressed_tokens / total_original_tokens if total_original_tokens > 0 else 1.0
                logger.debug(f"Stats progress {i+1}/{len(original_sequences)}, ratio: {current_ratio:.3f}")
        compression_ratio = total_compressed_tokens / total_original_tokens if total_original_tokens > 0 else 1.0
        stats = {
            'original_token_count': total_original_tokens,
            'compressed_token_count': total_compressed_tokens,
            'compression_ratio': compression_ratio,
            'tokens_saved': total_original_tokens - total_compressed_tokens,
            'correct_decoding_count': correct_decoding,
            'total_sequences': len(original_sequences),
            'accuracy': correct_decoding / len(original_sequences) if original_sequences else 0.0
        }
        if self.debug:
            logger.debug(f"Compression stats - original: {total_original_tokens}, compressed: {total_compressed_tokens}")
            logger.debug(f"Ratio: {compression_ratio:.4f}, saved: {stats['tokens_saved']} tokens")
            logger.debug(f"Correct decoding: {correct_decoding}/{len(original_sequences)} ({stats['accuracy']*100:.1f}%)")
        return stats

    def show_examples(self, sequences: List[List[int]], num_examples: int = 3) -> List[Dict[str, Any]]:
        """Show compression examples."""
        self._check_token_sequences(sequences)
        examples = []
        logger.info(f"Showing first {num_examples} compression examples")
        for i, seq in enumerate(sequences[:num_examples]):
            encoded = self.encode(seq)
            decoded = self.decode(encoded)
            is_correct = (decoded == seq)
            example = {
                'index': i,
                'original': seq,
                'encoded': encoded,
                'decoded': decoded,
                'original_length': len(seq),
                'compressed_length': len(encoded),
                'compression_ratio': len(encoded) / len(seq) if len(seq) > 0 else 1.0,
                'is_correct': is_correct
            }
            logger.debug(f"Example {i}: {seq} -> {encoded} -> {decoded} [{chr(10003) if is_correct else chr(10007)}]")
            examples.append(example)
        return examples

    # === Internal methods ===
    def _initialize_pair_frequencies(self, id_sequences: List[List[int]]):
        """Initialize pair frequency stats."""
        self.pair_freqs.clear()
        for id_seq in id_sequences:
            for i in range(len(id_seq) - 1):
                pair = (id_seq[i], id_seq[i + 1])
                self.pair_freqs[pair] += 1
    
    def _optimized_merge_and_update(self, id_sequences: List[List[int]], merge_pair: Tuple[int, int], new_id: int):
        """Optimized merge + frequency update in a single pass."""
        left_id, right_id = merge_pair
        frequency_changes = defaultdict(int)
        
        # Single pass: merge sequences and record frequency changes
        for seq in id_sequences:
            i = 0
            while i < len(seq) - 1:
                if seq[i] == left_id and seq[i + 1] == right_id:
                    # Record frequency changes (before modification)
                    frequency_changes[(left_id, right_id)] -= 1
                    
                    # Handle left neighbor
                    if i > 0:
                        left_neighbor = seq[i-1]
                        frequency_changes[(left_neighbor, left_id)] -= 1
                        frequency_changes[(left_neighbor, new_id)] += 1
                    
                    # Handle right neighbor
                    if i + 2 < len(seq):
                        right_neighbor = seq[i+2]
                        frequency_changes[(right_id, right_neighbor)] -= 1
                        frequency_changes[(new_id, right_neighbor)] += 1
                    
                    # Apply merge (in-place)
                    seq[i] = new_id
                    del seq[i + 1]
                    
                    # Don't advance i; check same position again
                else:
                    i += 1
        
        # Batch-apply frequency changes
        for pair, change in frequency_changes.items():
            self.pair_freqs[pair] += change
            if self.pair_freqs[pair] <= 0:
                del self.pair_freqs[pair]
    
    def _build_base_vocab(self, token_sequences: List[List[int]]):
        unique_tokens = set()
        for seq in token_sequences:
            unique_tokens.update(seq)
        for token in sorted(unique_tokens):
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1

    def _convert_to_id_sequences(self, token_sequences: List[List[int]]) -> List[List[int]]:
        id_sequences = []
        for seq in token_sequences:
            id_seq = [self.token_to_id[token] for token in seq]
            id_sequences.append(id_seq)
        return id_sequences

    def _count_pair_frequencies(self, id_sequences: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """Legacy frequency counting, superseded by _initialize_pair_frequencies."""
        pair_freqs = defaultdict(int)
        for id_seq in id_sequences:
            for i in range(len(id_seq) - 1):
                pair = (id_seq[i], id_seq[i + 1])
                pair_freqs[pair] += 1
        return dict(pair_freqs)

    def _apply_merge_to_sequences(self, id_sequences: List[List[int]], pair: Tuple[int, int], new_id: int) -> List[List[int]]:
        """Legacy merge method, superseded by _optimized_merge_and_update."""
        new_sequences = []
        for id_seq in id_sequences:
            new_seq = self._apply_single_merge(id_seq, pair[0], pair[1], new_id)
            new_sequences.append(new_seq)
        return new_sequences

    def _apply_single_merge(self, id_seq: List[int], left_id: int, right_id: int, new_id: int) -> List[int]:
        new_seq = []
        i = 0
        while i < len(id_seq):
            if (i < len(id_seq) - 1 and id_seq[i] == left_id and id_seq[i + 1] == right_id):
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(id_seq[i])
                i += 1
        return new_seq

    def _expand_token(self, token: Any) -> List[int]:
        if isinstance(token, tuple) and len(token) == 2:
            left_expanded = self._expand_token(token[0])
            right_expanded = self._expand_token(token[1])
            return left_expanded + right_expanded
        else:
            return [token]

    def _print_vocab(self):
        if self.debug:
            logger.debug("Current vocabulary:")
            for token, id_val in list(self.token_to_id.items())[:10]:
                logger.debug(f"  {token} -> ID{id_val}")
            if len(self.token_to_id) > 10:
                logger.debug(f"  ... {len(self.token_to_id) - 10} more tokens")

    def _print_final_codebook(self):
        if self.debug:
            logger.debug("Final codebook:")
            logger.debug(f"Total vocab: {len(self.token_to_id)}")
            logger.debug(f"Merge rules: {len(self.merge_rules)}")
            logger.debug("Base tokens:")
            base_tokens = [(token, id_val) for token, id_val in self.token_to_id.items() if not isinstance(token, tuple)]
            for token, id_val in base_tokens[:5]:
                logger.debug(f"  {token} -> ID{id_val}")
            logger.debug("Merged tokens:")
            merged_tokens = [(token, id_val) for token, id_val in self.token_to_id.items() if isinstance(token, tuple)]
            for token, id_val in merged_tokens[:5]:
                logger.debug(f"  {token} -> ID{id_val}")
            if len(merged_tokens) > 5:
                logger.debug(f"  ... {len(merged_tokens) - 5} more merged tokens")
    
    def save(self, path: str):
        """
        Save BPE model to file.
        
        Args:
            path: Save path
        """
        import pickle
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model data
        model_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'merge_rules': self.merge_rules,
            'next_id': self.next_id,
            'num_merges': self.num_merges,
            'min_frequency': self.min_frequency,
            'debug': self.debug
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.debug:
            logger.debug(f"BPE model saved to: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'StandardBPECompressor':
        """
        Load BPE model from file.
        
        Args:
            path: Model file path
            
        Returns:
            StandardBPECompressor
        """
        import pickle
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        instance = cls(
            num_merges=model_data['num_merges'],
            min_frequency=model_data['min_frequency'],
            debug=model_data['debug']
        )
        
        # Restore model state
        instance.token_to_id = model_data['token_to_id']
        instance.id_to_token = model_data['id_to_token']
        instance.merge_rules = model_data['merge_rules']
        instance.next_id = model_data['next_id']
        
        if instance.debug:
            logger.debug(f"BPE model loaded from {path}")
        
        return instance 