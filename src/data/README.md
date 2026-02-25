# Data Layer

The data layer handles dataset loading, serialized sequence access, and BPE codebook management. Everything goes through a single interface (`UnifiedDataInterface`).

Key design choices:
- **Fixed splits**: train/val/test indices are stored as JSON files — no random splitting at runtime.
- **Read-only interface**: UDI only reads pre-built artifacts. Run `prepare_data_new.py` first to generate them.
- **Fail-fast**: missing files raise errors immediately; no silent fallbacks.

## Components

- **`UnifiedDataInterface`** (`unified_data_interface.py`) — main entry point for accessing sequences, labels, vocabs, and BPE codebooks
- **`BaseDataLoader`** (`base_loader.py`) — abstract base class for dataset loaders
- **`UnifiedDataFactory`** (`unified_data_factory.py`) — registry that maps dataset names to loader classes

## Supported Datasets

Loaders are in `src/data/loader/`. Currently registered:

| Dataset | Loader | Type |
|---------|--------|------|
| `qm9` | `qm9_loader.py` | Molecular (regression) |
| `qm9test` | `qm9test_loader.py` | QM9 subset (~10%) |
| `zinc` | `zinc_loader.py` | Molecular (regression) |
| `aqsol` | `aqsol_loader.py` | Solubility (regression) |
| `mutagenicity` | `mutagenicity_loader.py` | Mutagenicity (classification) |
| `proteins` | `proteins_loader.py` | Protein (classification) |
| `dd` | `dd_loader.py` | D&D protein (classification) |
| `peptides_func` | `peptides_func_loader.py` | Peptide function (multi-label) |
| `peptides_struct` | `peptides_struct_loader.py` | Peptide structure (multi-target regression) |
| `molhiv` | `molhiv_loader.py` | HIV activity (classification) |
| `mnist` | `mnist_loader.py` | MNIST superpixel graphs |
| `dblp`, `code2`, `coildel`, `colors3`, `twitter` | respective loaders | Various graph tasks |
| `synthetic` | `synthetic_loader.py` | Synthetic graphs for testing |

## Directory Layout

```
data/
├── <dataset>/
│   ├── data.pkl              # Graph data (required)
│   ├── train_index.json      # Train split indices (required)
│   ├── val_index.json        # Val split indices (required)
│   └── test_index.json       # Test split indices (required)
│
└── processed/
    └── <dataset>/
        ├── serialized_data/<method>/single/
        │   └── serialized_data.pickle
        └── vocab/<method>/bpe/single/
            └── vocab.json
```

## Usage

```python
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface

cfg = ProjectConfig()
udi = UnifiedDataInterface(cfg, "qm9test")

# Flat sequences for pre-training
train_seq, val_seq, test_seq = udi.get_training_data_flat(method="feuler")

# Sequences with labels for fine-tuning
(train_seqs, train_props), (val_seqs, val_props), (test_seqs, test_props) = \
    udi.get_training_data(method="feuler")

# Vocab and BPE
vocab_manager = udi.get_vocab(method="feuler")
```

## Adding a New Dataset

1. Create a loader class inheriting from `BaseDataLoader` in `src/data/loader/`
2. Implement `_load_processed_data`, `_extract_labels`, `get_node_attribute`, `get_edge_attribute`, etc.
3. Register it in `unified_data_factory.py`
4. Place data files under `data/<dataset_name>/`
