import json
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def save_pretrained_alibaba_model(
    model_dir: Path,
    model_name_or_path="Alibaba-NLP/gte-multilingual-base",
):
    """Save pretrained Alibaba model `model_name_or_path` to local directory `model_dir` for offline use.
    Refer to: https://huggingface.co/Alibaba-NLP/new-impl/discussions/2#662b08d04d8c3d0a09c88fa3

    NOTE: After it is downloaded, trust_remote_code=True is still required but will be offline.

    """
    from huggingface_hub import hf_hub_download

    pth_config = model_dir / "config.json"

    model_name_or_path = "Alibaba-NLP/gte-multilingual-base"
    # Download the tokenizer and model (internet required)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Overwrite config
    cfg = json.loads(pth_config.read_text())
    cfg["auto_map"] = {
        "AutoConfig": "configuration.NewConfig",
        "AutoModel": "modeling.NewModel",
        "AutoModelForMaskedLM": "modeling.NewForMaskedLM",
        "AutoModelForMultipleChoice": "modeling.NewForMultipleChoice",
        "AutoModelForQuestionAnswering": "modeling.NewForQuestionAnswering",
        "AutoModelForSequenceClassification": "modeling.NewForSequenceClassification",
        "AutoModelForTokenClassification": "modeling.NewForTokenClassification",
    }
    pth_config.write_text(json.dumps(cfg))

    # Download the relevant files
    hf_hub_download(
        repo_id="Alibaba-NLP/new-impl",
        filename="modeling.py",
        local_dir=model_dir.as_posix(),
    )
    hf_hub_download(
        repo_id="Alibaba-NLP/new-impl",
        filename="configuration.py",
        local_dir=model_dir.as_posix(),
    )
    
if __name__ == "__main__":
    save_pretrained_alibaba_model(Path("gte_model"))
  