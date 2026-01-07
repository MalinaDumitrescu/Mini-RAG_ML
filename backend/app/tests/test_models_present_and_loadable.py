from pathlib import Path

from backend.app.core.paths import FINETUNED_DIR, MODELS_DIR

def test_finetuned_embeddings_exist():
    assert FINETUNED_DIR.exists(), "FINETUNED_DIR missing. Run scripts/train_embeddings.py"
    assert (FINETUNED_DIR / "config.json").exists() or (FINETUNED_DIR / "modules.json").exists(), \
        "Finetuned embeddings look incomplete."

def test_lora_adapter_exists():
    lora_dir = MODELS_DIR / "llm_lora_qwen05b"
    assert lora_dir.exists(), "LoRA dir missing. Run train_llm_lora.py"
    assert (lora_dir / "adapter_config.json").exists(), "adapter_config.json missing (LoRA incomplete)"
    assert (lora_dir / "adapter_model.safetensors").exists() or (lora_dir / "adapter_model.bin").exists(), \
        "LoRA weights missing"
