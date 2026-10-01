import json
from pathlib import Path

import structlog

logger = structlog.get_logger()


def download_hf_file_cache_first(
    repo_id: str, filename: str, revision: str | None = None
) -> Path:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached_path = try_to_load_from_cache(
        repo_id,
        filename,
        revision=revision,
    )
    if isinstance(cached_path, str):
        logger.debug("Using cached Hugging Face file", repo_id=repo_id, filename=filename)
        return Path(cached_path)

    logger.info("Hugging Face file is missing from cache; downloading", repo_id=repo_id, filename=filename)
    return Path(hf_hub_download(repo_id, filename, revision=revision))


def resolve_hf_reference(
    hf_repo: str, reference: str | Path, revision: str | None = None
) -> Path:
    if isinstance(reference, Path):
        return reference
    if reference.startswith("file://"):
        return Path(reference.removeprefix("file://"))

    repo_id = hf_repo
    filename = reference
    if reference.startswith("hf://"):
        parts = reference.removeprefix("hf://").split("/", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid Hugging Face reference: {reference}")
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2]
    return download_hf_file_cache_first(repo_id, filename, revision)


def checkpoint_info_cache_first(hf_repo: str):
    from huggingface_hub.errors import EntryNotFoundError
    from moshi.models import loaders
    from moshi.models.loaders import CheckpointInfo

    try:
        config_path = resolve_hf_reference(hf_repo, "config.json")
    except EntryNotFoundError:
        return CheckpointInfo.from_hf_repo(hf_repo)

    config = json.loads(config_path.read_text())
    moshi_weights = resolve_hf_reference(
        hf_repo, config.get("moshi_name", loaders.MOSHI_NAME)
    )
    mimi_weights = resolve_hf_reference(
        hf_repo, config.get("mimi_name", loaders.MIMI_NAME)
    )
    tokenizer = resolve_hf_reference(
        hf_repo, config.get("tokenizer_name", loaders.TEXT_TOKENIZER_NAME)
    )
    mimi_config_name = config.get("mimi_config_name")
    mimi_config_path = (
        resolve_hf_reference(hf_repo, mimi_config_name)
        if mimi_config_name is not None
        else None
    )
    lora_name = config.get("lora_name")
    lora_weights = (
        resolve_hf_reference(hf_repo, lora_name)
        if lora_name is not None
        else None
    )

    return CheckpointInfo.from_hf_repo(
        hf_repo,
        moshi_weights=moshi_weights,
        mimi_weights=mimi_weights,
        tokenizer=tokenizer,
        config_path=config_path,
        mimi_config_path=mimi_config_path,
        lora_weights=lora_weights,
    )