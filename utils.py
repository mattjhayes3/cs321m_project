from typing import List
import modal
import time
from interfaces import Question
from evaluate_generated import app, MODEL_REGISTRY, MODEL_SUBSET
from modal_image import loop_image

# Persistent model cache volume
hf_cache_vol = modal.Volume.from_name("hf-cache-volume", create_if_missing=True)

def format_option(idx: int, text: str) -> str:
    """
    Formats a multiple-choice option consistently with uppercase letters.
    e.g., format_option(0, "option text") -> "(A) option text"
    """
    return f"({chr(65 + idx)}) {text}"

def format_question(
    question: Question, 
    include_answer: bool = False, 
    extra_options: List[str] = None
) -> str:
    """
    Unified formatting utility to represent a Question object consistently for LLM prompts.
    Represents option choices and the correct answer choice consistently using the format_option helper.
    """
    options = list(question.options)
    if extra_options:
        options.extend(extra_options)

    options_str = "\n".join([
        f"  {format_option(i, opt)}" for i, opt in enumerate(options)
    ])

    blocks = [
        f"Question:\n{question.question_text}",
        f"Options:\n{options_str}"
    ]

    if include_answer:
        ans_key = question.correct_answer.strip().upper()
        if len(ans_key) != 1 or not ("A" <= ans_key <= "Z"):
            raise ValueError(
                f"Invalid correct answer key format in Question. Expected a single uppercase letter (A-Z), "
                f"got: '{ans_key}' (Question ID: {question.id})"
            )
        
        idx = ord(ans_key) - 65
        if 0 <= idx < len(question.options):
            ans_text = question.options[idx]
            blocks.append(f"Correct Answer: {format_option(idx, ans_text)}")
        else:
            raise ValueError(
                f"Correct answer choice letter '{ans_key}' (index {idx}) is out of bounds "
                f"for the options array of length {len(question.options)}: {question.options} "
                f"(Question ID: {question.id})"
            )

    return "\n\n".join(blocks)


# ────────────────────────────────────────────────────────────────
# CHEAP CPU PRE-DOWNLOADER
# ────────────────────────────────────────────────────────────────

@app.function(
    image=loop_image,
    volumes={
        "/hf_cache": hf_cache_vol
    },
    env={"HF_HOME": "/hf_cache", "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=14400,  # 4 hours max
)
def pre_download_models_remote():
    """Pre-download all models in MODEL_SUBSET to the persistent cache volume on a cheap CPU worker."""
    import os
    import time
    from huggingface_hub import snapshot_download, login

    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"], add_to_git_credential=False)

    # Clean up stale locks to prevent FileExistsError during snapshot downloads
    locks_dir = "/hf_cache/hub/.locks"
    if os.path.lexists(locks_dir):
        print(f"🧹 Cleaning up stale HF download locks directory/link: {locks_dir}")
        import shutil
        try:
            if os.path.islink(locks_dir):
                os.unlink(locks_dir)
            else:
                shutil.rmtree(locks_dir, ignore_errors=True)
        except Exception as del_err:
            print(f"  ⚠️  Failed to delete locks directory: {del_err}")

    print(f"\n=== PRE-DOWNLOADING {len(MODEL_SUBSET)} MODELS TO CACHE VOLUME ===")
    start_time = time.time()

    for i, model_short_name in enumerate(MODEL_SUBSET):
        reg = MODEL_REGISTRY.get(model_short_name)
        if not reg:
            print(f"  ❌ Unknown model: {model_short_name}")
            continue
        hf_id = reg["hf_id"]

        # HF cache dir for this model: /hf_cache/hub/models--{org}--{model}
        cache_dir_name = f"models--{hf_id.replace('/', '--')}"
        cache_path = os.path.join("/hf_cache", "hub", cache_dir_name)

        print(f"\n[{i+1}/{len(MODEL_SUBSET)}] Downloading: {model_short_name} ({hf_id})...")
        t0 = time.time()
        try:
            snapshot_download(
                repo_id=hf_id,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                local_files_only=False,
            )
            print(f"  ✅ Success in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  ⚠️  First attempt failed: {e}")
            # Check for corrupted cache dir (missing blobs/ or refs/, or broken symlinks)
            if os.path.lexists(cache_path):
                print(f"  🔧 Removing corrupted cache entry/link: {cache_path}")
                import shutil
                try:
                    if os.path.islink(cache_path):
                        os.unlink(cache_path)
                    else:
                        shutil.rmtree(cache_path, ignore_errors=True)
                except Exception as del_err:
                    print(f"  ⚠️  Failed to delete {cache_path}: {del_err}")
                
                # Retry
                try:
                    snapshot_download(
                        repo_id=hf_id,
                        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                        local_files_only=False,
                    )
                    print(f"  ✅ Retry success in {time.time() - t0:.1f}s")
                except Exception as e2:
                    print(f"  ❌ Retry also failed for {hf_id}: {e2}")
            else:
                print(f"  ❌ Failed to download {hf_id}: {e}")

    total_time = time.time() - start_time
    print(f"\n=== PRE-DOWNLOAD COMPLETE: {len(MODEL_SUBSET)} models in {total_time/60:.1f} minutes ===")
    
    print("Committing all changes to hf-cache-volume to persist model downloads...")
    hf_cache_vol.commit()


@app.local_entrypoint()
def pre_download():
    """Kicks off the CPU pre-downloader."""
    print("Launching cheap CPU pre-downloader on Modal...")
    pre_download_models_remote.remote()
