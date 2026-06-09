"""upload_hf.py — push a merged model from the Modal artifacts volume to the HuggingFace Hub.

Uses the `huggingface` Modal secret (HF_TOKEN). The model card (README.md) is added locally to the
image and copied into the model dir before upload.

  modal run train/upload_hf.py --model-dir /artifacts/qwen3-8b-toolbase-merged \
      --repo-id dennisonb/reversible-circuit-8b-tool
"""
from pathlib import Path
import modal

app = modal.App("ecdsa-hf-upload")
HERE = Path(__file__).parent.resolve()
CARD = (HERE.parent / "artifacts" / "MODEL_CARD_8B.md").resolve()

img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("huggingface_hub>=0.25", "hf_transfer")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
       .add_local_file(str(CARD), remote_path="/root/README.md"))

artifacts = modal.Volume.from_name("ecdsa-artifacts", create_if_missing=True)


@app.function(image=img, timeout=3600, volumes={"/artifacts": artifacts},
              secrets=[modal.Secret.from_name("huggingface")])
def upload(model_dir: str, repo_id: str):
    import os, shutil
    from huggingface_hub import HfApi
    tok = os.environ["HF_TOKEN"]
    api = HfApi(token=tok)
    # place the honest model card as README.md inside the model dir
    try:
        shutil.copy("/root/README.md", os.path.join(model_dir, "README.md"))
    except Exception as e:
        print("[upload] WARN copying README:", e)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"[upload] uploading {model_dir} -> {repo_id} ...", flush=True)
    api.upload_folder(folder_path=model_dir, repo_id=repo_id, repo_type="model",
                      commit_message="reversible-circuit-8b-tool: SFT base + honest model card")
    url = f"https://huggingface.co/{repo_id}"
    print(f"HF_UPLOAD_DONE {url}", flush=True)
    return url


@app.local_entrypoint()
def main(model_dir: str = "/artifacts/qwen3-8b-toolbase-merged",
         repo_id: str = "dennisonb/reversible-circuit-8b-tool"):
    print(upload.remote(model_dir=model_dir, repo_id=repo_id))
