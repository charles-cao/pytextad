"""Upload the PyTextAD datasets to their Hugging Face dataset repo (maintainers only).

    hf auth login                      # once, with a write token
    python scripts/upload_to_hub.py <folder with the .jsonl files>

Checks every file against the SHA-256 in pytextad/datasets.py, creates the dataset repo if
it does not exist, uploads the files to data/ and scripts/hub_README.md as the dataset card,
downloads the files again and re-checks them, and prints the commit id to put in
DATASETS[...]["revision"].
"""
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi, hf_hub_download

from pytextad.datasets import DATASETS, _sha256, load_dataset


def main(folder):
    repos = {m["repo_id"] for m in DATASETS.values()}
    assert len(repos) == 1, repos
    repo = repos.pop()
    card = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hub_README.md")
    with tempfile.TemporaryDirectory() as tmp:          # repo layout: README.md, data/*.jsonl
        for name, m in DATASETS.items():
            src = os.path.join(folder, os.path.basename(m["filename"]))
            got = _sha256(src)
            assert got == m["sha256"], f"{src}: SHA-256 {got} does not match pytextad/datasets.py"
            dst = os.path.join(tmp, m["filename"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        shutil.copyfile(card, os.path.join(tmp, "README.md"))
        api = HfApi()
        api.create_repo(repo, repo_type="dataset", exist_ok=True)
        info = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=tmp,
                                 commit_message="Add PyTextAD datasets")
    commit = getattr(info, "oid", None) or str(info)
    for name, m in DATASETS.items():
        p = hf_hub_download(repo, m["filename"], repo_type="dataset", revision=commit, force_download=True)
        assert _sha256(p) == m["sha256"], f"{name}: uploaded file differs"
        print(name, "ok:", load_dataset(name))
    print("commit:", commit)


if __name__ == "__main__":
    main(sys.argv[1])
