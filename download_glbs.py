import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from tqdm import tqdm


def ensure_can_write(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def is_direct_glb(url):
    if not url:
        return False
    parsed = urlparse(url)
    if "sketchfab.com" in parsed.netloc.lower() or "fab.com" in parsed.netloc.lower():
        return False
    return parsed.path.lower().endswith((".glb", ".gltf"))


def main():
    parser = argparse.ArgumentParser(description="Download only direct GLB/GLTF URLs from a manifest.")
    parser.add_argument("--manifest", default="outputs/manifest.csv")
    parser.add_argument("--out_dir", default="glbs")
    parser.add_argument("--status_out", default="outputs/download_status.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    status_path = Path(args.status_out)
    ensure_can_write(status_path, args.overwrite)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest).fillna("")
    rows = []
    for _, item in tqdm(manifest.iterrows(), total=len(manifest), desc="downloading"):
        model_id = item.get("model_id", "")
        url = item.get("glb_url", "") or item.get("source_url", "")
        target = out_dir / f"{model_id}{Path(urlparse(url).path).suffix or '.glb'}"

        if not is_direct_glb(url):
            rows.append({"model_id": model_id, "url": url, "status": "not_direct_download", "path": ""})
            continue
        if target.exists() and not args.overwrite:
            rows.append({"model_id": model_id, "url": url, "status": "exists", "path": str(target)})
            continue

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            target.write_bytes(response.content)
            rows.append({"model_id": model_id, "url": url, "status": "downloaded", "path": str(target)})
        except Exception as exc:
            rows.append({"model_id": model_id, "url": url, "status": f"error: {exc}", "path": ""})

    pd.DataFrame(rows).to_csv(status_path, index=False)
    print(f"Wrote download status to {status_path}")


if __name__ == "__main__":
    main()
