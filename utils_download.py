import csv
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests


DIRECT_EXTENSIONS = {".glb", ".gltf", ".zip", ".obj", ".fbx"}
THREE_D_PRIORITY = [".glb", ".gltf", ".fbx", ".obj"]


def safe_slug(text, max_len=80):
    text = str(text or "").strip().lower()
    text = re.sub(r"[^\w\s.-]+", "", text)
    text = re.sub(r"[\s.-]+", "_", text)
    text = text.strip("_")
    return (text or "model")[:max_len]


def _normalize_column(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_column(columns, aliases):
    normalized = {_normalize_column(col): col for col in columns}
    for alias in aliases:
        found = normalized.get(_normalize_column(alias))
        if found is not None:
            return found
    return None


def infer_columns(df):
    columns = list(df.columns)
    inferred = {
        "model_id": _find_column(columns, ["model_id", "id", "model id", "asset_id", "asset id"]),
        "model_name": _find_column(columns, ["model_name", "name", "title", "model", "model title", "动物", "animal"]),
        "url": _find_column(columns, ["source_url", "url", "link", "model_url", "page_url"]),
        "sketchfab_url": _find_column(columns, ["sketchfab_url", "sketchfab", "sketchfab link"]),
        "fab_url": _find_column(columns, ["fab_url", "fab", "fab link"]),
        "glb_url": _find_column(columns, ["glb_url", "glb", "direct_glb"]),
        "download_url": _find_column(columns, ["download_url", "download", "download link", "direct_url"]),
        "notes": _find_column(columns, ["notes", "note", "comment", "comments"]),
    }
    if not any(inferred[key] for key in ["url", "sketchfab_url", "fab_url", "glb_url", "download_url"]):
        for column in columns:
            values = df[column].dropna().astype(str).str.strip()
            if values.empty:
                continue
            url_like = values.str.contains(r"^https?://", case=False, regex=True).mean()
            if url_like >= 0.5:
                inferred["url"] = column
                break

    if inferred["model_name"] is None:
        url_columns = {value for key, value in inferred.items() if key.endswith("url") or key == "url"}
        for column in columns:
            if column in url_columns:
                continue
            values = df[column].dropna().astype(str).str.strip()
            if not values.empty:
                inferred["model_name"] = column
                break

    if inferred["model_id"] is None:
        inferred["model_id"] = inferred["model_name"]

    return inferred


def row_value(row, column):
    if not column or column not in row or pd.isna(row[column]):
        return ""
    value = str(row[column]).strip()
    return "" if value.lower() == "nan" else value


def detect_site_type(url):
    if not url:
        return "missing"
    parsed = urlparse(str(url).strip())
    host = parsed.netloc.lower()
    path = unquote(parsed.path.lower())
    if Path(path).suffix in DIRECT_EXTENSIONS:
        return "direct"
    if "sketchfab.com" in host:
        return "sketchfab"
    if host == "fab.com" or host.endswith(".fab.com"):
        return "fab"
    return "webpage"


def sketchfab_uid_from_url(url):
    match = re.search(r"/3d-models/[^/]*-([0-9a-f]{32})(?:[/?#]|$)", str(url), flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"/models/([0-9a-f]{32})(?:[/?#]|$)", str(url), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def fetch_sketchfab_download_info(model_uid, api_token, timeout=30):
    if not model_uid:
        raise ValueError("missing Sketchfab model uid")
    if not api_token:
        raise ValueError("missing Sketchfab API token")
    url = f"https://api.sketchfab.com/v3/models/{model_uid}/download"
    last_error = None
    for scheme in ["Token", "Bearer"]:
        response = requests.get(url, headers={"Authorization": f"{scheme} {api_token}"}, timeout=timeout)
        if response.status_code == 401:
            last_error = response
            continue
        response.raise_for_status()
        return response.json()
    if last_error is not None:
        last_error.raise_for_status()
    raise RuntimeError("failed to fetch Sketchfab download info")


def is_direct_download_url(url):
    return detect_site_type(url) == "direct"


def extension_from_url(url, default=".bin"):
    suffix = Path(unquote(urlparse(str(url)).path)).suffix.lower()
    return suffix if suffix in DIRECT_EXTENSIONS else default


def download_direct_file(url, out_path, overwrite=False, timeout=60):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return {"status": "exists", "bytes": out_path.stat().st_size, "path": str(out_path)}

    part_path = out_path.with_suffix(out_path.suffix + ".part")
    headers = {}
    mode = "wb"
    existing = 0
    if part_path.exists() and not overwrite:
        existing = part_path.stat().st_size
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"

    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        if response.status_code == 416 and part_path.exists():
            part_path.replace(out_path)
            return {"status": "downloaded", "bytes": out_path.stat().st_size, "path": str(out_path)}
        if response.status_code != 206:
            mode = "wb"
        response.raise_for_status()
        with open(part_path, mode + ("" if "b" in mode else "b")) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if not part_path.exists() or part_path.stat().st_size == 0:
        raise RuntimeError("downloaded file is empty")
    part_path.replace(out_path)
    return {"status": "downloaded", "bytes": out_path.stat().st_size, "path": str(out_path)}


def extract_zip(zip_path, extract_dir, overwrite=False):
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    if extract_dir.exists() and any(extract_dir.iterdir()) and not overwrite:
        return str(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    return str(extract_dir)


def find_best_3d_file(folder, prefer_format="glb"):
    folder = Path(folder)
    if not folder.exists():
        return None
    prefer = f".{prefer_format.lower().lstrip('.')}" if prefer_format != "any" else None
    priority = list(THREE_D_PRIORITY)
    if prefer in priority:
        priority.remove(prefer)
        priority.insert(0, prefer)
    candidates = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in THREE_D_PRIORITY]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (priority.index(path.suffix.lower()), len(path.parts), str(path).lower()))
    return candidates[0]


def write_metadata_json(path, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata.setdefault("updated_at", datetime.utcnow().isoformat(timespec="seconds") + "Z")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def append_status(csv_path, row, fieldnames=None):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(row.keys())
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_debug_page(page, out_dir, model_id, reason):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{safe_slug(model_id)}_{safe_slug(reason)}"
    screenshot_path = base.with_suffix(".png")
    html_path = base.with_suffix(".html")
    try:
        page.screenshot(path=str(screenshot_path), full_page=True)
    except Exception:
        screenshot_path = ""
    try:
        html_path.write_text(page.content(), encoding="utf-8")
    except Exception:
        html_path = ""
    return {"screenshot": str(screenshot_path), "html": str(html_path)}


def copy_or_symlink(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        return str(dst)
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)
    return str(dst)
