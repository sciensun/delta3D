import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm

from utils_download import (
    append_status,
    copy_or_symlink,
    detect_site_type,
    download_direct_file,
    extension_from_url,
    extract_zip,
    fetch_sketchfab_download_info,
    find_best_3d_file,
    infer_columns,
    is_direct_download_url,
    row_value,
    safe_slug,
    save_debug_page,
    sketchfab_uid_from_url,
    write_metadata_json,
)


STATUS_FIELDS = [
    "model_id",
    "model_name",
    "source_url",
    "site_type",
    "direct_url",
    "local_dir",
    "expected_file",
    "status",
    "reason",
    "downloaded_file",
    "best_3d_file",
]

BLOCKED_TEXT = {
    "login_required": ["log in", "sign in", "login required", "sign in to download"],
    "purchase_required": ["buy now", "purchase", "add to cart", "price", "checkout"],
    "captcha_or_manual_required": ["captcha", "verify you are human", "cloudflare", "manual verification"],
    "no_permission": ["not available for download", "download unavailable", "no download", "not downloadable"],
}

DOWNLOAD_TEXTS = [
    "Download 3D Model",
    "Download model",
    "Download",
    "Get model",
    "Download options",
    "Download asset",
]

FORMAT_TEXTS = {
    "glb": ["GLB", "glTF Binary", "glTF binary", "glTF"],
    "gltf": ["glTF", "GLTF"],
    "fbx": ["FBX"],
    "obj": ["OBJ"],
    "any": ["GLB", "glTF", "FBX", "OBJ", "Original format", "Original"],
}

TEXTURE_TEXTS = {
    "4k": ["4K", "4096"],
    "2k": ["2K", "2048"],
    "1k": ["1K", "1024"],
    "highest": ["8K", "8192", "4K", "4096", "2K", "2048", "1K", "1024"],
    "any": [],
}


def ensure_output_paths(status_csv, overwrite):
    status_csv = Path(status_csv)
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    if status_csv.exists() and overwrite:
        status_csv.unlink()
    for sibling in ["download_log.txt", "manual_required.csv"]:
        path = status_csv.parent / sibling
        if path.exists() and overwrite:
            path.unlink()
    (status_csv.parent / "debug_screenshots").mkdir(parents=True, exist_ok=True)


def log_message(status_csv, message):
    log_path = Path(status_csv).parent / "download_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def choose_url(row, columns):
    for key in ["download_url", "glb_url", "url", "sketchfab_url", "fab_url"]:
        value = row_value(row, columns.get(key))
        if value:
            return value
    return ""


def normalized_rows(excel, out_dir, limit=None, start_index=0):
    df = pd.read_excel(excel, engine="openpyxl").fillna("")
    columns = infer_columns(df)
    rows = []
    subset = df.iloc[start_index:]
    if limit is not None:
        subset = subset.iloc[:limit]
    for seq, (_, row) in enumerate(subset.iterrows(), start=start_index + 1):
        name = row_value(row, columns.get("model_name")) or f"model_{seq:03d}"
        model_id = row_value(row, columns.get("model_id")) or f"{seq:03d}_{safe_slug(name)}"
        source_url = choose_url(row, columns)
        site_type = detect_site_type(source_url)
        local_dir = Path(out_dir) / safe_slug(model_id)
        ext = extension_from_url(source_url, ".glb" if site_type == "direct" else ".zip")
        expected = local_dir / f"source{ext}"
        rows.append(
            {
                "model_id": model_id,
                "model_name": name,
                "source_url": source_url,
                "site_type": site_type,
                "direct_url": source_url if is_direct_download_url(source_url) else "",
                "local_dir": str(local_dir),
                "expected_file": str(expected),
                "notes": row_value(row, columns.get("notes")),
            }
        )
    return rows


def existing_download(local_dir):
    local_dir = Path(local_dir)
    for name in ["source.glb", "source.gltf", "source.zip", "source.fbx", "source.obj"]:
        path = local_dir / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def finalize_download(record, file_path, prefer_format, overwrite):
    file_path = Path(file_path)
    local_dir = Path(record["local_dir"])
    best_3d = ""
    if file_path.suffix.lower() == ".zip":
        extract_dir = local_dir / "extracted"
        extract_zip(file_path, extract_dir, overwrite=overwrite)
        candidate = find_best_3d_file(extract_dir, prefer_format=prefer_format)
        if candidate:
            best_3d = str(candidate)
            if candidate.suffix.lower() == ".glb":
                copy_or_symlink(candidate, local_dir / "source.glb")
    elif file_path.suffix.lower() in {".glb", ".gltf", ".fbx", ".obj"}:
        best_3d = str(file_path)

    metadata = {
        "model_id": record["model_id"],
        "model_name": record["model_name"],
        "source_url": record["source_url"],
        "site_type": record["site_type"],
        "downloaded_file": str(file_path),
        "best_3d_file": best_3d,
        "format": file_path.suffix.lower().lstrip("."),
    }
    write_metadata_json(local_dir / "metadata.json", **metadata)
    return best_3d


def status_row(record, status, reason="", downloaded_file="", best_3d_file=""):
    return {
        "model_id": record.get("model_id", ""),
        "model_name": record.get("model_name", ""),
        "source_url": record.get("source_url", ""),
        "site_type": record.get("site_type", ""),
        "direct_url": record.get("direct_url", ""),
        "local_dir": record.get("local_dir", ""),
        "expected_file": record.get("expected_file", ""),
        "status": status,
        "reason": reason,
        "downloaded_file": downloaded_file,
        "best_3d_file": best_3d_file,
    }


def write_manual_required(status_csv, row):
    path = Path(status_csv).parent / "manual_required.csv"
    append_status(path, row, fieldnames=STATUS_FIELDS)


def click_if_visible(page, text, timeout=1200):
    locators = [
        page.get_by_role("button", name=re_escape(text)),
        page.get_by_role("link", name=re_escape(text)),
        page.get_by_text(text, exact=False),
    ]
    for locator in locators:
        try:
            if locator.first.is_visible(timeout=timeout):
                locator.first.click(timeout=timeout)
                return True
        except Exception:
            continue
    return False


def re_escape(text):
    import re

    return re.compile(re.escape(text), re.IGNORECASE)


def accept_cookie_banners(page):
    for text in ["Accept all", "Accept", "I agree", "Agree", "Allow all", "Got it"]:
        try:
            page.get_by_role("button", name=re_escape(text)).first.click(timeout=1000)
            return
        except Exception:
            pass


def detect_blocked_reason(page):
    try:
        text = page.locator("body").inner_text(timeout=3000).lower()
    except Exception:
        return ""
    for reason, needles in BLOCKED_TEXT.items():
        if any(needle in text for needle in needles):
            return reason
    return ""


def try_choose_options(page, prefer_format, prefer_texture):
    format_choices = FORMAT_TEXTS.get(prefer_format, FORMAT_TEXTS["glb"])
    if prefer_format != "glb":
        format_choices += FORMAT_TEXTS["glb"]
    for text in format_choices:
        if click_if_visible(page, text, timeout=800):
            break
    for text in TEXTURE_TEXTS.get(prefer_texture, []):
        if click_if_visible(page, text, timeout=800):
            break


def browser_download(record, page, args):
    page.goto(record["source_url"], wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    accept_cookie_banners(page)

    blocked = detect_blocked_reason(page)
    if blocked:
        return "manual_required", blocked, "", ""

    clicked = False
    for text in DOWNLOAD_TEXTS:
        if click_if_visible(page, text):
            clicked = True
            break
    if not clicked:
        blocked = detect_blocked_reason(page)
        return ("manual_required", blocked, "", "") if blocked else ("skipped", "no_download_button", "", "")

    time.sleep(1.0)
    try_choose_options(page, args.prefer_format, args.prefer_texture)

    local_dir = Path(record["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        with page.expect_download(timeout=30000) as download_info:
            clicked_download = False
            for text in DOWNLOAD_TEXTS + ["Download now", "Confirm download", "Start download"]:
                if click_if_visible(page, text, timeout=1200):
                    clicked_download = True
                    break
            if not clicked_download:
                raise TimeoutError("no final download click target found")
        download = download_info.value
    except Exception:
        blocked = detect_blocked_reason(page)
        return ("manual_required", blocked, "", "") if blocked else ("manual_required", "download_timeout", "", "")

    suggested = download.suggested_filename or "source.bin"
    suffix = Path(suggested).suffix.lower() or ".bin"
    target_name = "source.zip" if suffix == ".zip" else f"source{suffix}"
    target = local_dir / target_name
    if target.exists() and not args.overwrite:
        best = finalize_download(record, target, args.prefer_format, args.overwrite)
        return "exists", "already_downloaded", str(target), best
    download.save_as(str(target))
    if not target.exists() or target.stat().st_size == 0:
        return "failed", "downloaded_file_empty", str(target), ""
    best = finalize_download(record, target, args.prefer_format, args.overwrite)
    return "downloaded", "", str(target), best


def handle_direct(record, args):
    local_dir = Path(record["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)
    ext = extension_from_url(record["source_url"], ".bin")
    target = local_dir / f"source{ext}"
    if args.dry_run:
        return "dry_run", "direct_download_url", str(target), ""
    result = download_direct_file(record["source_url"], target, overwrite=args.overwrite)
    best = finalize_download(record, target, args.prefer_format, args.overwrite)
    return result["status"], "", str(target), best


def choose_sketchfab_asset(download_info, prefer_format):
    keys = []
    if prefer_format in {"glb", "gltf"}:
        keys.extend(["gltf", "glb"])
    elif prefer_format in {"fbx", "obj"}:
        keys.extend(["source", prefer_format, "gltf", "glb"])
    keys.extend(["gltf", "glb", "source", "usdz"])

    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        asset = download_info.get(key)
        if isinstance(asset, dict) and asset.get("url"):
            return key, asset["url"]
    for key, asset in download_info.items():
        if isinstance(asset, dict) and asset.get("url"):
            return key, asset["url"]
    return "", ""


def handle_sketchfab_api(record, args):
    token = args.sketchfab_api_token or os.environ.get("SKETCHFAB_API_TOKEN", "")
    if not token:
        return None
    uid = sketchfab_uid_from_url(record["source_url"])
    if not uid:
        return status_row(record, "manual_required", "sketchfab_uid_not_found")
    local_dir = Path(record["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        info = fetch_sketchfab_download_info(uid, token)
        asset_name, asset_url = choose_sketchfab_asset(info, args.prefer_format)
        if not asset_url:
            return status_row(record, "manual_required", "no_api_download_option")
        ext = extension_from_url(asset_url, ".zip")
        target = local_dir / ("source.zip" if ext == ".zip" else f"source{ext}")
        result = download_direct_file(asset_url, target, overwrite=args.overwrite)
        best = finalize_download(record, target, args.prefer_format, args.overwrite)
        write_metadata_json(
            local_dir / "metadata.json",
            model_id=record["model_id"],
            model_name=record["model_name"],
            source_url=record["source_url"],
            site_type=record["site_type"],
            sketchfab_uid=uid,
            api_asset=asset_name,
            downloaded_file=str(target),
            best_3d_file=best,
            format=target.suffix.lower().lstrip("."),
        )
        return status_row(record, result["status"], f"sketchfab_api:{asset_name}", str(target), best)
    except Exception as exc:
        reason = str(exc)
        if "403" in reason:
            return status_row(record, "manual_required", "no_download_permission")
        if "401" in reason:
            return status_row(record, "manual_required", "api_login_required")
        return status_row(record, "manual_required", f"sketchfab_api_failed: {exc}")


def handle_record(record, args, page=None):
    local_dir = Path(record["local_dir"])
    existing = existing_download(local_dir)
    if existing and not args.overwrite:
        best = finalize_download(record, existing, args.prefer_format, args.overwrite)
        return status_row(record, "exists", "already_downloaded", str(existing), best)
    if not record["source_url"]:
        return status_row(record, "skipped", "missing_url")
    if args.dry_run:
        return status_row(record, "dry_run", record["site_type"])
    if record["site_type"] == "direct":
        try:
            status, reason, downloaded, best = handle_direct(record, args)
            return status_row(record, status, reason, downloaded, best)
        except Exception as exc:
            return status_row(record, "failed", f"direct_download_failed: {exc}")
    if record["site_type"] == "sketchfab":
        api_result = handle_sketchfab_api(record, args)
        if api_result is not None and api_result["status"] in {"downloaded", "exists"}:
            return api_result
        if api_result is not None and page is None:
            return api_result
    if record["site_type"] in {"sketchfab", "fab", "webpage"}:
        if page is None:
            return status_row(record, "manual_required", "browser_unavailable")
        try:
            status, reason, downloaded, best = browser_download(record, page, args)
            return status_row(record, status, reason, downloaded, best)
        except Exception as exc:
            return status_row(record, "failed", f"browser_download_failed: {exc}")
    return status_row(record, "skipped", "unsupported_site")


def make_browser(args):
    from playwright.sync_api import sync_playwright

    if args.headful and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "--headful requires a running XServer/DISPLAY. "
            "Run without --headful in this terminal, or run from a graphical desktop/SSH X11 session."
        )

    p = sync_playwright().start()
    if args.use_persistent_browser:
        context = p.chromium.launch_persistent_context(
            args.browser_profile,
            headless=not args.headful,
            accept_downloads=True,
        )
        browser = None
    else:
        browser = p.chromium.launch(headless=not args.headful)
        context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    return p, browser, context, page


def main():
    parser = argparse.ArgumentParser(description="Download model files from Excel links conservatively.")
    parser.add_argument("--excel", required=True)
    parser.add_argument("--out_dir", default="downloads")
    parser.add_argument("--status_csv", default="outputs/download_status.csv")
    parser.add_argument("--prefer_format", default="glb", choices=["glb", "gltf", "fbx", "obj", "any"])
    parser.add_argument("--prefer_texture", default="4k", choices=["4k", "2k", "1k", "highest", "any"])
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--use_persistent_browser", action="store_true")
    parser.add_argument("--browser_profile", default=".browser_profile")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sketchfab_api_token", default=os.environ.get("SKETCHFAB_API_TOKEN", ""))
    args = parser.parse_args()

    ensure_output_paths(args.status_csv, args.overwrite)
    records = normalized_rows(args.excel, args.out_dir, args.limit, args.start_index)
    browser_needed = any(record["site_type"] in {"sketchfab", "fab", "webpage"} for record in records) and not args.dry_run

    p = browser = context = page = None
    if browser_needed:
        try:
            p, browser, context, page = make_browser(args)
        except Exception as exc:
            log_message(args.status_csv, f"WARNING: failed to start browser: {exc}")

    for record in tqdm(records, desc="downloading"):
        result = handle_record(record, args, page=page)
        if args.debug and page is not None and result["status"] in {"failed", "manual_required", "skipped"}:
            save_debug_page(page, Path(args.status_csv).parent / "debug_screenshots", record["model_id"], result["reason"] or result["status"])
        append_status(args.status_csv, result, fieldnames=STATUS_FIELDS)
        if result["status"] == "manual_required":
            write_manual_required(args.status_csv, result)
        log_message(args.status_csv, f"{result['model_id']} {result['status']} {result['reason']}")

    if context is not None:
        context.close()
    if browser is not None:
        browser.close()
    if p is not None:
        p.stop()


if __name__ == "__main__":
    main()
