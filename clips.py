from playwright.sync_api import sync_playwright
from PIL import Image
from pathlib import Path
import os
import re
import time

urls = [
    "https://sketchfab.com/3d-models/deer-carved-wood-statue-99be78357eee48069420ee38af0eb5a7",
    "https://sketchfab.com/3d-models/porcelain-bird-5446c198097e43cf8e3244d8f4d52f4c",
    "https://sketchfab.com/3d-models/stylized-deer-6dcc0df864ef43fdba0e40af7f083802",
    "https://sketchfab.com/3d-models/cat-egypt-451b471a2bfa4108839e2d4f167a67c8",
    "https://sketchfab.com/3d-models/lowe-bfb0f0cdbd324049b444188c9dff7ef2",
    "https://sketchfab.com/3d-models/bear-a9bc9acf52fe4bcface7d4ee8f8f3d2a",
    "https://sketchfab.com/3d-models/polar-bear-ornament-artec-leo-3d-scan-841cb663e5324418885f5664bba0fdfa",
    "https://sketchfab.com/3d-models/folk-art-circus-horse-antique-237c83689e384cbf907c2a8c1442c35f",
    "https://sketchfab.com/3d-models/hirsch-schloss-waldthausen-mainz-6dffd950cd5640cb96fb3c297febaed0",
    "https://sketchfab.com/3d-models/pigeon-1-640af0b61ea9454da08d4b75cb6bb6ae",
    "https://sketchfab.com/3d-models/laying-lion-5e6be40c0b48451fb98043f5be1e18f1",
    "https://sketchfab.com/3d-models/t-ronda5a-09192ac2ddf94a85b56b85fada30286a",
    "https://sketchfab.com/3d-models/wooden-lion-791850b543fe4616b26a08bbc0134bad",
    "https://sketchfab.com/3d-models/skulptur-lowengebaude-halle-saale-3d-scan-e2eda09c15354e369691babdb13cd341",
    "https://sketchfab.com/3d-models/the-jennings-dog-a3b10c4681e34440aa61ea2c9a80c233",
    "https://sketchfab.com/3d-models/sculpture-of-a-dog-d479cc72dcf94d5ea4e0ca982ec39abe",
    "https://sketchfab.com/3d-models/the-hound-of-alcibiades-21cb14e2560048d1be5a26f2cfda7800",
    "https://sketchfab.com/3d-models/3d-scan-dog-5e7b61f9c5cc40acbf300df6eda6f90b",
    "https://sketchfab.com/3d-models/lion-neutral-standing-pose-cc1cbfc212d74b968bd265509c1b0d84",
    "https://sketchfab.com/3d-models/lying-cat-79ea1a182a074aae823800ed3fa26615",
    "https://sketchfab.com/3d-models/stoops-bird-owl-a3b5805e898244b8972d0c6d721a0815",
    "https://sketchfab.com/3d-models/eagle-sculpture-811daf3a66fe4777a2d26b2fca951528",
    "https://sketchfab.com/3d-models/urial-animal-4e1cb826f1284549934f011363e708f5",
    "https://sketchfab.com/3d-models/concrete-cat-fed6469620524e6c9e0f47b6ccf97448",
    "https://sketchfab.com/3d-models/bird-ornament-export-test-128bf307e92f49af93c6fa10b450b32c",
]

limit = int(os.environ.get("CLIPS_LIMIT", "0"))
if limit > 0:
    urls = urls[:limit]

out_dir = Path("thumbs_300")
out_dir.mkdir(exist_ok=True)

def slug_from_url(url):
    slug = url.rstrip("/").split("/")[-1]
    slug = re.sub(r"-[0-9a-f]{20,}$", "", slug)
    slug = slug.replace("-", "_")
    return slug[:60]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 420, "height": 420}, device_scale_factor=1)

    for i, url in enumerate(urls, 1):
        name = f"{i:03d}_{slug_from_url(url)}"
        print("capturing", name, flush=True)

        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(8)

        raw_path = out_dir / f"{name}_raw.png"
        final_path = out_dir / f"{name}.png"

        page.screenshot(path=str(raw_path), full_page=False)

        with Image.open(raw_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            s = min(w, h)
            left = (w - s) // 2
            top = (h - s) // 2
            img = img.crop((left, top, left + s, top + s)).resize((300, 300))
            img.save(final_path)

        if raw_path.exists():
            raw_path.unlink()

    browser.close()
