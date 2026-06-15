# Manual ChatGPT Data Preparation Pipeline

This pipeline prepares rendered views and prompts for:

```text
assets/3D/big_carved_wooden_elephant_sculpture.glb
```

It does not use `OPENAI_API_KEY`, does not call the OpenAI API, and does not generate standard images automatically. You run image generation manually in ChatGPT.

## 1. Prepare Renders And Upload Pack

From the repository root:

```bash
bash scripts/prepare_elephant_manual_chatgpt.sh
```

The script renders:

```text
assets/prepared/big_carved_wooden_elephant_sculpture/renders_original/full36/
assets/prepared/big_carved_wooden_elephant_sculpture/renders_original/key8/
assets/prepared/big_carved_wooden_elephant_sculpture/renders_original/tripo/
```

It also writes:

```text
assets/prepared/big_carved_wooden_elephant_sculpture/prompts/prompts_standard_key8.json
assets/prepared/big_carved_wooden_elephant_sculpture/prompts/chatgpt_manual_prompts.md
assets/prepared/big_carved_wooden_elephant_sculpture/chatgpt_upload_pack/
```

The upload pack contains paired files:

```text
01_key_e000_a000.png
01_prompt.txt
02_key_e000_a045.png
02_prompt.txt
...
08_key_e000_a315.png
08_prompt.txt
```

## 2. Manually Use ChatGPT

Open ChatGPT and repeat this for all 8 views:

1. Upload `01_key_e000_a000.png`.
2. Paste the text from `01_prompt.txt`.
3. Download the generated result as `01_standard.png`.
4. Repeat for `02` through `08`.

Save the generated results here:

```text
assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/key8_manual/
```

Expected filenames:

```text
01_standard.png
02_standard.png
03_standard.png
04_standard.png
05_standard.png
06_standard.png
07_standard.png
08_standard.png
```

## 3. Collect Manual Results

After saving the generated images, run:

```bash
python scripts/collect_manual_chatgpt_results.py
```

The helper checks which `01` to `08` outputs exist, writes:

```text
assets/prepared/big_carved_wooden_elephant_sculpture/generated_standard/manual_collection_report.json
```

It then copies the preferred front three-quarter result, initially `02_standard.png`, to:

```text
assets/prepared/big_carved_wooden_elephant_sculpture/tripo_input/standard_front_3quarter.png
```

## 4. Upload To Tripo

Upload this image to Tripo image-to-3D:

```text
assets/prepared/big_carved_wooden_elephant_sculpture/tripo_input/standard_front_3quarter.png
```

After downloading the Tripo GLB, place it at:

```text
assets/prepared/big_carved_wooden_elephant_sculpture/tripo_standard/source_standard.glb
```

Then render it with:

```bash
bash scripts/render_tripo_output.sh
```

## Notes

Generated ChatGPT images are weak references, not pixel-aligned ground truth. Review them for silhouette consistency, object identity, and semantic consistency before using them for any later training.
