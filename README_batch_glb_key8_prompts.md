# Batch GLB Key8 Prompt Preparation

Put new `.glb` files in:

```text
assets/3D/
```

Then run:

```bash
bash scripts/prepare_unprocessed_glbs_key8_prompts.sh
```

The script scans `assets/3D/*.glb` and skips objects that already have at least 8 key8 renders plus:

```text
assets/prepared/<object_id>/prompts/prompts_standard_key8.json
```

For each unprocessed object, it writes:

```text
assets/prepared/<object_id>/
  renders_original/key8/
    000_key8_az000.png
    ...
    007_key8_az315.png
    views_meta.json
  prompts/
    prompts_standard_key8.json
    chatgpt_manual_prompts.md
  chatgpt_upload_pack/
    01_key_e000_a000.png
    01_prompt.txt
    ...
    08_key_e000_a315.png
    08_prompt.txt
  logs/
```

Options:

```bash
bash scripts/prepare_unprocessed_glbs_key8_prompts.sh --dry-run
bash scripts/prepare_unprocessed_glbs_key8_prompts.sh --force
bash scripts/prepare_unprocessed_glbs_key8_prompts.sh --resolution 768
```

`--force` regenerates outputs even if the object appears processed.
