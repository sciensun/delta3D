# Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction

## [Project page](https://ingra14m.github.io/Deformable-Gaussians/) | [Paper](https://arxiv.org/abs/2309.13101)

![Teaser image](assets/teaser.png)

This repository contains the official implementation associated with the paper "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction".



## News

- **[5/26/2024]** [Lightweight-Deformable-GS](https://github.com/ingra14m/Lightweight-Deformable-GS) has been integrated into this repo. For the original version aligned with paper, please check the [paper](https://github.com/ingra14m/Deformable-3D-Gaussians/tree/paper) branch.
- **[5/24/2024]** An optimized version [Lightweight-Deformable-GS](https://github.com/ingra14m/Lightweight-Deformable-GS) has been released. It offers 50% reduced storage, 200% increased FPS, and no decrease in rendering metrics.
- **[2/27/2024]** Deformable-GS is accepted by CVPR 2024. Our another work, [SC-GS](https://yihua7.github.io/SC-GS-web/) (with higher quality, less points and faster FPS than vanilla 3D-GS), is also accepted. See you in Seattle.
- **[11/16/2023]** Full code and real-time viewer released.
- **[11/4/2023]** update the computation of LPIPS in metrics.py. Previously, the `lpipsPyTorch` was unable to execute on CUDA, prompting us to switch to the `lpips` library (~20x faster).
- **[10/25/2023]** update **real-time viewer** on project page. Many, many thanks to @[yihua7](https://github.com/yihua7) for implementing the real-time viewer adapted for Deformable-GS. Also, thanks to @[ashawkey](https://github.com/ashawkey) for releasing the original GUI.



## Dataset

In our paper, we use:

- synthetic dataset from [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html).
- real-world dataset from [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/) and [Hyper-NeRF](https://hypernerf.github.io/).
- The dataset in the supplementary materials comes from [DeVRF](https://jia-wei-liu.github.io/DeVRF/).

We organize the datasets as follows:

```shell
├── data
│   | D-NeRF 
│     ├── hook
│     ├── standup 
│     ├── ...
│   | NeRF-DS
│     ├── as
│     ├── basin
│     ├── ...
│   | HyperNeRF
│     ├── interp
│     ├── misc
│     ├── vrig
```

> I have identified an **inconsistency in the D-NeRF's Lego dataset**. Specifically, the scenes corresponding to the training set differ from those in the test set. This discrepancy can be verified by observing the angle of the flipped Lego shovel. To meaningfully evaluate the performance of our method on this dataset, I recommend using the **validation set of the Lego dataset** as the test set. See more in [D-NeRF dataset used in Deformable-GS](https://github.com/ingra14m/Deformable-3D-Gaussians/releases/tag/v0.1-pre-released)



## Pipeline

![Teaser image](assets/pipeline.png)



## Run

### Environment

```shell
git clone https://github.com/ingra14m/Deformable-3D-Gaussians --recursive
cd Deformable-3D-Gaussians

conda create -n deformable_gaussian_env python=3.7
conda activate deformable_gaussian_env

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -r requirements.txt
```



### Train

**D-NeRF:**

```shell
python train.py -s path/to/your/d-nerf/dataset -m output/exp-name --eval --is_blender
```

**NeRF-DS/HyperNeRF:**

```shell
python train.py -s path/to/your/real-world/dataset -m output/exp-name --eval --iterations 20000
```

**6DoF Transformation:**

We have also implemented the 6DoF transformation of 3D-GS, which may lead to an improvement in metrics but will reduce the speed of training and inference.

```shell
# D-NeRF
python train.py -s path/to/your/d-nerf/dataset -m output/exp-name --eval --is_blender --is_6dof

# NeRF-DS & HyperNeRF
python train.py -s path/to/your/real-world/dataset -m output/exp-name --eval --is_6dof --iterations 20000
```

You can also **train with the GUI:**

```shell
python train_gui.py -s path/to/your/dataset -m output/exp-name --eval --is_blender
```

- click `start` to start training, and click `stop` to stop training.
- The GUI viewer is still under development, many buttons do not have corresponding functions currently. We plan to :
  - [ ] reload checkpoints from the pre-trained model.
  - [ ] Complete the functions of the other vacant buttons in the GUI.



### Render & Evaluation

```shell
python render.py -m output/exp-name --mode render
python metrics.py -m output/exp-name
```

We provide several modes for rendering:

- `render`: render all the test images
- `time`: time interpolation tasks for D-NeRF dataset
- `all`: time and view synthesis tasks for D-NeRF dataset
- `view`: view synthesis tasks for D-NeRF dataset
- `original`: time and view synthesis tasks for real-world dataset



## Results

### D-NeRF Dataset

**Quantitative Results**

<img src="assets/results/D-NeRF/Quantitative.jpg" alt="Image1" style="zoom:50%;" />

**Qualitative Results**

 <img src="assets/results/D-NeRF/bouncing.gif" alt="Image1" style="zoom:25%;" />  <img src="assets/results/D-NeRF/hell.gif" alt="Image1" style="zoom:25%;" />  <img src="assets/results/D-NeRF/hook.gif" alt="Image3" style="zoom:25%;" />  <img src="assets/results/D-NeRF/jump.gif" alt="Image4" style="zoom:25%;" /> 

 <img src="assets/results/D-NeRF/lego.gif" alt="Image5" style="zoom:25%;" />  <img src="assets/results/D-NeRF/mutant.gif" alt="Image6" style="zoom:25%;" />  <img src="assets/results/D-NeRF/stand.gif" alt="Image7" style="zoom:25%;" />  <img src="assets/results/D-NeRF/trex.gif" alt="Image8" style="zoom:25%;" /> 

**400x400 Resolution**

|          | PSNR  | SSIM   | LPIPS (VGG) | FPS  | Mem   | Num. (k) |
| -------- | ----- | ------ | ----------- | ---- | ----- | -------- |
| bouncing | 41.46 | 0.9958 | 0.0046      | 112  | 13.16 | 55622    |
| hell     | 42.11 | 0.9885 | 0.0153      | 375  | 3.72  | 15733    |
| hook     | 37.77 | 0.9897 | 0.0103      | 128  | 11.74 | 49613    |
| jump     | 39.10 | 0.9930 | 0.0090      | 217  | 6.81  | 28808    |
| mutant   | 43.73 | 0.9969 | 0.0029      | 124  | 11.45 | 48423    |
| standup  | 45.38 | 0.9967 | 0.0032      | 210  | 5.94  | 25102    |
| trex     | 38.40 | 0.9959 | 0.0041      | 85   | 18.6  | 78624    |
| Average  | 41.14 | 0.9938 | 0.0070      | 179  | 10.20 | 43132    |

### NeRF-DS Dataset

<img src="assets/results/NeRF-DS/Quantitative.jpg" alt="Image1" style="zoom:50%;" />

See more visualization on our [project page](https://ingra14m.github.io/Deformable-Gaussians/).



### HyperNeRF Dataset

Since the **camera pose** in HyperNeRF is less precise compared to NeRF-DS, we use HyperNeRF as a reference for partial visualization and the display of Failure Cases, but do not include it in the calculation of quantitative metrics. The results of the HyperNeRF dataset can be viewed on the [project page](https://ingra14m.github.io/Deformable-Gaussians/).



### Real-Time Viewer

https://github.com/ingra14m/Deformable-3D-Gaussians/assets/63096187/ec26d0b9-c126-4e23-b773-dcedcf386f36



## Acknowledgments

We sincerely thank the authors of [3D-GS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html), [HyperNeRF](https://hypernerf.github.io/), [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/), and [DeVRF](https://jia-wei-liu.github.io/DeVRF/), whose codes and datasets were used in our work. We thank [Zihao Wang](https://github.com/Alen-Wong) for the debugging in the early stage, preventing this work from sinking. We also thank the reviewers and AC for not being influenced by PR, and fairly evaluating our work. This work was mainly supported by ByteDance MMLab.




## BibTex

```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}
```

And thanks to the authors of [3D Gaussians](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) for their excellent code, please consider also cite this repository:

```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## Stylized Animal Model Labeling Tools

This repository also includes small helper scripts for organizing preview images and generating candidate labels for stylized 3D animal deformation research. The labels are not final decisions; they are intended for manual correction.

Expected inputs:

```bash
previews/      # one preview image per model
models.xlsx    # model names plus GLB or Sketchfab URLs
```

Basic workflow:

```bash
mkdir -p previews
# Put one preview image per model in previews/
# Put the spreadsheet in the project root as models.xlsx

bash run_label_pipeline.sh
```

The single-entry script runs the three steps below:

```bash
python build_manifest.py --image_dir previews --excel models.xlsx --out outputs/manifest.csv --match_mode order

python auto_tag_images.py --manifest outputs/manifest.csv --vocab outputs/deformation_vocab.yaml --out outputs/labels_auto.xlsx --jsonl outputs/labels.jsonl

python make_contact_sheet.py --manifest outputs/manifest.csv --labels outputs/labels_auto.xlsx --out outputs/contact_sheet.png
```

The auto-tagging step also writes `outputs/labels_for_manual.xlsx`, which is the editable manual review copy.

Pipeline inputs you need to prepare:

- `previews/`: put preview images here, one image per model.
- `models.xlsx`: put this Excel file in the project root. It should contain model names and GLB or Sketchfab URLs.
- Matching defaults to row order. To match by filename stem, run `MATCH_MODE=stem bash run_label_pipeline.sh`.
- Existing outputs are protected. To regenerate them, run `OVERWRITE=true bash run_label_pipeline.sh`.
- Custom paths are supported, for example `IMAGE_DIR=thumbs_300 EXCEL=my_models.xlsx OUT_DIR=outputs bash run_label_pipeline.sh`.

Optional GLB utilities:

```bash
python download_glbs.py --manifest outputs/manifest.csv --out_dir glbs
python extract_glb_features.py --manifest outputs/manifest.csv --glb_dir glbs --out outputs/geometry_features.csv
```

All generated files are protected by default. Pass `--overwrite` to replace an existing output file.

## Dataset Downloader

Use this pipeline when you have an Excel file of model links and want to build a local download folder. The downloader is conservative: it does not bypass paywalls, login restrictions, captcha, DRM, or website access controls. If a page requires login, purchase, manual confirmation, captcha, or lacks download permission, the row is recorded as `manual_required` or skipped.

Inputs you prepare:

- Put the spreadsheet in the project root as `models.xlsx`, or pass `--excel /path/to/file.xlsx`.
- The spreadsheet may contain columns such as `model_id`, `name`, `title`, `url`, `sketchfab_url`, `fab_url`, `glb_url`, `download_url`, and `notes`.
- Direct `.glb`, `.gltf`, `.zip`, `.obj`, and `.fbx` URLs are downloaded with `requests`.
- Sketchfab/Fab/webpage links are opened with Playwright and only normal visible webpage download controls are used.

Step 1: test the first 3 rows in the current terminal. This runs headless, which works on servers without an XServer:

```bash
/usr/bin/python download_dataset.py --excel models.xlsx --out_dir downloads --limit 3 --use_persistent_browser --debug
```

Step 2: run all rows:

```bash
/usr/bin/python download_dataset.py --excel models.xlsx --out_dir downloads --use_persistent_browser --debug
```

Use `--headful` only from a terminal with a graphical desktop/XServer, for example when you need to log in manually once and keep cookies in `.browser_profile`:

```bash
/usr/bin/python download_dataset.py --excel models.xlsx --out_dir downloads --limit 3 --headful --use_persistent_browser --debug
```

From Windows SSH, start an XServer such as VcXsrv/Xming/MobaXterm, connect with X11 forwarding, log in once, then run the downloader:

```bash
ssh -Y user@server
cd /home/shichang/Deformable-3D-Gaussians
/usr/bin/python browser_login.py --url https://sketchfab.com/login --browser_profile .browser_profile
/usr/bin/python download_dataset.py --excel models.xlsx --out_dir downloads --limit 3 --headful --use_persistent_browser --debug --overwrite
```

After the login cookies are saved, later runs can usually be headless:

```bash
/usr/bin/python download_dataset.py --excel models.xlsx --out_dir downloads --use_persistent_browser --debug --overwrite
```

If Google/Epic/Apple login refuses the Playwright browser, use a Sketchfab API token instead. Get the token in your normal Windows browser from your Sketchfab account/developer settings, then pass it to the server:

```bash
cd /home/shichang/Deformable-3D-Gaussians
export SKETCHFAB_API_TOKEN="paste_your_token_here"

/usr/bin/python download_dataset.py --excel models.xlsx --out_dir downloads --limit 3 --debug --overwrite
```

With `SKETCHFAB_API_TOKEN` set, Sketchfab rows are attempted through the official download API first. Rows without download permission are still recorded as `manual_required`; the script does not bypass account, license, purchase, or permission restrictions.

Step 3: inspect downloaded files:

```bash
/usr/bin/python inspect_downloads.py --downloads downloads --out outputs/download_manifest.csv
```

Step 4: optionally generate previews:

```bash
/usr/bin/python extract_glb_preview.py --manifest outputs/download_manifest.csv --out_dir previews_from_glb --size 300
```

Downloader outputs:

- `downloads/`: one folder per model, containing `source.glb`, `source.zip`, `extracted/`, `metadata.json`, and related files when available.
- `outputs/download_status.csv`: row-by-row download status.
- `outputs/download_log.txt`: readable log.
- `outputs/manual_required.csv`: rows that need manual action.
- `outputs/debug_screenshots/`: screenshots and HTML snapshots for failed/debug rows.

Repeated runs skip existing downloaded files unless `--overwrite` is passed.
