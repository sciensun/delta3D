import json
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from scene.label_style_deform_model import LabelStyleDeformModel
from utils.general_utils import safe_state
from utils.local_geometry_features import build_knn, compute_local_geometry_features


def prepare_output(args):
    if not args.model_path:
        args.model_path = os.path.join("./output", str(uuid.uuid4())[:10])
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args_style_distill"), "w") as f:
        f.write(str(Namespace(**vars(args))))


def load_delta_z(path, invert=False, device="cuda"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    names = cfg["label_names"]
    src = torch.tensor([cfg["source_style"][name] for name in names], dtype=torch.float32, device=device)
    tgt = torch.tensor([cfg["target_style"][name] for name in names], dtype=torch.float32, device=device)
    delta = tgt - src
    if invert:
        delta = -delta
    return delta, cfg


def smoothness_loss(xyz, d_xyz, k=16, sample=4096):
    try:
        n = xyz.shape[0]
        count = min(sample, n)
        idx = torch.randperm(n, device=xyz.device)[:count]
        nn_idx = build_knn(xyz, k=k, query_xyz=xyz[idx], chunk_size=512)
        return (d_xyz[idx, None, :] - d_xyz[nn_idx]).norm(dim=-1).mean()
    except Exception:
        return torch.zeros((), device=xyz.device)


def tensor_to_pil(image):
    image = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((image * 255.0).astype(np.uint8))


def save_interpolation(scene, gaussians, model, features, delta_z, pipe, background, args, is_6dof=False):
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    out_root = os.path.join(args.model_path, "style_interpolation")
    cameras = scene.getTrainCameras()
    with torch.no_grad():
        for alpha in alphas:
            d_xyz, d_scaling, _ = model(features, delta_z, alpha=alpha)
            d_rotation = torch.zeros((gaussians.get_xyz.shape[0], 4), dtype=d_xyz.dtype, device=d_xyz.device)
            alpha_dir = os.path.join(out_root, "alpha_{:03d}".format(int(round(alpha * 100))))
            os.makedirs(alpha_dir, exist_ok=True)
            for cam in cameras:
                image = render(cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, is_6dof)["render"]
                tensor_to_pil(image).save(os.path.join(alpha_dir, cam.image_name + ".png"))


def training(dataset, opt, pipe, args):
    prepare_output(args)
    point_cloud_dir = os.path.join(args.model_path, "point_cloud")
    if args.load_iteration and not os.path.isdir(point_cloud_dir):
        raise RuntimeError(
            "Stage 2 distillation needs a trained source 3DGS at '{}'. "
            "A GLB file cannot be loaded directly as GaussianModel. "
            "Run Stage 0 source 3DGS training first, then rerun this script.".format(point_cloud_dir)
        )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)

    payload = torch.load(args.mined_delta_path, map_location="cuda")
    mined_d_xyz = payload["d_xyz"].to("cuda")
    mined_d_scaling = payload["d_scaling"].to("cuda")
    if mined_d_xyz.shape[0] != gaussians.get_xyz.shape[0]:
        raise RuntimeError("Mined delta N does not match source Gaussian count.")

    delta_z, label_cfg = load_delta_z(args.label_config, invert=args.invert_delta_z)
    with torch.no_grad():
        features = compute_local_geometry_features(
            gaussians.get_xyz.detach(),
            scaling=gaussians.get_scaling.detach(),
            opacity=gaussians.get_opacity.detach(),
            k=args.feature_knn,
        ).detach()

    model = LabelStyleDeformModel(
        feature_dim=features.shape[1],
        delta_z_dim=delta_z.shape[0],
        max_d_xyz=args.max_d_xyz,
        max_d_scaling=args.max_d_scaling,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.distill_lr)

    n = features.shape[0]
    progress = tqdm(range(opt.iterations), desc="Style distill")
    for iteration in range(1, opt.iterations + 1):
        d_xyz, d_scaling, edit_mask = model(features, delta_z, alpha=args.alpha_train)
        fit_xyz = F.smooth_l1_loss(d_xyz, mined_d_xyz)
        fit_scaling = F.smooth_l1_loss(d_scaling, mined_d_scaling)
        loss = args.lambda_delta_fit * fit_xyz + args.lambda_scaling_fit * fit_scaling
        if args.lambda_smooth > 0:
            loss = loss + args.lambda_smooth * smoothness_loss(
                gaussians.get_xyz.detach(), d_xyz, k=args.feature_knn, sample=min(4096, n)
            )
        loss = loss + args.lambda_mask_sparse * edit_mask.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration % 10 == 0:
            progress.set_postfix({"loss": "{:.6f}".format(loss.item())})
            progress.update(10)

    progress.close()
    save_path = args.save_style_model_path or os.path.join(args.model_path, "label_style_deform_latest.pt")
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    model.save_weights(save_path)
    meta_path = os.path.splitext(save_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mined_delta_path": args.mined_delta_path,
                "label_config": args.label_config,
                "label_names": label_cfg["label_names"],
                "delta_z": delta_z.detach().cpu().tolist(),
                "feature_dim": features.shape[1],
            },
            f,
            indent=2,
        )

    if args.render_interpolation:
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        save_interpolation(scene, gaussians, model, features, delta_z, pipe, background, args, dataset.is_6dof)


if __name__ == "__main__":
    parser = ArgumentParser("Distill label-conditioned style deformation")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--mined_delta_path", required=True)
    parser.add_argument("--label_config", default="configs/style_labels_elephant.json")
    parser.add_argument("--invert_delta_z", action="store_true")
    parser.add_argument("--max_d_xyz", type=float, default=0.03)
    parser.add_argument("--max_d_scaling", type=float, default=0.08)
    parser.add_argument("--distill_lr", type=float, default=1e-3)
    parser.add_argument("--lambda_delta_fit", type=float, default=1.0)
    parser.add_argument("--lambda_scaling_fit", type=float, default=0.3)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_mask_sparse", type=float, default=0.01)
    parser.add_argument("--feature_knn", type=int, default=32)
    parser.add_argument("--alpha_train", type=float, default=1.0)
    parser.add_argument("--save_style_model_path", default=None)
    parser.add_argument("--render_interpolation", action="store_true")
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)
