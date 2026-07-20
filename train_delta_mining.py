import os
import sys
import uuid
import warnings
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from scene.free_delta_model import FreeDeltaModel
from scene.part_delta_model import PartDeltaModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss
from utils.mask_utils import image_to_mask, mask_loss
from utils.style_image_utils import find_style_target_path, load_style_target_rgba, match_image_size


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise ValueError("Expected boolean value.")


class OptionalLPIPS:
    def __init__(self, device="cuda"):
        self.loss_fn = None
        try:
            import lpips

            self.loss_fn = lpips.LPIPS(net="vgg").to(device).eval()
        except Exception as exc:
            warnings.warn("LPIPS unavailable; using weak RGB L1 only: {}".format(exc))

    @property
    def available(self):
        return self.loss_fn is not None

    def __call__(self, image, target):
        if self.loss_fn is None:
            return None
        return self.loss_fn(image[None].clamp(0, 1) * 2 - 1, target[None].clamp(0, 1) * 2 - 1).mean()


def prepare_output(args):
    if not args.model_path:
        args.model_path = os.path.join("./output", str(uuid.uuid4())[:10])
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args_delta_mining"), "w") as f:
        f.write(str(Namespace(**vars(args))))


def smoothness_loss(xyz, d_xyz, k=16, sample=4096, chunk=512):
    try:
        n = xyz.shape[0]
        if n <= 1:
            return torch.zeros((), device=xyz.device)
        count = min(sample, n)
        sample_idx = torch.randperm(n, device=xyz.device)[:count]
        vals = []
        for start in range(0, count, chunk):
            idx = sample_idx[start : start + chunk]
            dist = torch.cdist(xyz[idx].detach(), xyz.detach())
            nn_idx = torch.topk(dist, k=min(k + 1, n), largest=False).indices[:, 1:]
            vals.append((d_xyz[idx, None, :] - d_xyz[nn_idx]).norm(dim=-1).mean())
        return torch.stack(vals).mean()
    except Exception as exc:
        warnings.warn("Skipping smoothness loss because KNN failed: {}".format(exc))
        return torch.zeros((), device=xyz.device)


def masked_l1_loss(image, target, mask):
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask[None], size=image.shape[-2:], mode="nearest")[0]
    denom = mask.sum().clamp_min(1.0) * image.shape[0]
    return ((image - target).abs() * mask).sum() / denom


def load_optional_mask(path, expected_n, device="cuda"):
    if not path:
        return None
    mask = torch.load(path, map_location=device).bool().flatten()
    if mask.shape[0] != expected_n:
        raise ValueError("Mask length {} does not match Gaussian count {}.".format(mask.shape[0], expected_n))
    return mask.to(device=device)


def save_effective_delta(path, gaussians, d_xyz, d_rotation, d_scaling, metadata, extra=None):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "d_xyz": d_xyz.detach().cpu(),
        "d_scaling": d_scaling.detach().cpu(),
        "d_rotation": d_rotation.detach().cpu(),
        "source_xyz": gaussians.get_xyz.detach().cpu(),
        "metadata": metadata or {},
    }
    try:
        payload["source_scaling"] = gaussians.get_scaling.detach().cpu()
    except Exception:
        pass
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def valid_target_cameras(cameras, target_root):
    valid = []
    missing = []
    for cam in cameras:
        path = find_style_target_path(cam, target_root, required=False)
        if path:
            valid.append(cam)
        else:
            missing.append(cam.image_name)
    if missing:
        warnings.warn("Skipping {} views without weak targets: {}".format(len(missing), ", ".join(missing[:8])))
    if not valid:
        raise RuntimeError("No target images found in {}".format(target_root))
    return valid


def training(dataset, opt, pipe, args):
    prepare_output(args)
    point_cloud_dir = os.path.join(args.model_path, "point_cloud")
    if args.load_iteration and not os.path.isdir(point_cloud_dir):
        raise RuntimeError(
            "Stage 1 delta mining needs a trained source 3DGS at '{}'. "
            "Your source GLB in assets/3D is not a GaussianModel yet. "
            "First render/export a Blender dataset from the GLB, then run train.py to create "
            "{}/iteration_<N>/point_cloud.ply. See README_two_stage_delta_pipeline.md.".format(
                point_cloud_dir, point_cloud_dir
            )
        )
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, shuffle=False)
    if not args.freeze_gaussians:
        gaussians.training_setup(opt)

    num_gaussians = gaussians.get_xyz.shape[0]
    foreground_mask = load_optional_mask(args.foreground_mask_path, num_gaussians)
    foreground_mask_f = foreground_mask.float()[:, None] if foreground_mask is not None else None
    if args.part_labels_path:
        part_labels = torch.load(args.part_labels_path, map_location="cuda").long().flatten()
        if foreground_mask is None:
            foreground_mask = part_labels >= 0
            foreground_mask_f = foreground_mask.float()[:, None]
        delta_model = PartDeltaModel(
            gaussians.get_xyz.detach(),
            part_labels,
            foreground_mask=foreground_mask,
            num_parts=args.num_parts,
            max_d_xyz=args.max_d_xyz,
            max_d_scaling=args.max_d_scaling,
            assignment_temperature=args.part_assignment_temperature,
            disable_d_scaling=args.disable_d_scaling,
            enable_rotation=args.enable_delta_rotation,
        ).cuda()
    else:
        delta_model = FreeDeltaModel(
            num_gaussians,
            max_d_xyz=args.max_d_xyz,
            max_d_scaling=args.max_d_scaling,
            enable_rotation=args.enable_delta_rotation,
            disable_d_scaling=args.disable_d_scaling,
        ).cuda()
    optimizer = torch.optim.Adam(delta_model.parameters(), lr=args.free_delta_lr)
    lpips_loss = OptionalLPIPS("cuda") if args.weak_target else None

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_cameras = valid_target_cameras(scene.getTrainCameras(), args.target_image_root)
    viewpoint_stack = None
    progress = tqdm(range(opt.iterations), desc="Delta mining")

    for iteration in range(1, opt.iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()

        d_xyz, d_rotation, d_scaling = delta_model.step()
        if foreground_mask_f is not None:
            d_xyz = d_xyz * foreground_mask_f
            d_scaling = d_scaling * foreground_mask_f
            d_rotation = d_rotation * foreground_mask_f
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image = render_pkg["render"].clamp(0, 1)

        target_rgba = load_style_target_rgba(
            viewpoint_cam,
            args.target_image_root,
            device="cuda",
            required=False,
            composite_white=args.composite_target_white,
        )
        if target_rgba is None:
            continue
        target = match_image_size(target_rgba[:3], image).clamp(0, 1)

        with torch.no_grad():
            source_pkg = render(viewpoint_cam, gaussians, pipe, background, 0.0, 0.0, 0.0, dataset.is_6dof)
            source_mask = image_to_mask(source_pkg["render"].clamp(0, 1))
        target_mask = image_to_mask(target_rgba)
        render_mask = image_to_mask(image)
        mask_ref = source_mask if source_mask.sum() > 1 else target_mask
        if args.foreground_loss:
            rgb_mask = ((render_mask > 0.5) | (target_mask > 0.5) | (mask_ref > 0.5)).to(dtype=image.dtype)
            rgb_term = masked_l1_loss(image, target, rgb_mask)
        else:
            rgb_term = l1_loss(image, target)

        lpips_term = lpips_loss(image, target) if lpips_loss is not None and lpips_loss.available else None
        if args.weak_target and lpips_term is not None:
            loss = args.lambda_lpips * lpips_term + args.lambda_rgb_weak * rgb_term
        else:
            loss = args.lambda_rgb_weak * rgb_term
        loss = loss + args.lambda_mask * mask_loss(render_mask, mask_ref)

        if foreground_mask_f is not None and foreground_mask_f.sum() > 0:
            delta_reg = ((d_xyz ** 2).sum(dim=-1, keepdim=True) * foreground_mask_f).sum() / foreground_mask_f.sum()
            delta_reg = delta_reg + ((d_scaling ** 2).sum(dim=-1, keepdim=True) * foreground_mask_f).sum() / foreground_mask_f.sum()
        else:
            delta_reg = (d_xyz ** 2).mean() + (d_scaling ** 2).mean()
        loss = loss + args.lambda_delta * delta_reg
        if not args.disable_d_scaling and args.scale_positive_penalty > 0:
            loss = loss + args.scale_positive_penalty * (F.relu(d_scaling) ** 2).mean()
        if args.lambda_smooth > 0:
            loss = loss + args.lambda_smooth * smoothness_loss(
                gaussians.get_xyz, d_xyz, k=args.smooth_knn, sample=args.smooth_sample
            )

        optimizer.zero_grad(set_to_none=True)
        if gaussians.optimizer is not None:
            gaussians.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not args.freeze_gaussians and gaussians.optimizer is not None:
            gaussians.optimizer.step()
            gaussians.update_learning_rate(iteration)

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device("cpu")
        if iteration % 10 == 0:
            progress.set_postfix({"loss": "{:.6f}".format(loss.item())})
            progress.update(10)
        if iteration in args.save_iterations or iteration == opt.iterations:
            d_xyz, d_rotation, d_scaling = delta_model.step()
            if foreground_mask_f is not None:
                d_xyz = d_xyz * foreground_mask_f
                d_scaling = d_scaling * foreground_mask_f
                d_rotation = d_rotation * foreground_mask_f
            metadata = {
                "object_id": args.object_id,
                "direction": args.direction,
                "target_root": args.target_image_root,
                "iteration": iteration,
                "loss_weights": {
                    "lambda_lpips": args.lambda_lpips,
                    "lambda_rgb_weak": args.lambda_rgb_weak,
                    "lambda_mask": args.lambda_mask,
                    "lambda_delta": args.lambda_delta,
                    "lambda_smooth": args.lambda_smooth,
                    "scale_positive_penalty": args.scale_positive_penalty,
                },
                "disable_d_scaling": args.disable_d_scaling,
                "composite_target_white": args.composite_target_white,
                "foreground_loss": args.foreground_loss,
                "foreground_mask_path": args.foreground_mask_path,
                "part_labels_path": args.part_labels_path,
                "num_parts": args.num_parts,
                "part_assignment_temperature": args.part_assignment_temperature,
            }
            latest = args.save_delta_path or os.path.join(args.model_path, "mined_delta_latest.pt")
            extra = {}
            if foreground_mask is not None:
                extra["foreground_mask"] = foreground_mask.detach().cpu()
            if isinstance(delta_model, PartDeltaModel):
                _, _, _, part_d_xyz, part_d_scaling, part_d_rotation = delta_model.forward()
                extra.update(
                    {
                        "part_d_xyz": part_d_xyz.detach().cpu(),
                        "part_d_scaling": part_d_scaling.detach().cpu(),
                        "part_d_rotation": part_d_rotation.detach().cpu(),
                        "part_weights": delta_model.part_weights.detach().cpu(),
                        "part_labels": delta_model.part_labels.detach().cpu(),
                    }
                )
            save_effective_delta(latest, gaussians, d_xyz, d_rotation, d_scaling, metadata, extra)
            save_effective_delta(
                os.path.join(args.model_path, "mined_delta_iter_{}.pt".format(iteration)),
                gaussians,
                d_xyz,
                d_rotation,
                d_scaling,
                metadata,
                extra,
            )

    progress.close()


if __name__ == "__main__":
    parser = ArgumentParser("Free delta mining")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--target_image_root", required=True)
    parser.add_argument("--object_id", default="big_carved_wooden_elephant_sculpture")
    parser.add_argument("--direction", default="stylized_to_standard")
    parser.add_argument("--max_d_xyz", type=float, default=0.03)
    parser.add_argument("--max_d_scaling", type=float, default=0.08)
    parser.add_argument("--free_delta_lr", type=float, default=1e-3)
    parser.add_argument("--lambda_lpips", type=float, default=1.0)
    parser.add_argument("--lambda_rgb_weak", type=float, default=0.02)
    parser.add_argument("--lambda_mask", type=float, default=0.1)
    parser.add_argument("--lambda_delta", type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--smooth_knn", type=int, default=16)
    parser.add_argument("--smooth_sample", type=int, default=4096)
    parser.add_argument("--save_delta_path", default=None)
    parser.add_argument("--freeze_gaussians", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--enable_delta_rotation", action="store_true", default=False)
    parser.add_argument("--disable_d_scaling", action="store_true", default=False)
    parser.add_argument("--scale_positive_penalty", type=float, default=0.0)
    parser.add_argument("--weak_target", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--composite_target_white", action="store_true")
    parser.add_argument("--foreground_loss", action="store_true")
    parser.add_argument("--foreground_mask_path", default=None)
    parser.add_argument("--part_labels_path", default=None)
    parser.add_argument("--num_parts", type=int, default=16)
    parser.add_argument("--part_assignment_temperature", type=float, default=0.15)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 2000, 3000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(False)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)
