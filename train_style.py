#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
import warnings
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.style_deform_model import StyleDeformModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.style_image_utils import (
    foreground_mask_from_rgba_or_rgb,
    load_style_target_image,
    load_style_target_rgba,
    match_image_size,
    render_foreground_mask,
)
from utils.style_reg_utils import deformation_l2_reg, identity_deform_reg
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise ValueError("Expected a boolean value.")


class OptionalLPIPSLoss:
    def __init__(self, device="cuda"):
        self.loss_fn = None
        try:
            import lpips

            self.loss_fn = lpips.LPIPS(net="vgg").to(device)
            self.loss_fn.eval()
        except Exception as exc:
            warnings.warn(
                "LPIPS is unavailable ({}). Weak-target training will fall back to low-weight L1.".format(exc),
                RuntimeWarning,
            )

    @property
    def available(self):
        return self.loss_fn is not None

    def __call__(self, image, target):
        if self.loss_fn is None:
            return None
        image = image.clamp(0.0, 1.0).unsqueeze(0) * 2.0 - 1.0
        target = target.clamp(0.0, 1.0).unsqueeze(0) * 2.0 - 1.0
        return self.loss_fn(image, target).mean()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, args):
    tb_writer = prepare_output_and_logger(args)
    dataset.model_path = args.model_path
    gaussians = GaussianModel(dataset.sh_degree)
    style_deform = StyleDeformModel(
        style_names=args.style_names,
        style_dim=args.style_dim,
        num_freqs=args.style_num_freqs,
        hidden_dim=args.style_hidden_dim,
        depth=args.style_depth,
        max_d_xyz=args.max_d_xyz,
        max_d_scaling=args.max_d_scaling,
        enable_rotation=args.enable_style_rotation,
    ).cuda()
    style_deform.train_setting(args)
    lpips_loss = OptionalLPIPSLoss(device="cuda") if args.weak_target else None

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling, edit_mask = 0.0, 0.0, 0.0, None
        else:
            N = gaussians.get_xyz.shape[0]
            alpha = torch.full((N, 1), args.style_alpha, device="cuda")
            # Scheme C: time-conditioned deformation (fid/time_input) is replaced by
            # a style-conditioned deformation driven by style_name, style embedding, and alpha.
            d_xyz, d_rotation, d_scaling, edit_mask = style_deform.step(
                gaussians.get_xyz.detach(), style_name=args.style_name, alpha=alpha
            )

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        # depth = render_pkg_re["depth"]

        # Loss
        # Generated stylized images are weak style references, not pixel-aligned ground truth.
        # By default we avoid RGB L1/SSIM as the main supervision and use LPIPS + source-mask consistency.
        style_rgba = load_style_target_rgba(viewpoint_cam, args.style_target_path, split="train", device="cuda")
        style_gt = match_image_size(style_rgba[:3], image)
        target_mask = foreground_mask_from_rgba_or_rgb(style_rgba)
        Ll1 = l1_loss(image, style_gt)

        if args.weak_target:
            lpips_term = lpips_loss(image, style_gt) if lpips_loss is not None and lpips_loss.available else None
            if lpips_term is None:
                style_loss = args.lambda_rgb_weak * Ll1
            else:
                style_loss = args.lambda_lpips * lpips_term
                if args.use_rgb_weak:
                    style_loss = style_loss + args.lambda_rgb_weak * Ll1

            with torch.no_grad():
                source_pkg = render(viewpoint_cam, gaussians, pipe, background, 0.0, 0.0, 0.0, dataset.is_6dof)
                source_mask = render_foreground_mask(source_pkg["render"], background)
            render_mask = render_foreground_mask(image, background)
            # The target mask is extracted for inspection/filtering, but the consistency
            # term uses the source render mask because generated targets are only weakly paired.
            target_mask = match_image_size(target_mask, render_mask)
            mask_loss = F.l1_loss(render_mask, source_mask)
            loss = args.lambda_style * style_loss + args.lambda_mask * mask_loss
        else:
            style_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, style_gt))
            mask_loss = torch.zeros((), device="cuda")
            loss = args.lambda_style * style_loss

        if edit_mask is not None:
            loss = loss + args.lambda_delta * deformation_l2_reg(d_xyz, d_scaling, edit_mask)
        if iteration >= opt.warm_up and args.lambda_id > 0.0 and iteration % args.id_every == 0:
            # Identity regularization keeps alpha=0 as the canonical, undeformed 3DGS state.
            loss = loss + args.lambda_id * identity_deform_reg(
                style_deform, gaussians.get_xyz.detach(), args.style_name
            )
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), style_deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof, args)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                style_deform.save_weights(args.model_path, iteration)

            # Densification
            if not args.freeze_gaussians and iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                if not args.freeze_gaussians:
                    gaussians.optimizer.step()
                    gaussians.update_learning_rate(iteration)
                style_deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                style_deform.optimizer.zero_grad()
                style_deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, style_deform, load2gpu_on_the_fly, is_6dof=False, args=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    xyz = scene.gaussians.get_xyz
                    alpha = torch.full((xyz.shape[0], 1), args.style_alpha, device="cuda")
                    # Scheme C validation path: render the style-conditioned deformation, not fid/time deformation.
                    d_xyz, d_rotation, d_scaling, _ = style_deform.step(
                        xyz.detach(), style_name=args.style_name, alpha=alpha
                    )
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    if args is not None and config["name"] == "train":
                        gt_image = torch.clamp(
                            load_style_target_image(viewpoint, args.style_target_path, split="train", device="cuda"),
                            0.0,
                            1.0,
                        )
                        gt_image = match_image_size(gt_image, image)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--style_target_path", type=str, required=True)
    parser.add_argument("--style_name", type=str, default="round")
    parser.add_argument("--style_names", type=str, default="round")
    parser.add_argument("--style_alpha", type=float, default=1.0)
    parser.add_argument("--style_dim", type=int, default=16)
    parser.add_argument("--style_num_freqs", type=int, default=6)
    parser.add_argument("--style_hidden_dim", type=int, default=128)
    parser.add_argument("--style_depth", type=int, default=4)
    parser.add_argument("--style_deform_lr", type=float, default=1e-3)
    parser.add_argument("--max_d_xyz", type=float, default=0.05)
    parser.add_argument("--max_d_scaling", type=float, default=0.10)
    parser.add_argument("--enable_style_rotation", action="store_true", default=False)
    parser.add_argument("--lambda_style", type=float, default=1.0)
    parser.add_argument("--lambda_delta", type=float, default=1e-3)
    parser.add_argument("--lambda_id", type=float, default=0.1)
    parser.add_argument("--id_every", type=int, default=10)
    parser.add_argument("--freeze_gaussians", action="store_true")
    parser.add_argument("--weak_target", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--lambda_lpips", type=float, default=1.0)
    parser.add_argument("--lambda_rgb_weak", type=float, default=0.05)
    parser.add_argument("--lambda_mask", type=float, default=0.1)
    parser.add_argument("--use_rgb_weak", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args)

    # All done
    print("\nTraining complete.")
