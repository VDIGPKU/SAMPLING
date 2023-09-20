import os
import glob
import models
import lpips
import model_io
import models
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loss import SILogLoss, BinsChamferLoss
from utils import restore_model
from utils import run_shell_cmd
from utils import get_embedder
from utils import AverageMeter
from utils import inverse
from utils import disparity_normalization_vis
from network.ssim import SSIM
from network.layers import edge_aware_loss
from network.layers import edge_aware_loss_v2
from network.layers import psnr
from operations import rendering_utils
from operations import mpi_rendering
from operations.homography_sampler import HomographySample
from models.PAN import DepthPredictionNetwork
from models.CPN.unet import FeatMaskNetwork
from network.monodepth2.resnet_encoder import ResnetEncoder
from network.monodepth2.depth_decoder import DepthDecoder


def _get_disparity_list(config, B, device=torch.device("cuda:0")):
    S_coarse, S_fine = config["mpi.num_bins_coarse"], config["mpi.num_bins_fine"]
    disparity_start, disparity_end = config["mpi.disparity_start"], config["mpi.disparity_end"]

    if config.get("mpi.fix_disparity", False):
        if len(config.get("mpi.disparity_list", torch.zeros((1)))) == S_coarse + 1:
            disparity_coarse_src = torch.from_numpy(config["mpi.disparity_list"][1:]).to(
                    dtype=torch.float32, device=device
            ).unsqueeze(0).repeat(B, 1)   
        else:
            disparity_coarse_src = torch.linspace(
                    disparity_start, disparity_end, S_coarse, dtype=torch.float32,
                    device=device
            ).unsqueeze(0).repeat(B, 1)   
    else:
        if len(config.get("mpi.disparity_list", torch.zeros((1)))) == S_coarse + 1:
            disparity_coarse_src = rendering_utils.uniformly_sample_disparity_from_bins(
                    batch_size=B,
                    disparity_np=config["mpi.disparity_list"],
                    device=device
            )
        else:
            disparity_coarse_src = rendering_utils.uniformly_sample_disparity_from_linspace_bins(
                    batch_size=B,
                    num_bins=S_coarse,
                    start=disparity_start,
                    end=disparity_end,
                    device=device
            )
    return disparity_coarse_src


class SynthesisTask():
    def __init__(self, config, logger, is_val=False):
        self.embedder, out_dim = get_embedder(config["model.pos_encoding_multires"])
        self.min_depth = config["mpi.disparity_end"]
        self.w_chamfer = 0.1

        self.backbone = ResnetEncoder(num_layers=50,
                                      pretrained=config.get("model.imagenet_pretrained", True)).to(
                device=torch.device("cuda:0"))
        self.pan = DepthPredictionNetwork([0.0001,80]).to(device=torch.device("cuda:0"))
        self.mnet = FeatMaskNetwork().to(device=torch.device("cuda:0"))
        self.decoder = DepthDecoder(
    
                num_ch_enc=self.backbone.num_ch_enc,
                use_alpha=config.get("mpi.use_alpha", False),
                num_output_channels=4,
                scales=range(4),
                use_skips=True,
                embedder=self.embedder,
                embedder_out_dim=out_dim,
        ).to(device=torch.device("cuda:0"))
        print("pan", self.pan)
        print("mnet", self.mnet)
        print("backbone",self.backbone)
        print("decoder",self.decoder)
        params = [
            {"params": self.backbone.parameters(), "lr": config["lr.backbone_lr"]},
            {"params": self.decoder.parameters(), "lr": config["lr.decoder_lr"]},
            {"params": self.pan.parameters(), "lr": config["lr.pan_lr"]},
            {"params": self.mnet.parameters(), "lr": config["lr.pan_lr"]}
        ]
        self.optimizer = torch.optim.Adam(params, weight_decay=config["lr.weight_decay"])
        if config["global_rank"] == 0:
            self.lpips = lpips.LPIPS(net="vgg").cuda()
            self.lpips.requires_grad = False

            if config["training.pretrained_checkpoint_path"] and \
                    config["training.pretrained_checkpoint_path"].startswith("hdfs"):
                run_shell_cmd(["hdfs", "dfs", "-get", config["training.pretrained_checkpoint_path"], "."],
                              logger)
                config["training.pretrained_checkpoint_path"] = os.path.basename(
                        config["training.pretrained_checkpoint_path"])

            restore_model(config["training.pretrained_checkpoint_path"],
                          self.backbone, self.decoder, self.optimizer,
                          logger=logger)

        if not is_val:
            process_group = torch.distributed.new_group(range(dist.get_world_size()))
            self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone, process_group)
            self.backbone = DDP(self.backbone, find_unused_parameters=True)
            self.backbone.train()

            self.decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder, process_group)
            self.decoder = DDP(self.decoder, find_unused_parameters=True)
            self.decoder.train()

            self.pan = nn.SyncBatchNorm.convert_sync_batchnorm(self.pan, process_group)
            self.pan = DDP(self.pan, find_unused_parameters=True)
            self.pan.train()
            self.mnet = nn.SyncBatchNorm.convert_sync_batchnorm(self.mnet, process_group)
            self.mnet = DDP(self.mnet, find_unused_parameters=True)
            self.mnet.train()
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                               config["lr.decay_steps"],
                                                               gamma=config["lr.decay_gamma"])
        else:
            self.backbone = nn.DataParallel(self.backbone)
            self.decoder = nn.DataParallel(self.decoder)
            self.pan = nn.DataParallel(self.pan)
            self.mnet = nn.DataParallel(self.mnet)

        H_tgt, W_tgt = config["data.img_h"], config["data.img_w"]
        self.homography_sampler_list = \
            [HomographySample(H_tgt, W_tgt, device=torch.device("cuda:0")),
             HomographySample(int(H_tgt / 2), int(W_tgt / 2), device=torch.device("cuda:0")),
             HomographySample(int(H_tgt / 4), int(W_tgt / 4), device=torch.device("cuda:0")),
             HomographySample(int(H_tgt / 8), int(W_tgt / 8), device=torch.device("cuda:0"))]
        self.upsample_list = \
            [nn.Identity(),
             nn.Upsample(size=(int(H_tgt / 2), int(W_tgt / 2))),
             nn.Upsample(size=(int(H_tgt / 4), int(W_tgt / 4))),
             nn.Upsample(size=(int(H_tgt / 8), int(W_tgt / 8)))]

        self.ssim = SSIM(size_average=True).cuda()

        self.config = config
        self.tb_writer = config.get("tb_writer", None)
        self.logger = logger

        self.init_data(torch.device("cuda:0"))
        self.train_losses = {
            "loss": AverageMeter("train_loss"),
            "loss_rgb_src": AverageMeter("train_loss_rgb_src"),
            "loss_ssim_src": AverageMeter("train_loss_ssim_src"),
            "loss_rgb_tgt": AverageMeter("train_loss_rgb_tgt"),
            "loss_ssim_tgt": AverageMeter("train_loss_ssim_tgt"),
            "lpips_tgt": AverageMeter("train_lpips_tgt"),
            "psnr_tgt": AverageMeter("train_psnr_tgt"),
        }
        self.val_losses = {
            "loss_rgb_src": AverageMeter("val_loss_rgb_src"),
            "loss_ssim_src": AverageMeter("val_loss_ssim_src"),
            "loss_rgb_tgt": AverageMeter("val_loss_rgb_tgt"),
            "loss_ssim_tgt": AverageMeter("val_loss_ssim_tgt"),
            "lpips_tgt": AverageMeter("val_lpips_tgt"),
            "psnr_tgt": AverageMeter("val_psnr_tgt"),
        }

        self.current_epoch = 0
        self.global_step = 0

    def init_data(self, device):
        B, H, W = self.config["data.per_gpu_batch_size"], self.config["data.img_h"], self.config["data.img_w"]
        L = self.config["data.num_tgt_views"]
        self.src_imgs = torch.zeros((B, 3, H, W), dtype=torch.float32, device=device)
        self.K_src = torch.zeros((B, 3, 3), dtype=torch.float32, device=device)
        self.K_src_inv = torch.zeros((B, 3, 3), dtype=torch.float32, device=device)
        self.tgt_imgs = torch.zeros((B, 3, H, W), dtype=torch.float32, device=device)
        self.G_src_tgt = torch.zeros((B, 4, 4), dtype=torch.float32, device=device)
        self.K_tgt = torch.zeros((B, 3, 3), dtype=torch.float32, device=device)
        self.K_tgt_inv = torch.zeros((B, 3, 3), dtype=torch.float32, device=device)

    def set_data(self, src_items, tgt_items):
  
        self.src_imgs.resize_as_(src_items["img"]).copy_(src_items["img"])   
        self.K_src.resize_as_(src_items["K"]).copy_(src_items["K"])   
        self.K_src_inv.resize_as_(src_items["K_inv"]).copy_(src_items["K_inv"])

        self.tgt_imgs.resize_as_(tgt_items["img"]).copy_(tgt_items["img"])   
        self.G_src_tgt.resize_as_(tgt_items["G_src_tgt"]).copy_(tgt_items["G_src_tgt"])   
        self.K_tgt.resize_as_(tgt_items["K"]).copy_(tgt_items["K"])   
        self.K_tgt_inv.resize_as_(tgt_items["K_inv"]).copy_(tgt_items["K_inv"])   

        self.G_tgt_src = inverse(self.G_src_tgt)
        torch.cuda.synchronize()

    def compute_scale_factor(self, disparity_syn_pt3dsrc, pt3d_disp_src):
        B = pt3d_disp_src.size()[0]
        if self.config["data.name"] in ["kitti_raw", "TT"]:
            return torch.ones(B, dtype=torch.float32).cuda()

         
        scale_factor = torch.exp(torch.mean(
                torch.log(disparity_syn_pt3dsrc) - torch.log(pt3d_disp_src),
                dim=2, keepdim=False)).squeeze(1)   
        return scale_factor

    def mpi_predictor(self, src_imgs_BCHW, disparity_BS):
         
        conv1_out, block1_out, block2_out, block3_out, block4_out = self.backbone(src_imgs_BCHW)
        outputs = self.decoder([conv1_out, block1_out, block2_out, block3_out, block4_out],
                               disparity_BS)
        output_list = [outputs[("disp", 0)], outputs[("disp", 1)], outputs[("disp", 2)], outputs[("disp", 3)]]
        return output_list

    def loss_fcn_per_scale(self, scale,
                           mpi_all_src, disparity_all_src,
                           scale_factor=None,
                           is_val=False):

        src_imgs_scaled = self.upsample_list[scale](self.src_imgs)
        tgt_imgs_scaled = self.upsample_list[scale](self.tgt_imgs)
        B, _, H_img_scaled, W_img_scaled = src_imgs_scaled.size()

        K_src_scaled = self.K_src / (2 ** scale)
        K_src_scaled[:, 2, 2] = 1
        K_tgt_scaled = self.K_tgt / (2 ** scale)
        K_tgt_scaled[:, 2, 2] = 1
        torch.cuda.synchronize()
        K_src_scaled_inv = torch.inverse(K_src_scaled)

        assert mpi_all_src.size(3) == H_img_scaled, mpi_all_src.size(4) == W_img_scaled
        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
                self.homography_sampler_list[scale].meshgrid,
                disparity_all_src,
                K_src_scaled_inv
        )

        mpi_all_rgb_src = mpi_all_src[:, :, 0:3, :, :]  
        mpi_all_sigma_src = mpi_all_src[:, :, 3:, :, :] 
        src_imgs_syn, src_depth_syn, blend_weights, weights = mpi_rendering.render(
                mpi_all_rgb_src,
                mpi_all_sigma_src,
                xyz_src_BS3HW,
                use_alpha=self.config.get("mpi.use_alpha", False),
                is_bg_depth_inf=self.config.get("mpi.render_tgt_rgb_depth", False)
        )
        if self.config.get("training.src_rgb_blending", True):
            mpi_all_rgb_src = blend_weights * src_imgs_scaled.unsqueeze(1) + (1 - blend_weights) * mpi_all_rgb_src
            src_imgs_syn, src_depth_syn = mpi_rendering.weighted_sum_mpi(
                    mpi_all_rgb_src,
                    xyz_src_BS3HW,
                    weights,
                    is_bg_depth_inf=self.config.get("mpi.render_tgt_rgb_depth", False)
            )
        src_disparity_syn = torch.reciprocal(src_depth_syn)


        scale_factor = torch.ones(B, dtype=torch.float32).cuda()   

        render_results = self.render_novel_view(mpi_all_rgb_src, mpi_all_sigma_src,
                                                disparity_all_src, self.G_tgt_src,
                                                K_src_scaled_inv, K_tgt_scaled,
                                                scale=scale,
                                                scale_factor=scale_factor)
        tgt_imgs_syn = render_results["tgt_imgs_syn"]
        tgt_disparity_syn = render_results["tgt_disparity_syn"]
        tgt_mask_syn = render_results["tgt_mask_syn"]

        disp_lambda = 0.0 if self.config["data.name"] in ["kitti_raw", "TT"] else 1.0
        smoothness_lambda_v1 = self.config.get("loss.smoothness_lambda_v1", 0.5)
        smoothness_lambda_v2 = self.config.get("loss.smoothness_lambda_v2", 1.0)

        with torch.no_grad():
            loss_rgb_src = torch.mean(torch.abs(src_imgs_syn - src_imgs_scaled))
            loss_ssim_src = 1 - self.ssim(src_imgs_syn, src_imgs_scaled)
            loss_smooth_src = edge_aware_loss(src_imgs_scaled, src_disparity_syn,
                                              gmin=self.config["loss.smoothness_gmin"],
                                              grad_ratio=self.config.get("loss.smoothness_grad_ratio", 0.1))

        rgb_tgt_valid_mask = torch.ge(tgt_mask_syn, self.config["mpi.valid_mask_threshold"]).to(torch.float32)
        loss_map = torch.abs(tgt_imgs_syn - tgt_imgs_scaled) * rgb_tgt_valid_mask
        loss_rgb_tgt = loss_map.mean()

        loss_smooth_tgt = smoothness_lambda_v1 * edge_aware_loss(
                tgt_imgs_scaled,
                tgt_disparity_syn,
                gmin=self.config["loss.smoothness_gmin"],
                grad_ratio=self.config.get("loss.smoothness_grad_ratio", 0.1))
        loss_smooth_tgt_v2 = smoothness_lambda_v2 * edge_aware_loss_v2(tgt_imgs_scaled, tgt_disparity_syn)
        loss_smooth_src_v2 = smoothness_lambda_v2 * edge_aware_loss_v2(src_imgs_scaled, src_disparity_syn)
        loss_ssim_tgt = 1 - self.ssim(tgt_imgs_syn, tgt_imgs_scaled)

        with torch.no_grad():
            lpips_tgt = self.lpips(tgt_imgs_syn, tgt_imgs_scaled).mean() \
                if (is_val and scale == 0) \
                else torch.tensor(0.0)

            psnr_tgt = psnr(tgt_imgs_syn, tgt_imgs_scaled).mean()

        loss = loss_rgb_tgt + loss_ssim_tgt \
               + loss_smooth_tgt \
               + loss_smooth_src_v2 + loss_smooth_tgt_v2

        loss_dict = {"loss": loss,
                     "loss_rgb_src": loss_rgb_src,
                     "loss_ssim_src": loss_ssim_src,
                     "loss_smooth_src": loss_smooth_src,
                     "loss_smooth_tgt": loss_smooth_tgt,
                     "loss_smooth_src_v2": loss_smooth_src_v2,
                     "loss_smooth_tgt_v2": loss_smooth_tgt_v2,
                     "loss_rgb_tgt": loss_rgb_tgt,
                     "loss_ssim_tgt": loss_ssim_tgt,
                     "lpips_tgt": lpips_tgt,
                     "psnr_tgt": psnr_tgt,
                     }

        visualization_dict = {"src_disparity_syn": src_disparity_syn,
                              "tgt_disparity_syn": tgt_disparity_syn,
                              "tgt_imgs_syn": tgt_imgs_syn,
                              "tgt_mask_syn": tgt_mask_syn,
                              "src_imgs_syn": src_imgs_syn}

        return loss_dict, visualization_dict, scale_factor

    def loss_fcn(self, is_val):
        loss_dict_list, visualization_dict_list = [], []

        endpoints = self.network_forward()

        scale_factor = None
        scale_list = list(range(4))
        for scale in scale_list:
            loss_dict_tmp, visualization_dict_tmp, scale_factor = self.loss_fcn_per_scale(
                    scale,
                    endpoints["mpi_all_src_list"][scale],
                    endpoints["disparity_all_src"],
                    scale_factor,
                    is_val=is_val
            )
            loss_dict_list.append(loss_dict_tmp)
            visualization_dict_list.append(visualization_dict_tmp)

        loss_dict = loss_dict_list[0]
        visualization_dict = visualization_dict_list[0]
        for scale in scale_list[1:]:
            if self.config.get("training.use_multi_scale", True):
                loss_dict["loss"] += (loss_dict_list[scale]["loss_rgb_tgt"] + loss_dict_list[scale]["loss_ssim_tgt"])

            loss_dict["loss"] += (
                    loss_dict_list[scale]["loss_smooth_src_v2"] + loss_dict_list[scale]["loss_smooth_tgt_v2"])
        loss_dict["loss"] += endpoints["loss_ada"]
        return loss_dict, visualization_dict

    def network_forward(self):
        B, H_img, W_img = self.src_imgs.size(0), self.src_imgs.size(2), self.src_imgs.size(3)
        L = self.tgt_imgs.size(1)
        S_fine = self.config["mpi.num_bins_fine"]
        loss_rank = 0
        loss_assign = 0
        disparity_coarse_src = _get_disparity_list(self.config, B, device=self.src_imgs.device)

        disparity_coarse_src_ada = self.pan(disparity_coarse_src, F.interpolate(self.img.float(), scale_factor=0.25), F.interpolate(self.depth.float(), scale_factor=0.25))
        mask = self.mnet(self.img, self.depth, disparity_coarse_src_ada)
        _, S = disparity_coarse_src_ada.shape
        loss_assign_matrix = torch.zeros(size=[B,S,H_img, W_img])
        for i in range(B):
            for j in range(S):
                if disparity_coarse_src_ada[i][j] >= 1:
                    disparity_coarse_src_ada[i][j] = 0.9999
                if j != S -1:
                    loss_rank += max(0, float(disparity_coarse_src_ada[i][j+1] - disparity_coarse_src_ada[i][j]))
                disparity_per_element = torch.ones(size=[H_img, W_img]).cuda() * disparity_coarse_src_ada[i][j]
                disparity_sub_depth =  self.depth[i,0,:,:]/80 - disparity_per_element
                loss_assign_matrix[i,j] = torch.mul(mask[i][j], disparity_sub_depth.abs())
        loss_assign = loss_assign_matrix.sum()
        loss_rank = 100 * loss_rank / (S - 1)
        loss_assign = 10 * loss_assign / (H_img * W_img)
        xyz_src_BS3HW_coarse = mpi_rendering.get_src_xyz_from_plane_disparity(
                self.homography_sampler_list[0].meshgrid,
                disparity_coarse_src_ada,
                self.K_src_inv
        )

         
        mpi_all_src_list, disparity_all_src = mpi_rendering.predict_mpi_coarse_to_fine(
                self.mpi_predictor,
                self.src_imgs,
                xyz_src_BS3HW_coarse,
                disparity_coarse_src_ada,
                S_fine,
                is_bg_depth_inf=self.config.get("mpi.render_tgt_rgb_depth", False)
        )

        return {
            "mpi_all_src_list": mpi_all_src_list,
            "disparity_all_src": disparity_all_src,
            "loss_ada": loss_rank + loss_assign
        }

    def render_novel_view(self, mpi_all_rgb_src, mpi_all_sigma_src,
                          disparity_all_src, G_tgt_src,
                          K_src_inv, K_tgt, scale=0, scale_factor=None):

        if scale_factor is not None:
            with torch.no_grad():
                G_tgt_src = torch.clone(G_tgt_src)
                G_tgt_src[:, 0:3, 3] = G_tgt_src[:, 0:3, 3] / scale_factor.view(-1, 1)

        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
                self.homography_sampler_list[scale].meshgrid,
                disparity_all_src,
                K_src_inv
        )

        xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
                xyz_src_BS3HW,
                G_tgt_src
        )

         
        tgt_imgs_syn, tgt_depth_syn, tgt_mask_syn = mpi_rendering.render_tgt_rgb_depth(
                self.homography_sampler_list[scale],
                mpi_all_rgb_src,
                mpi_all_sigma_src,
                disparity_all_src,
                xyz_tgt_BS3HW,
                G_tgt_src,
                K_src_inv,
                K_tgt,
                use_alpha=self.config.get("mpi.use_alpha", False),
                is_bg_depth_inf=self.config.get("mpi.render_tgt_rgb_depth", False)
        )
        tgt_disparity_syn = torch.reciprocal(tgt_depth_syn)

        return {
            "tgt_imgs_syn": tgt_imgs_syn,
            "tgt_disparity_syn": tgt_disparity_syn,
            "tgt_mask_syn": tgt_mask_syn
        }

    def run_eval(self, val_data_loader):
        self.logger.info("Start running evaluation on validation set:")
        self.backbone.eval()
        self.decoder.eval()

        for val_loss_item in self.val_losses.values():
            val_loss_item.reset()

        batch_count = 0
        with torch.no_grad():
            for step, (src_items, tgt_items, depth, img_raw, depth_raw) in enumerate(val_data_loader):
                batch_count += 1
                if self.config.get("global_rank", 0) == 0 and batch_count % 1000 == 0:
                    self.logger.info("    Eval progress: {}/{}".format(batch_count,
                                                                       len(val_data_loader)))

                self.set_data(src_items, tgt_items)
                loss_dict, visualization_dict = self.loss_fcn(is_val=True)
                self.log_val(step, loss_dict, visualization_dict)

            self.logger.info("Evaluation finished, average losses: ")
            for v in self.val_losses.values():
                self.logger.info("    {}".format(v))

            for key, value in self.val_losses.items():
                self.tb_writer.add_scalar(key + "/val", value.avg, self.global_step)

        self.backbone.train()
        self.decoder.train()

    def log_val(self, step, loss_dict, visualization_dict):
        B, H_img, W_img = self.src_imgs.size(0), self.src_imgs.size(2), self.src_imgs.size(3)
        L = 1

        for key, value in self.val_losses.items():
            value.update(loss_dict[key].item(), n=B)


        if self.global_step == self.config["training.eval_interval"]:
            src_imgs_BL = self.src_imgs.unsqueeze(1).repeat(1, L, 1, 1, 1).reshape(B * L, 3, H_img,
                                                                                   W_img).contiguous()
            src_imgs_BL_grid = torchvision.utils.make_grid(src_imgs_BL)
            self.tb_writer.add_image("00_src_images", src_imgs_BL_grid, step)

            tgt_imgs_BL = self.tgt_imgs.reshape(B * L, 3, H_img, W_img).contiguous()
            gt_tgt_grid = torchvision.utils.make_grid(tgt_imgs_BL)
            self.tb_writer.add_image("01_gt_tgt_images", gt_tgt_grid, step)

        syn_src_grid = torchvision.utils.make_grid(visualization_dict["src_imgs_syn"])
        self.tb_writer.add_image(
                "02_syn_src_images/step_%d" % (self.global_step), syn_src_grid, step)

        syn_src_disp_grid = torchvision.utils.make_grid(
                disparity_normalization_vis(visualization_dict["src_disparity_syn"])
        )
        self.tb_writer.add_image(
                "03_syn_src_disparity_map/step_%d" % (self.global_step), syn_src_disp_grid, step)

         
        syn_tgt_grid = torchvision.utils.make_grid(visualization_dict["tgt_imgs_syn"])
        self.tb_writer.add_image(
                "04_syn_tgt_images/step_%d" % (self.global_step), syn_tgt_grid, step)

        syn_tgt_disp_grid = torchvision.utils.make_grid(
                disparity_normalization_vis(visualization_dict["tgt_disparity_syn"])
        )
        self.tb_writer.add_image(
                "05_syn_tgt_disparity_map/step_%d" % (self.global_step), syn_tgt_disp_grid, step)

    def log_training(self, epoch, step, global_step, dataset_length, loss_dict):
        loss = loss_dict["loss"]
        loss_rgb_tgt = loss_dict["loss_rgb_tgt"]
        loss_ssim_tgt = loss_dict["loss_ssim_tgt"]
        loss_rgb_src = loss_dict["loss_rgb_src"]
        loss_ssim_src = loss_dict["loss_ssim_src"]
        loss_smooth_src = loss_dict["loss_smooth_src"]
        loss_smooth_tgt = loss_dict["loss_smooth_tgt"]

        self.logger.info(
                "epoch [%.3d] step [%d/%d] global_step = %d total_loss = %.4f encoder_lr = %.7f\n"
                "        src: rgb = %.4f\n"
                "        src: ssim = %.4f\n"
                "        src: smooth = %.4f\n"
                "        tgt: rgb = %.4f\n"
                "        tgt: ssim = %.4f\n"
                "        tgt: smooth = %.4f\n" %
                (epoch, step, dataset_length, self.global_step,
                 loss.item(), self.optimizer.param_groups[0]["lr"],
                 loss_rgb_src.item(),
                 loss_ssim_src.item(),
                 loss_smooth_src.item(),
                 loss_rgb_tgt.item(),
                 loss_ssim_tgt.item(),
                 loss_smooth_tgt.item(),
                 ))

        for key, value in self.train_losses.items():
            self.tb_writer.add_scalar(key + "/train", loss_dict[key].item(), global_step)
            value.update(loss_dict[key].item())

    def train_epoch(self, train_data_loader, val_data_loader, epoch):
        if hasattr(train_data_loader, "sampler"):
            train_data_loader.sampler.set_epoch(epoch)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.backbone.train()
        self.decoder.train()

        self.current_epoch = epoch
        self.config["current_epoch"] = epoch

         
        for train_loss_item in self.train_losses.values():
            train_loss_item.reset()

         
        for step, (src_items, tgt_items, depth, img_raw, depth_raw) in enumerate(train_data_loader):
         
            step += 1

            self.global_step += 1
            self.set_data(src_items, tgt_items)

            img = src_items['img'].cuda()
            self.img = img
            self.depth = depth.cuda()

            loss_dict, visualization_dict = self.loss_fcn(is_val=False)
            loss = loss_dict["loss"]
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            if step > 0 and step % 500 == 0 and self.config["global_rank"] == 0:
                self.log_training(self.current_epoch,
                                  step,
                                  self.global_step,
                                  len(train_data_loader),
                                  loss_dict)

            if step > 0 and step % 5000 == 0 and self.config["global_rank"] == 0:
                 
                checkpoint_path = os.path.join(self.config["local_workspace"],
                                               "checkpoint_latest.pth")
                torch.save({"backbone": self.backbone.state_dict(),
                            "decoder": self.decoder.state_dict(),
                            "optimizer": self.optimizer.state_dict()}, checkpoint_path)
                self.logger.info("Latest checkpoint saved at {}".format(checkpoint_path))

                if "hdfs_workspace" in self.config:
                    run_shell_cmd(["hdfs", "dfs", "-put", "-f", checkpoint_path,
                                   self.config["hdfs_workspace"]], self.logger)
                    run_shell_cmd(["hdfs", "dfs", "-put", "-f", self.config["log_file"],
                                   self.config["hdfs_workspace"]], self.logger)

            if self.config["global_rank"] == 0 \
                    and self.global_step > 0 \
                    and (self.global_step == 2000 or (self.global_step % self.config["training.eval_interval"] == 0)):
                self.run_eval(val_data_loader)

                 
                checkpoint_path = os.path.join(self.config["local_workspace"],
                                               "checkpoint_%012d.pth" % self.global_step)
                tb_event_path = sorted(glob.glob(os.path.join(self.config["local_workspace"],
                                                              "events.out.tfevents.*")))[-1]
                torch.save({"backbone": self.backbone.state_dict(),
                            "decoder": self.decoder.state_dict()},
                           checkpoint_path)
                if "hdfs_workspace" in self.config:
                    run_shell_cmd(["hdfs", "dfs", "-put", "-f", checkpoint_path,
                                   self.config["hdfs_workspace"]], self.logger)
                    run_shell_cmd(["hdfs", "dfs", "-put", "-f", self.config["log_file"],
                                   self.config["hdfs_workspace"]], self.logger)
                    run_shell_cmd(["hdfs", "dfs", "-put", "-f", tb_event_path,
                                   self.config["hdfs_workspace"]], self.logger)

    def train(self, train_data_loader, val_data_loader):
        for epoch in range(1, self.config["training.epochs"] + 1):
            self.current_epoch = epoch
            self.train_epoch(train_data_loader, val_data_loader, epoch)

            self.lr_scheduler.step()
            if self.config["global_rank"] == 0:
                self.logger.info("Epoch finished, average losses: ")
                for v in self.train_losses.values():
                    self.logger.info("    {}".format(v))

