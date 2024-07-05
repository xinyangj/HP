from typing import Dict, Tuple
import argparse
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch.backends import cudnn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data.dataloader import DataLoader
import wandb
import os
from tqdm import tqdm
from utils.common_utils import (save_checkpoint, parse, dprint, time_log, compute_param_norm,
                                freeze_bn, zero_grad_bn, RunningAverage, Timer)
from utils.dist_utils import all_reduce_dict
from utils.wandb_utils import set_wandb
from utils.seg_utils import UnsupervisedMetrics, batched_crf, get_metrics
from build import (build_model, build_criterion, build_dataset, build_dataloader, build_optimizer)
from pytorch_lightning.utilities.seed import seed_everything
from torchvision import datasets, transforms
import numpy as np
from torch.optim import Adam, AdamW
from loss import SupConLoss
from visualize import visualization
from collections import defaultdict 
from torcheval.metrics import BinaryAUROC
import numpy as np
from dataset.data import create_cityscapes_colormap, create_pascal_label_colormap

def run(opt: dict, is_test: bool = False, is_debug: bool = False):
    is_train = (not is_test)
    seed_everything(seed=0)
    scaler = torch.cuda.amp.GradScaler(init_scale=2048, growth_interval=1000, enabled=True)

    # -------------------- Folder Setup (Task-Specific) --------------------------#
    prefix = "{}/{}_{}".format(opt["output_dir"], opt["dataset"]["data_type"], opt["wandb"]["name"])
    opt["full_name"] = prefix

    # -------------------- Distributed Setup --------------------------#
    if (opt["num_gpus"] == 0) or (not torch.cuda.is_available()):
        raise ValueError("Run requires at least 1 GPU.")

    if (opt["num_gpus"] > 1) and (not dist.is_initialized()):
        assert dist.is_available()
        dist.init_process_group(backend="nccl")  # nccl for NVIDIA GPUs
        world_size = int(dist.get_world_size())
        local_rank = int(dist.get_rank())
        torch.cuda.set_device(local_rank)
        print_fn = partial(dprint, local_rank=local_rank)  # only prints when local_rank == 0
        is_distributed = True
    else:
        world_size = 1
        local_rank = 0
        print_fn = print
        is_distributed = False

    cudnn.benchmark = True

    is_master = (local_rank == 0)
    wandb_save_dir = set_wandb(opt, local_rank, force_mode="disabled" if (is_debug or is_test) else None)

    if not wandb_save_dir:
        wandb_save_dir = os.path.join(opt["output_dir"], opt["wandb"]["name"])
    if is_test:
        wandb_save_dir = "/".join(opt["checkpoint"].split("/")[:-1])

    train_dataset = build_dataset(opt["dataset"], mode="train", model_type=opt["model"]["pretrained"]["model_type"])
    train_loader_memory = build_dataloader(train_dataset, opt["dataloader"], shuffle=True)

    # ------------------------ DataLoader ------------------------------#
    if is_train:
        train_dataset = build_dataset(opt["dataset"], mode="train", model_type=opt["model"]["pretrained"]["model_type"])
        train_loader = build_dataloader(train_dataset, opt["dataloader"], shuffle=True)
    else:
        train_loader = None

    val_dataset = build_dataset(opt["dataset"], mode="val", model_type=opt["model"]["pretrained"]["model_type"])
    val_loader = build_dataloader(val_dataset, opt["dataloader"], shuffle=False,
                                  batch_size=world_size*32)

    # -------------------------- Define -------------------------------#
    net_model, linear_model, cluster_model, cam_model, supcluster_model, supcluster_cam_model = build_model(opt=opt["model"],
                                                         n_classes=val_dataset.n_classes,
                                                         is_direct=opt["eval"]["is_direct"])

    device = torch.device("cuda", local_rank)
    net_model = net_model.to(device)
    linear_model = linear_model.to(device)
    cluster_model = cluster_model.to(device)
    cam_model = cam_model.to(device)
    supcluster_model = supcluster_model.to(device)
    supcluster_cam_model = supcluster_cam_model.to(device)


    model = net_model
    model_m = model

    print_fn("Model:")
    print_fn(model_m)

    label_cmap = create_pascal_label_colormap()

    '''
    for i, v in enumerate(label_cmap):
        print(v)
        if v[0] == 128 and v[1] == 128 and v[2] == 0:
            print(i)
            raise SystemExit
    '''
        

    # --------------------------- Evaluate with Best --------------------------------#
    loading_dir = os.path.join(opt['output_dir'], opt['checkpoint'])
    checkpoint_loaded = torch.load(f"{loading_dir}/ckpt.pth", map_location=device)
    net_model.load_state_dict(checkpoint_loaded['net_model_state_dict'], strict=True)
    linear_model.load_state_dict(checkpoint_loaded['linear_model_state_dict'], strict=True)
    cluster_model.load_state_dict(checkpoint_loaded['cluster_model_state_dict'], strict=True)
    cam_model.load_state_dict(checkpoint_loaded['cam_model_state_dict'], strict=True)
    supcluster_model.load_state_dict(checkpoint_loaded['supcluster_model_state_dict'], strict=True)
    supcluster_cam_model.load_state_dict(checkpoint_loaded['supcluster_cam_model_state_dict'], strict=True)

    loss_, metrics_ = evaluate(net_model, linear_model, cluster_model, cam_model, 
                               supcluster_model, supcluster_cam_model, val_loader, device=device,
                               opt=opt, n_classes=train_dataset.n_classes)
    s = time_log()
    s += f" ------------------- before crf ---------------------\n"
    for metric_k, metric_v in metrics_.items():
        s += f"before crf{metric_k} : {metric_v:.2f}\n"
    print_fn(s)



    loss_, metrics_ = evaluate(net_model, linear_model, cluster_model, cam_model, 
        val_loader, device=device, opt=opt, n_classes=train_dataset.n_classes, is_crf=opt["eval"]["is_crf"])

    s = time_log()
    s += f" -------------------after crf ---------------------\n"
    for metric_k, metric_v in metrics_.items():
        s += f"[after crf] {metric_k} : {metric_v:.2f}\n"
    print_fn(s)


def evaluate(net_model: nn.Module,
             linear_model: nn.Module,
             cluster_model: nn.Module,
             cam_model: nn.Module, 
             supcluster_model: nn.Module, 
             supcluster_cam_model: nn.Module, 
             eval_loader: DataLoader,
             device: torch.device,
             opt: Dict,
             n_classes: int,
             is_crf: bool = False,
             data_type: str = "",
             ) -> Tuple[float, Dict[str, float]]:

    net_model.eval()
    linear_model.eval()
    cluster_model.eval()
    cam_model.eval()
    supcluster_model.eval()
    supcluster_cam_model.eval()

    cluster_metrics = UnsupervisedMetrics(
        "Cluster_", n_classes, opt["eval"]["extra_clusters"], True)
    supcluster_metrics = UnsupervisedMetrics(
        "SupCluster_", n_classes, opt["eval"]["extra_supclusters"], True)
    linear_metrics = UnsupervisedMetrics(
        "Linear_", n_classes, 0, False)
    cam_metrics = BinaryAUROC()
    supcluster_cam_metrics = BinaryAUROC()

    vis_data = {}
    vis_count = 0
    vis_num = 100

    normalize_mean = np.array([0.485, 0.456, 0.406])
    normalize_std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        eval_stats = RunningAverage()

        for i, data in enumerate(tqdm(eval_loader)):
            img: torch.Tensor = data['img'].to(device, non_blocking=True)
            label: torch.Tensor = data['label'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                output = net_model(img)
            feats = output[0]
            head_code = output[1]

            with torch.cuda.amp.autocast(enabled=True):
                cluster_loss, cluster_preds, clusters = cluster_model(head_code, 2)
                heat_map, logits, supcluster_heat_map, supcluster_logits, att, _= cam_model(head_code, clusters)#cluster_model.clusters)
                #supcluster_output = supcluster_model(head_code, cluster_model.clusters, None, is_direct=False)
                #supcluster_heat_map, supcluster_logits  = supcluster_cam_model(head_code, supcluster_model.clusters)
                #heat_map = F.sigmoid(heat_map)
            #print(head_code[0, 0], head_code[0, 0].size())
            #head_code = F.interpolate(head_code, label.shape[-2:], mode='bicubic', align_corners=False)
            
            heat_map = F.interpolate(heat_map, label.shape[-2:], mode='bicubic', align_corners=False)
            supcluster_heat_map = F.interpolate(supcluster_heat_map, label.shape[-2:], mode='bicubic', align_corners=False)
            
            head_code = F.interpolate(head_code, label.shape[-2:], mode='bilinear', align_corners=False)
            if is_crf:
                with torch.cuda.amp.autocast(enabled=True):
                    linear_preds = torch.log_softmax(linear_model(head_code), dim=1)

                with torch.cuda.amp.autocast(enabled=True):
                    cluster_loss, cluster_preds = cluster_model(head_code, 2, log_probs=True, is_direct=opt["eval"]["is_direct"])
                linear_preds = batched_crf(img, linear_preds).argmax(1).cuda()
                cluster_preds = batched_crf(img, cluster_preds).argmax(1).cuda()

            else:
                with torch.cuda.amp.autocast(enabled=True):
                    linear_preds = linear_model(head_code)
                    linear_preds = F.interpolate(linear_preds, label.shape[-2:], mode='bicubic', align_corners=False)
                    linear_preds = linear_preds.argmax(1)

                with torch.cuda.amp.autocast(enabled=True):
                    #print(head_code[0, 0], head_code[0, 0].size())
                    #raise SystemExit
                    #cluster_loss, cluster_preds = cluster_model(head_code, None, is_direct=opt["eval"]["is_direct"])
                    cluster_loss, cluster_preds, _ = cluster_model(head_code, None, is_direct=opt["eval"]["is_direct"])
                    #supcluster_loss, supcluster_preds = supcluster_model(head_code, cluster_model.clusters, None, is_direct=False)
                
                cluster_preds = F.interpolate(cluster_preds, label.shape[-2:], mode='bicubic', align_corners=False)
                #supcluster_preds = F.interpolate(supcluster_preds, label.shape[-2:], mode='bicubic', align_corners=False)

                cluster_conf = cluster_preds
                #supcluster_conf = supcluster_preds
                
                cluster_preds = cluster_preds.argmax(1)
                #supcluster_preds = supcluster_preds.argmax(1)


            binary_label = torch.zeros_like(label)
            binary_label[label != 11] = 0
            binary_label[label == 11] = 1
            binary_label = torch.max(torch.max(binary_label, -1)[0], -1)[0]
            linear_metrics.update(linear_preds, label)
            cluster_metrics.update(cluster_preds, label)
            #supcluster_metrics.update(supcluster_preds, label)
            cam_metrics.update(F.sigmoid(logits), binary_label)
            supcluster_cam_metrics.update(F.sigmoid(supcluster_logits), binary_label.long())

            eval_stats.append(cluster_loss)

            
            
            
            if vis_count < vis_num:
                print(binary_label)
                #binary_label = 1 - binary_label
                binary_label = binary_label.cpu()
                vis_count += np.sum(binary_label.numpy().astype(np.int32))

                vis_im = data['img'][binary_label == 1]
                vis_im = (vis_im * normalize_std[None, :, None, None] + normalize_mean[None, :, None, None]) * 255
                vis_im = torch.clip(vis_im, 0, 255).to(torch.uint8)
                vis_im = torch.permute(vis_im, (0, 2, 3, 1))
                #heat_map = heat_map[:, 1]
                heat_map = heat_map.squeeze()
                supcluster_heat_map = supcluster_heat_map.squeeze()
                if 'img_path' not in vis_data:
                    vis_data['img'] = vis_im
                    vis_data['img_path'] = data['ind'][binary_label == 1]
                    vis_data['label'] = label.cpu()[binary_label == 1]
                    vis_data['linear_preds'] = linear_preds.cpu()[binary_label == 1]
                    vis_data['cluster_preds'] = cluster_preds.cpu()[binary_label == 1]
                    vis_data['cluster_conf'] = cluster_conf.cpu()[binary_label == 1]
                    #vis_data['supcluster_preds'] = supcluster_preds.cpu()[binary_label == 1]
                    #vis_data['supcluster_conf'] = supcluster_conf.cpu()[binary_label == 1]
                    vis_data['cam_preds'] = heat_map.cpu()[binary_label == 1]
                    vis_data['supcluster_cam_preds'] = supcluster_heat_map.cpu()[binary_label == 1]
                    vis_data['att'] = att.cpu()[binary_label == 1]

                else:
                    vis_data['img'] = torch.cat([vis_data['img'], vis_im], dim = 0)
                    vis_data['img_path'] = torch.cat([vis_data['img_path'], data['ind'][binary_label == 1]], dim = 0)
                    vis_data['label'] = torch.cat([vis_data['label'], label.cpu()[binary_label == 1]], dim = 0)
                    vis_data['linear_preds'] = torch.cat([vis_data['linear_preds'], linear_preds.cpu()[binary_label == 1]], dim = 0)
                    vis_data['cluster_preds'] = torch.cat([vis_data['cluster_preds'], cluster_preds.cpu()[binary_label == 1]], dim = 0)
                    vis_data['cluster_conf'] = torch.cat([vis_data['cluster_conf'], cluster_conf.cpu()[binary_label == 1]], dim = 0)
                    #vis_data['supcluster_preds'] = torch.cat([vis_data['supcluster_preds'], supcluster_preds.cpu()[binary_label == 1]], dim = 0)
                    #vis_data['supcluster_conf'] = torch.cat([vis_data['supcluster_conf'], supcluster_conf.cpu()[binary_label == 1]], dim = 0)
                    vis_data['cam_preds'] = torch.cat([vis_data['cam_preds'], heat_map.cpu()[binary_label == 1]], dim = 0)
                    vis_data['supcluster_cam_preds'] = torch.cat([vis_data['supcluster_cam_preds'], supcluster_heat_map.cpu()[binary_label == 1]], dim = 0)
                    vis_data['att'] = torch.cat([vis_data['att'], att.cpu()[binary_label == 1]], dim = 0)
            print(vis_data['img'].size())
            #print(data.keys())
            save_dir='vis_tune_code_graph_att_cls15'
            #break

        eval_metrics = get_metrics(cluster_metrics, linear_metrics,cam_metrics, m4 = None, m5 = supcluster_cam_metrics)
        visualization(save_dir = save_dir,dataset_type = 'voc_stuff',saved_data = vis_data,cluster_metrics = cluster_metrics,is_label = True)
        return eval_stats.avg, eval_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, required=True, help="Path to option JSON file.")
    parser.add_argument("--test", action="store_true", help="Test mode, no WandB, highest priority.")
    parser.add_argument("--debug", action="store_true", help="Debug mode, no WandB, second highest priority.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint override")
    parser.add_argument("--data_path", type=str, default=None, help="Data path override")

    parser_args = parser.parse_args()
    parser_opt = parse(parser_args.opt)
    # if parser_args.checkpoint is not None:
    #     parser_opt["checkpoint"] = parser_args.checkpoint
    if parser_args.data_path is not None:
        parser_opt["dataset"]["data_path"] = parser_args.data_path

    run(parser_opt, is_test=parser_args.test, is_debug=parser_args.debug)


if __name__ == "__main__":
    main()
