import torch.nn.functional as F
import numpy as np
import os
from collections import defaultdict
from os.path import join
from dataset.data import create_cityscapes_colormap, create_pascal_label_colormap
from utils.seg_utils import unnorm
from PIL import Image
from utils.seg_utils import UnsupervisedMetrics
import matplotlib.pyplot as plt 


def prep_for_plot(img, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")
    else:
        img = img.unsqueeze(0)
    plot_img = unnorm(img).squeeze(0).squeeze(0).cpu().permute(1, 2, 0)
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())
    return plot_img


def visualization(save_dir: str, dataset_type: str, saved_data: defaultdict, cluster_metrics: UnsupervisedMetrics,
                  is_label: bool = False):
    if is_label:
        os.makedirs(join(save_dir, "label"), exist_ok=True)
    os.makedirs(join(save_dir, "cluster"), exist_ok=True)
    os.makedirs(join(save_dir, "raw_cluster"), exist_ok=True)
    os.makedirs(join(save_dir, "raw_supcluster"), exist_ok=True)
    os.makedirs(join(save_dir, "linear"), exist_ok=True)
    os.makedirs(join(save_dir, "rgb"), exist_ok=True)
    os.makedirs(join(save_dir, "cam"), exist_ok=True)

    if dataset_type.startswith("cityscapes"):
        label_cmap = create_cityscapes_colormap()
    else:
        label_cmap = create_pascal_label_colormap()

    for index in range(len(saved_data["img_path"])):
        file_name = str(saved_data["img_path"][index]).split("/")[-1].split(".")[0]
        if is_label:
            plot_label = (label_cmap[saved_data["label"][index]]).astype(np.uint8)
            Image.fromarray(plot_label).save(join(join(save_dir, "label", file_name + ".png")))

        plot_cluster = (label_cmap[cluster_metrics.map_clusters(saved_data["cluster_preds"][index])]).astype(np.uint8)
        Image.fromarray(plot_cluster).save(join(join(save_dir, "cluster", file_name + ".png")))

        plot_linear = (label_cmap[saved_data["linear_preds"][index]]).astype(np.uint8)
        Image.fromarray(plot_linear).save(join(join(save_dir, "linear", file_name + ".png")))
        '''
        print(label_cmap.shape)
        print(saved_data["linear_preds"][index].shape)
        print(np.unique(saved_data["cluster_preds"][index]))
        '''
        plot_rawcluster = (label_cmap[saved_data["cluster_preds"][index]]).astype(np.uint8)
        '''
        for row in plot_rawcluster:
            for c in row:
                new_one = True
                for o in out:
                    if c[0] == o[0] and c[1] == o[1] and c[2] == o[2]:
                        #print(c, end = " ")
                        new_one = False
                if new_one: out.append(list(c))
                
            #print()
        print(out)
        '''
        Image.fromarray(plot_rawcluster).save(join(join(save_dir, "raw_cluster", file_name + ".png")))
        
        plot_rawsupcluster = (label_cmap[saved_data["supcluster_preds"][index]]).astype(np.uint8)
        Image.fromarray(plot_rawsupcluster).save(join(join(save_dir, "raw_supcluster", file_name + ".png")))
        
        img = saved_data['img'][index].cpu().numpy()
        Image.fromarray(img).save(join(join(save_dir, "rgb", file_name + ".png")))

        plot_cam = saved_data['cam_preds'][index].cpu().numpy()
        plt.imshow(plot_cam, cmap = 'jet')
        #cbar = plt.colorbar(ticks=[0, 1]) 
 
        plt.savefig(join(join(save_dir, "cam", file_name + ".png")))

def visualization_label(save_dir: str, saved_data: defaultdict):
    label_cmap = create_pascal_label_colormap()

    for index in range(saved_data["label"][0].size(0)):
        os.makedirs(join('./visualize/attn', str(index)), exist_ok=True)

        plot_label = (label_cmap[saved_data["label"][0][index]]).astype(np.uint8)
        imagename = str(index)+"_classlabel.png"
        Image.fromarray(plot_label).save(join(save_dir, str(index), imagename))
