import torch
import h5py
from matplotlib.colors import LogNorm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import openslide
import xml
from shapely.geometry import Polygon
import openslide
import h5py
import torch
import json
import cv2
import random
import seaborn as sns
import os
import xml.dom.minidom

def read_annotation(anno_file,return_type=False):
    anno_tumor = []
    anno_normal = []
    anno_type = set()
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(anno_file)
    annotations = DOMTree.documentElement.getElementsByTagName('Annotations')[0].getElementsByTagName('Annotation')
    for i in range(len(annotations)):
        anno_type.add(annotations[i].getAttribute('PartOfGroup'))
        if annotations[i].getAttribute('PartOfGroup') == 'Exclusion':
            coordinates = annotations[i].getElementsByTagName('Coordinates')
            _tmp = []
            for node in coordinates[0].childNodes:
                if type(node) == xml.dom.minidom.Element:
                    _tmp.append([int(float(node.getAttribute("X"))),int(float(node.getAttribute("Y")))])

            anno_normal.append(_tmp)
        elif annotations[i].getAttribute('PartOfGroup') != 'None':
            coordinates = annotations[i].getElementsByTagName('Coordinates')
            _tmp = []
            for node in coordinates[0].childNodes:
                if type(node) == xml.dom.minidom.Element:
                    _tmp.append([int(float(node.getAttribute("X"))),int(float(node.getAttribute("Y")))])

            anno_tumor.append(_tmp)
    if return_type:
        return anno_tumor,anno_normal,anno_type
    else:
        return anno_tumor,anno_normal


def get_label(coords, anno_file, _l=None):
    if anno_file is None:
        return None
    label = []
    annos_tumor, annos_normal = read_annotation(anno_file)
    annos_tumor_polygon = [Polygon(_anno) for _anno in annos_tumor]
    annos_normal_polygon = [Polygon(_anno) for _anno in annos_normal]
    annos_tumor_in_normal_idx = []
    for idx, _anno in enumerate(annos_tumor_polygon):
        for _anno_1 in annos_normal_polygon:
            if _anno.covered_by(_anno_1):
                annos_tumor_in_normal_idx.append(idx)

    for coord in coords:
        _patch = Polygon(
            [coord, [coord[0] + 512, coord[1]], [coord[0] + 512, coord[1] + 512], [coord[0], coord[1] + 512]])
        _flag = 0
        _flag_always = 0
        for idx, _anno in enumerate(annos_tumor_polygon):
            if _patch.intersects(_anno):
                _flag = 1
                if idx in annos_tumor_in_normal_idx:
                    _flag_always = 1
        if not _flag_always:
            for _anno_1 in annos_normal_polygon:
                if _patch.intersects(_anno_1):
                    _flag = 0

        if _flag:
            # label.append(1)
            if _l is not None:
                label.append(0)
            else:
                label.append(1)
        else:
            label.append(0)
    label = np.array(label)

    if _l is not None:
        # label[np.array(label == 0) * np.array(_l > 0)] = 1
        label[np.array(_l > 0)] = 1

    return label

def get_area(pos_anchors,margin_percentage = 2.0 ,center_anchors=None,width_height=None):
    if center_anchors is not None:
        margin = 10 * 512 * margin_percentage / 2
        center_anchor = (center_anchors[1],center_anchors[0])
    else:
        top, down, left, right = min(pos_anchors[:, 1]), max(pos_anchors[:, 1]), min(pos_anchors[:, 0]), max(pos_anchors[:, 0])
        center_anchor = ((top + down) // 2, (left + right) // 2)
        margin = max((down - top), (right - left)) * margin_percentage / 2

    top, down, left, right = np.array(
        [center_anchor[0]-margin, center_anchor[0]+margin, center_anchor[1]-margin, center_anchor[1]+margin],
        dtype=int
    )
    ori_coord = np.array([down,top,right,left])
    top,down,left,right = np.clip(top,0,width_height[1]),np.clip(down,0,width_height[1]),np.clip(left,0,width_height[0]),np.clip(right,0,width_height[0])
    _gap = np.array([down,top,right,left])
    _coord = np.array([top,down,left,right])
    top,down,left,right = _coord + (_gap - ori_coord)

    return top,down,left,right,(down-top) * (right-left)

def screen_coords(scores, coords, top_left, bot_right,cam=None):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    if cam is not None:
        return scores[mask], coords[mask], cam[mask]
    else:
        return scores[mask], coords[mask]

def random_area(width,height,min_area,max_area):
    while True:
        left = random.randint(0, width - 1)
        top = random.randint(0, height - 1)
        right = random.randint(left + 1, width)
        down = random.randint(top + 1, height)
        area = (right - left) * (down - top)
        if min_area <= area <= max_area:
            return (top, left, down, right)

str_lst = []
for i in range(1,112):
    if len(str(i)) < 2:
        str_lst.append('00' + str(i))
    elif len(str(i)) < 3:
        str_lst.append('0' + str(i))
    else:
        str_lst.append(str(i))
for str_id in str_lst:
    f_id = f'tumor_{str_id}'
    data_1 = torch.load(fr"D:\DATA\PLIP_MIL\draw_pictures\weights\adapter_attention\tumor{str_id}_attention.pt")
    h5 = h5py.File(fr"D:\DATA\PLIP_MIL\draw_pictures\h5\tumor_{str_id}.h5", "r")
    loc_1 = h5['coords']
    data_1 = data_1.squeeze().cpu().numpy()
    # data = data_1.flatten()
    tif_path = "E:/CAMELYON16/Raw_Data/" + f_id + ".tif"
    xml_path = "D:/DATA/PLIP_MIL/draw_pictures/lesion_annotations/" + f_id + ".xml"
    slide = openslide.OpenSlide(tif_path)
    width, height = slide.dimensions

    anno_tumor, anno_normal = read_annotation(xml_path)
    patch_num = 1
    patch_size = 512
    roi = 0
    roi_num = 0
    for _pos in range(len(anno_tumor)):
        _, _, _, _, area = get_area(np.array(anno_tumor[_pos]), width_height=[width, height])
        if area >= (patch_size * patch_num) ** 2:
            print(_pos, area ** 0.5 / patch_size)
            if area ** 0.5 / patch_size >= roi_num:
                roi = _pos
                roi_num = area ** 0.5 / patch_size
    crop_size = 512
    stride = 512
    vis_level = 3
    vis_size = None  # 448
    alpha = 0.3  # 0-no_img,1-all_img
    margin_percentage = 1.5
    roi_coord = None
    rel_norm = False
    filter_thr = 0.4
    filter_thr_cam = 0.5001
    is_norm = True
    save_figure = False
    _c_cam = np.array([0, 255, 255])
    _c_attn = np.array([255, 255, 255])
    pos_anchors = np.array(anno_tumor[roi])
    top, down, left, right, _ = get_area(pos_anchors, margin_percentage, width_height=[width, height])
    crop_coords = []
    for i in range(top, down, stride):
        for j in range(left, right, stride):
            if j + crop_size > width or i + crop_size > height:
                continue
            crop_coords.append((j, i))
    right, down = np.max(crop_coords, 0) + crop_size
    print(">>>>>>>>>>>> region coord gotten >>>>>>>>>>>")
    scale_level_ratio = np.array(slide.level_dimensions[vis_level]) / np.array(slide.level_dimensions[0])
    _w, _h = np.array((right - left, down - top)) * scale_level_ratio
    # _w,_h = (right-left, down-top)
    region = slide.read_region((left, top), list(range(slide.level_count))[vis_level], (int(_w), int(_h))).convert(
        'RGB')
    print(">>>>>>>>>>>> slide region gotten >>>>>>>>>>>")
    A_roi, coords_roi = screen_coords(data_1, loc_1, (left, top), (right, down))
    img = np.array(region)
    coords_roi_rel = []
    for i in range(len(coords_roi)):
        rel_x = coords_roi[i][0] - left
        rel_y = coords_roi[i][1] - top
        rel_x, rel_y = np.clip(rel_x, 0, right - left), np.clip(rel_y, 0, down - top)
        coords_roi_rel.append([rel_x, rel_y])
    print(">>>>>>>>>>>> relative coords computed >>>>>>>>>>>")
    #
    if is_norm:
        if rel_norm:
            A_roi = (A_roi - A_roi.min()) / (A_roi.max() - A_roi.min())
        else:
            A_roi = (A_roi - data_1.min()) / (data_1.max() - data_1.min())
    A_roi[A_roi < filter_thr] = 0
    heatmap_attn = np.zeros(img.shape)
    # heatmap_cam = np.zeros(img.shape)
    alpha_mat_attn = np.ones(img.shape) * alpha
    # alpha_mat_cam = np.ones(img.shape) * 0.5 * alpha
    for _idx, _coord in enumerate(coords_roi_rel):
        heatmap_attn[int(_coord[1] * scale_level_ratio[1]):int((_coord[1] + patch_size) * scale_level_ratio[1]),
        int(_coord[0] * scale_level_ratio[0]):int((_coord[0] + patch_size) * scale_level_ratio[0]), :] = A_roi[
                                                                                                             _idx] * 255

    print(">>>>>>>>>>>> heatmap computed >>>>>>>>>>>")
    # A_flag = False
    # cam_flag = False
    #
    #     if A_roi[_idx] != 0:
    #         heatmap_attn[int(_coord[1]*scale_level_ratio[1]):int((_coord[1]+patch_size)*scale_level_ratio[1]),int(_coord[0]*scale_level_ratio[0]):int((_coord[0]+patch_size)*scale_level_ratio[0]),:] = A_roi[_idx] * _c_attn
    #         A_flag = True
    #     else:
    #         alpha_mat_cam[int(_coord[1] * scale_level_ratio[1]):int((_coord[1] + patch_size) * scale_level_ratio[1]),int(_coord[0] * scale_level_ratio[0]):int((_coord[0] + patch_size) * scale_level_ratio[0]), :] = 0
    # if not A_flag and not cam_flag:
    #         alpha_mat_attn[int(_coord[1]*scale_level_ratio[1]):int((_coord[1]+patch_size)*scale_level_ratio[1]),int(_coord[0]*scale_level_ratio[0]):int((_coord[0]+patch_size)*scale_level_ratio[0]),:] = alpha
    # print(">>>>>>>>>>>> heatmap computed >>>>>>>>>>>")
    # blended_image = (alpha_mat_attn * img + alpha_mat_cam*img + (1-alpha_mat_attn) * heatmap_attn + (1-alpha_mat_cam) * heatmap_cam)
    # blended_image = blended_image.astype(np.uint8)
    blended_image = (alpha * img + (1 - alpha) * heatmap_attn[:, :, :3])
    blended_image = blended_image.astype(np.uint8)

    if vis_size is not None:
        plt.imshow(cv2.resize(blended_image, dsize=(vis_size, vis_size)))
        scale_ratio = (vis_size / img.shape[0]) * scale_level_ratio
    else:
        plt.imshow(blended_image)
        scale_ratio = scale_level_ratio
    print(">>>>>>>>>>>> heatmap draw done>>>>>>>>>>>")
    plt.xlim(0, (right - left) * scale_ratio[0])
    plt.ylim((down - top) * scale_ratio[1], 0)
    plt.axis("off")
    for anchors in (anno_tumor):
        _pos_anchors = np.array(anchors)
        plt.plot((_pos_anchors[:, 0] - left) * scale_ratio[0], (_pos_anchors[:, 1] - top) * scale_ratio[1],
                 color='deepskyblue')
    # if save_figure:
    path = os.path.join('D:/DATA/PLIP_MIL/draw_pictures/heatmap/adapter_attention_map', f_id + '_roi' + str(roi) + '.png')
    plt.savefig(path, dpi=450, bbox_inches='tight')
    plt.close()