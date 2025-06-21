import torch
import numpy as np
import h5py
from xml.dom.minidom import parse
import xml.dom.minidom
from openTSNE import TSNE
from shapely.geometry import Polygon
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

def read_annotation(anno_file, return_type=False):
    anno_tumor = []
    anno_normal = []
    anno_type = set()
    DOMTree = xml.dom.minidom.parse(anno_file)
    annotations = DOMTree.documentElement.getElementsByTagName('Annotations')[0].getElementsByTagName('Annotation')
    for i in range(len(annotations)):
        anno_type.add(annotations[i].getAttribute('PartOfGroup'))
        if annotations[i].getAttribute('PartOfGroup') == 'Exclusion':
            coordinates = annotations[i].getElementsByTagName('Coordinates')
            _tmp = []
            for node in coordinates[0].childNodes:
                if type(node) == xml.dom.minidom.Element:
                    _tmp.append([int(float(node.getAttribute("X"))), int(float(node.getAttribute("Y")))])

            anno_normal.append(_tmp)
        elif annotations[i].getAttribute('PartOfGroup') != 'None':
            coordinates = annotations[i].getElementsByTagName('Coordinates')
            _tmp = []
            for node in coordinates[0].childNodes:
                if type(node) == xml.dom.minidom.Element:
                    _tmp.append([int(float(node.getAttribute("X"))), int(float(node.getAttribute("Y")))])

            anno_tumor.append(_tmp)
    if return_type:
        return anno_tumor, anno_normal, anno_type
    else:
        return anno_tumor, anno_normal

def get_label(coords, anno_file, _l=None):
    if anno_file is None:
        return None

    annos_tumor, annos_normal = read_annotation(anno_file)
    annos_tumor_polygon = [Polygon(_anno) for _anno in annos_tumor]
    annos_normal_polygon = [Polygon(_anno) for _anno in annos_normal]

    annos_tumor_in_normal_idx = [
        idx for idx, _anno in enumerate(annos_tumor_polygon)
        if any(_anno.covered_by(_anno_1) for _anno_1 in annos_normal_polygon)
    ]

    label = np.zeros(len(coords), dtype=int)

    tumor_polygons = {poly: True for idx, poly in enumerate(annos_tumor_polygon) if idx not in annos_tumor_in_normal_idx}
    exclusion_polygons = {poly: True for idx, poly in enumerate(annos_tumor_polygon) if idx in annos_tumor_in_normal_idx}

    for i, coord in enumerate(coords):
        patch = Polygon([
            [coord[0], coord[1]],
            [coord[0] + 512, coord[1]],
            [coord[0] + 512, coord[1] + 512],
            [coord[0], coord[1] + 512]
        ])

        intersects_tumor = any(patch.intersects(poly) for poly in tumor_polygons)
        intersects_exclusion = any(patch.intersects(poly) for poly in exclusion_polygons)

        if intersects_tumor and not intersects_exclusion:
            if _l is not None:
                label[i] = 0
            else:
                label[i] = 1
        else:
            label[i] = 0

    if _l is not None:
        label[_l > 0] = 1

    return label

def plot(x, y, ax=None, title=None, draw_legend=True, draw_centers=False, draw_cluster_labels=False, colors=None, legend_kwargs=None, label_order=None, **kwargs):
    import matplotlib

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.8)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    size = deepcopy(y)
    point_size = deepcopy(y)
    point_size[size != 1] = 1
    point_size[size == 1] = 50
    point_size[size == 2] = 1

    fig = ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, s=point_size, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label + ': ' + str(len(x)),
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            mlines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=str(yi) + ': ' + str(len(y[y == yi])),
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    return fig

def tsne(feat, coords=None, anno_file=None, _l=None, **kwargs):
    try:
        label = get_label(coords, anno_file, _l)
    except Exception as e:
        print(f"Error in get_label: {e}")
        label = None

    feat = feat.cpu().numpy()
    try:
        embedding = TSNE(n_jobs=1).fit(feat)
    except Exception as e:
        print(f"Error in t-SNE: {e}")
        return None

    y = label if label is not None else np.array([1 for _ in range(len(embedding))])
    return plot(embedding, y, **kwargs)

directory_1 = r'D:\DATA\PLIP_MIL\draw_pictures\plip_feature'
for root, dirs, files in os.walk(directory_1):
    for file in files:
        if file.endswith('.pt'):
            file_path = os.path.join(root, file)
            name = file.split('.')[-2].split('_')[0] + '_' + file.split('.')[-2].split('_')[1]
            if name == 'tumor_099' or name == 'tumor_108':
                plip_feat = torch.load(file_path)
                patch = h5py.File(rf"D:\DATA\PLIP_MIL\draw_pictures\h5\{name}.h5", "r")
                coords = patch['coords'][:]
                plip_tsne = tsne(plip_feat, coords, rf"D:\DATA\PLIP_MIL\draw_pictures\lesion_annotations\{name}.xml", draw_legend=False)
                if plip_tsne is not None:
                    plip_tsne.get_figure().savefig(rf'D:\DATA\PLIP_MIL\draw_pictures\plip_png\{name}_plip .svg', dpi=600, bbox_inches='tight')
                    print(f"{name} saved successfully.")