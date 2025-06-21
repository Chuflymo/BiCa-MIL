import math
import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InnerAttention(nn.Module):
    def __init__(self, dim, head_dim=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 epeg=True, epeg_k=15, epeg_2d=False, epeg_bias=True, epeg_type='attn'):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.epeg_2d = epeg_2d
        self.epeg_type = epeg_type
        if epeg:
            padding = epeg_k // 2
            if epeg_2d:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, epeg_k, padding=padding, groups=num_heads, bias=epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, epeg_k, padding=padding,
                                        groups=head_dim * num_heads, bias=epeg_bias)
            else:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, (epeg_k, 1), padding=(padding, 0), groups=num_heads,
                                        bias=epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, (epeg_k, 1), padding=(padding, 0),
                                        groups=head_dim * num_heads, bias=epeg_bias)
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_regions*B, N, C)
        """
        B_, N, C = x.shape

        # x = self.pe(x)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.pe is not None and self.epeg_type == 'attn':
            pe = self.pe(attn)
            attn = attn + pe

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.pe is not None and self.epeg_type == 'value_bf':
            # B,H,N,C -> B,HC,N-0.5,N-0.5
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            # pe = torch.einsum('ahbd->abhd',pe).flatten(-2,-1)
            v = v + pe.reshape(B_, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # print(v.size())

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads * self.head_dim)

        if self.pe is not None and self.epeg_type == 'value_af':
            # print(v.size())
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            # print(pe.size())
            # print(v.size())
            x = x + pe.reshape(B_, self.num_heads * self.head_dim, N).transpose(-1, -2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 region with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class RegionAttntion(nn.Module):
    def __init__(self, dim, head_dim=None, num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., region_num=8, epeg=False, min_region_num=0, min_region_ratio=0., region_attn='native',
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        if region_attn == 'native':
            self.attn = InnerAttention(
                dim, head_dim=head_dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, epeg=epeg, **kwargs)
        # elif region_attn == 'ntrans':
        #     self.attn = NystromAttention(
        #         dim=dim,
        #         dim_head=head_dim,
        #         heads=num_heads,
        #         dropout=drop
        #     )

    def padding(self, x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H + _n, W + _n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H + _n, W + _n
            region_size = int(H // self.region_num)
            region_num = self.region_num

        add_length = H * W - L

        # if padding much，i will give up region attention. only for ablation
        if (add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape

        # padding
        x, H, W, add_length, region_num, region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # R-MSA
        attn_regions = self.attn(x_regions)  # nW*B, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]

        return x


class CrossRegionAttntion(nn.Module):
    def __init__(self, dim, head_dim=None, num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., region_num=8, epeg=False, min_region_num=0, min_region_ratio=0., crmsa_k=3,
                 crmsa_mlp=False, region_attn='native', **kwargs):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        self.attn = InnerAttention(
            dim, head_dim=head_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, epeg=epeg, **kwargs)

        self.crmsa_mlp = crmsa_mlp
        if crmsa_mlp:
            self.phi = [nn.Linear(self.dim, self.dim // 4, bias=False)]
            self.phi += [nn.Tanh()]
            self.phi += [nn.Linear(self.dim // 4, crmsa_k, bias=False)]
            self.phi = nn.Sequential(*self.phi)
        else:
            self.phi = nn.Parameter(
                torch.empty(
                    (self.dim, crmsa_k),
                )
            )
            nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def padding(self, x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H + _n, W + _n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H + _n, W + _n
            region_size = int(H // self.region_num)
            region_num = self.region_num

        add_length = H * W - L

        # if padding much，i will give up region attention. only for ablation
        if (add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape

        # padding
        x, H, W, add_length, region_num, region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # CR-MSA
        if self.crmsa_mlp:
            logits = self.phi.transpose(1, 2)  # W*B, sW, region_size*region_size
        else:
            logits = torch.einsum("w p c, c n -> w p n", x_regions, self.phi).transpose(1,
                                                                                        2)  # nW*B, sW, region_size*region_size

        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)

        logits_min, _ = logits.min(dim=-1)
        logits_max, _ = logits.max(dim=-1)
        dispatch_weights_mm = (logits - logits_min.unsqueeze(-1)) / (
                    logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)

        attn_regions = torch.einsum("w p c, w n p -> w n p c", x_regions, combine_weights).sum(dim=-2).transpose(0,
                                                                                                                 1)  # sW, nW, C

        if return_attn:
            attn_regions, _attn = self.attn(attn_regions, return_attn)  # sW, nW, C
            attn_regions = attn_regions.transpose(0, 1)  # nW, sW, C
        else:
            attn_regions = self.attn(attn_regions).transpose(0, 1)  # nW, sW, C

        attn_regions = torch.einsum("w n c, w n p -> w n p c", attn_regions,
                                    dispatch_weights_mm)  # nW, sW, region_size*region_size, C
        attn_regions = torch.einsum("w n p c, w n p -> w n p c", attn_regions, dispatch_weights).sum(
            dim=1)  # nW, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]

        return x


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head=8, drop_out=0.1, drop_path=0., ffn=False, ffn_act='gelu',
                 mlp_ratio=4., trans_dim=64, attn='rmsa', n_region=8, epeg=False, region_size=0, min_region_num=0,
                 min_region_ratio=0, qkv_bias=True, crmsa_k=3, epeg_k=15, **kwargs):
        super().__init__()

        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()
        if attn == 'ntrans':
            # self.attn = NystromAttention(
            #     dim=dim,
            #     dim_head=trans_dim,  # dim // 8
            #     heads=head,
            #     num_landmarks=256,  # number of landmarks dim // 2
            #     pinv_iterations=6,
            #     # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            #     residual=True,
            #     # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            #     dropout=drop_out
            # )
            pass
        elif attn == 'rmsa':
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                **kwargs
            )
        elif attn == 'crmsa':
            self.attn = CrossRegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                **kwargs
            )
        else:
            raise NotImplementedError
        # elif attn == 'rrt1d':
        #     self.attn = RegionAttntion1D(
        #         dim=dim,
        #         num_heads=head,
        #         drop=drop_out,
        #         region_num=n_region,
        #         head_dim=trans_dim,
        #         conv=epeg,
        #         **kwargs
        #     )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop_out) if ffn else nn.Identity()

    def forward(self, x, need_attn=False):

        x, attn = self.forward_trans(x, need_attn=need_attn)

        if need_attn:
            return x, attn
        else:
            return x

    def forward_trans(self, x, need_attn=False):
        attn = None

        if need_attn:
            z, attn = self.attn(self.norm(x), return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))

        x = x + self.drop_path(z)

        # FFN
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn


class RRTEncoder(nn.Module):
    def __init__(self, mlp_dim=512, pos_pos=0, pos='none', peg_k=7, attn='rmsa', region_num=8, drop_out=0.1, n_layers=2,
                 n_heads=8, drop_path=0., ffn=False, ffn_act='gelu', mlp_ratio=4., trans_dim=64, epeg=True, epeg_k=15,
                 region_size=0, min_region_num=0, min_region_ratio=0, qkv_bias=True, peg_bias=True, peg_1d=False,
                 cr_msa=True, crmsa_k=3, all_shortcut=False, crmsa_mlp=False, crmsa_heads=8, need_init=False, **kwargs):
        super(RRTEncoder, self).__init__()

        self.final_dim = mlp_dim

        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = all_shortcut

        self.layers = []
        for i in range(n_layers - 1):
            self.layers += [
                TransLayer(dim=mlp_dim, head=n_heads, drop_out=drop_out, drop_path=drop_path, ffn=ffn, ffn_act=ffn_act,
                           mlp_ratio=mlp_ratio, trans_dim=trans_dim, attn=attn, n_region=region_num, epeg=epeg,
                           region_size=region_size, min_region_num=min_region_num, min_region_ratio=min_region_ratio,
                           qkv_bias=qkv_bias, epeg_k=epeg_k, **kwargs)]
        self.layers = nn.Sequential(*self.layers)

        # CR-MSA
        self.cr_msa = TransLayer(dim=mlp_dim, head=crmsa_heads, drop_out=drop_out, drop_path=drop_path, ffn=ffn,
                                 ffn_act=ffn_act, mlp_ratio=mlp_ratio, trans_dim=trans_dim, attn='crmsa',
                                 qkv_bias=qkv_bias, crmsa_k=crmsa_k, crmsa_mlp=crmsa_mlp,
                                 **kwargs) if cr_msa else nn.Identity()

        # only for ablation
        if pos == 'ppeg':
            #self.pos_embedding = PPEG(dim=mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
            pass
        elif pos == 'sincos':
            #self.pos_embedding = SINCOS(embed_dim=mlp_dim)
            pass
        elif pos == 'peg':
            #self.pos_embedding = PEG(mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
            pass
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

        if need_init:
            self.apply(initialize_weights)

    def forward(self, x):
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
            x = x.transpose(1, 2)
            shape_len = 4

        batch, num_patches, C = x.shape
        x_shortcut = x

        # PEG/PPEG
        if self.pos_pos == -1:
            x = self.pos_embedding(x)

        # R-MSA within region
        for i, layer in enumerate(self.layers.children()):
            if i == 1 and self.pos_pos == 0:
                x = self.pos_embedding(x)
            x = layer(x)

        x = self.cr_msa(x)

        if self.all_shortcut:
            x = x + x_shortcut

        x = self.norm(x)

        if shape_len == 2:
            x = x.squeeze(0)
        elif shape_len == 4:
            x = x.transpose(1, 2)
            x = x.reshape(batch, C, int(num_patches ** 0.5), int(num_patches ** 0.5))
        return x


def region_partition(x, region_size):
    """
    Args:
        x: (B, H, W, C)
        region_size (int): region size
    Returns:
        regions: (num_regions*B, region_size, region_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
    return regions

def region_reverse(regions, region_size, H, W):
    """
    Args:
        regions: (num_regions*B, region_size, region_size, C)
        region_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)