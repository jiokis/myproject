# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`."""
    return 1.0 - 0.5 * eps, 0.5 * eps


def riemann_energy(x):
    """计算黎曼能量 E_Riem = ∫det(g(P))dP，用协方差矩阵近似度规张量"""
    # 展平特征维度 (batch*anchors*grid_h*grid_w, features)
    x_flat = x.view(-1, x.size(-1))
    if x_flat.size(0) < 2:  # 避免样本不足导致协方差计算错误
        return torch.tensor(0.0, device=x.device)
    g = torch.cov(x_flat.T)  # 度规张量近似
    return torch.det(g)  # 信息密度积分


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects."""

    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss


class ComputeLoss:
    """Computes total loss for YOLOv5 including classification, box, objectness and energy constraint losses."""

    sort_obj_iou = False

    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device
        h = model.hyp

        # 基础损失函数定义
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # 标签平滑
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))

        #  focal loss
        g = h["fl_gamma"]
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # 模型参数
        m = de_parallel(model).model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(m.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na, self.nc, self.nl, self.anchors, self.device = m.na, m.nc, m.nl, m.anchors, device

        # 新增：能量约束参数
        self.energy_beta = h.get("energy_beta", 0.01)  # 从超参获取权重，默认0.01


    def __call__(self, p, targets):
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        lenergy = torch.zeros(1, device=self.device)  # 新增：能量损失
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        # 计算基础损失
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)

            if n := b.shape[0]:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)

                # 边界框损失
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()

                # 目标性损失
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou

                # 分类损失
                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)

            # 目标性损失累计
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

            # 新增：计算当前层的黎曼能量损失
            layer_energy = riemann_energy(pi)
            lenergy += torch.var(layer_energy)  # 能量波动惩罚


        # 自动平衡
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # 损失加权
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        lenergy *= self.energy_beta  # 能量损失加权

        bs = tobj.shape[0]
        total_loss = (lbox + lobj + lcls + lenergy) * bs  # 总损失包含能量项
        # 返回总损失和各分量（新增能量损失到返回值）
        return total_loss, torch.cat((lbox, lobj, lcls, lenergy)).detach()


    def build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        g = 0.5
        off = (torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * g)

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]

            t = targets * gain
            if nt:
                r = t[..., 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]
                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            indices.append((b, a, gj.clamp_(0, shape[2]-1), gi.clamp_(0, shape[3]-1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch