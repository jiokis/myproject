import torch
import numpy as np
from tqdm import tqdm
from torch.optim import lr_scheduler
from scipy.ndimage import label  # 补充scipy导入
import torch.nn as nn
# 在train_topo.py最顶部添加（导入后直接写）
import argparse


def parse_opt():
    """解析训练参数（适配train_topo函数需求）"""
    parser = argparse.ArgumentParser()
    # 基础训练参数
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--save-dir', type=str, default='runs/train_coco_modified', help='结果保存目录')
    parser.add_argument('--exist-ok', action='store_true', help='允许覆盖现有目录')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='数据集配置文件')
    parser.add_argument('--cfg', type=str, default='models/yolov5s_modified.yaml', help='模型配置文件')
    parser.add_argument('--weights', type=str, default='', help='预训练权重（空表示从零训练）')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮次')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0', help='GPU设备（0表示第一块GPU）')
    parser.add_argument('--single-cls', action='store_true', help='是否单类别训练')
    parser.add_argument('--cache', action='store_true', help='是否缓存数据到内存')
    parser.add_argument('--rect', action='store_true', help='是否用矩形训练')
    parser.add_argument('--workers', type=int, default=2, help='数据加载线程数（Colab建议≤2）')
    parser.add_argument('--image-weights', action='store_true', help='是否用图像权重采样')
    parser.add_argument('--quad', action='store_true', help='是否用四通道加载')

    # 拓扑相关参数（关键！）
    parser.add_argument('--topo-monitor', action='store_true', help='是否开启拓扑监控（记录曲率/能量）')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp_topo.yaml', help='拓扑超参文件（含energy_beta等）')

    opt = parser.parse_args()
    return opt


# 主函数调用（train_topo.py末尾添加）


# 补充YOLOv5必要导入
from utils.dataloaders import create_dataloader
from utils.general import (

    check_img_size,
    init_seeds,
    check_dataset,
    increment_path,
    LOGGER,  # 日志对象
    colorstr,  # 颜色格式化函数
)
from utils.loss import ComputeLoss  # 替换YOLOv5Loss为实际损失类
from utils.torch_utils import (
    WORLD_SIZE,  # 分布式相关
    LOCAL_RANK,
    smart_optimizer, select_device
    # 优化器创建工具
)
from models.yolo import DetectionModel  # 模型类（检测任务）
from models.common import Conv  # 基础卷积类
import val as validate  # 验证模块


# 1. 定义自定义曲率感知卷积类（解决CurvatureAwareConv标红）
class CurvatureAwareConv(nn.Module):
    """曲率感知卷积（示例实现，需根据实际需求调整）"""
    def __init__(self, c1, c2, k=1, r_base=8):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=1, padding=k//2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()  # 沿用YOLOv5激活函数
        self.r_base = r_base  # 曲率感知参数

    def forward(self, x):
        # 示例：在普通卷积基础上增加曲率感知逻辑（需根据论文实现）
        x = self.conv(x)
        x = self.bn(x)
        # 这里可添加曲率计算相关操作（如基于r_base调整特征）
        return self.act(x)


# 2. 拓扑相关工具函数（保持不变）
def riemann_energy(feat_map):
    """计算特征图的黎曼能量（基于二阶导数的近似）"""
    grad_x = torch.abs(feat_map[:, :, 1:, :] - feat_map[:, :, :-1, :])
    grad_y = torch.abs(feat_map[:, :, :, 1:] - feat_map[:, :, :, :-1])
    return (grad_x.mean() + grad_y.mean()) / 2  # 简化版能量计算


def compute_euler_characteristic(feat_map, threshold=0.5):
    """近似计算特征图的欧拉示性数（基于连通分量）"""
    binary_map = (feat_map > threshold).float()  # 二值化特征图
    b, c, h, w = binary_map.shape
    total = 0
    for i in range(b):
        for j in range(c):
            labeled, num = label(binary_map[i, j].cpu().numpy())  # 已提前导入label
            total += num  # 简化为连通分量数
    return total / (b * c)  # 平均欧拉示性数


def adjust_topology_params(model, train_loader, num_samples=100):
    """基于训练集特征分布调整拓扑参数（如曲率感知卷积的r_base）"""
    model.eval()
    feat_vars = []
    with torch.no_grad():
        for imgs, _, _, _ in train_loader:
            imgs = imgs.to(next(model.parameters()).device) / 255.0
            _, feats = model(imgs)  # 获取最后一层特征
            feat_vars.append(torch.var(feats[-1]).item())
            if len(feat_vars) >= num_samples:
                break
    avg_var = np.mean(feat_vars)
    # 调整曲率感知卷积的r_base
    for m in model.modules():
        if isinstance(m, CurvatureAwareConv):
            m.r_base = max(4, int(8 * avg_var / 0.1))  # 归一化到基准值0.1
    model.train()
    return model


def train_topo(opt):
    # 1. 初始化基础配置
    init_seeds(opt.seed)
    device = select_device(opt.device, batch_size=opt.batch_size)
    save_dir = increment_path(opt.save_dir, exist_ok=opt.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. 日志初始化（使用YOLOv5原生GenericLogger）
    logger = None
    if LOCAL_RANK in {-1, 0}:  # 仅主进程初始化日志
        from yolov5.utils.loggers import GenericLogger  # 替换Loggers为实际日志类
        logger = GenericLogger(opt=opt, console_logger=LOGGER)

    # 3. 数据集加载
    data_dict = check_dataset(opt.data)
    nc = int(data_dict["nc"])
    train_path, val_path = data_dict["train"], data_dict["val"]

    # 计算网格大小（与模型步长匹配）
    model_dummy = DetectionModel(opt.cfg)  # 替换Model为实际模型类
    gs = max(int(model_dummy.stride.max()), 32)
    del model_dummy
    imgsz = check_img_size(opt.imgsz, gs)

    # 训练集加载器（适配分布式参数）
    train_loader, dataset = create_dataloader(
        path=train_path,
        imgsz=imgsz,
        batch_size=opt.batch_size // WORLD_SIZE,
        gs=gs,
        single_cls=opt.single_cls,
        hyp=opt.hyp,
        augment=True,
        cache=opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=opt.workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True
    )

    # 验证集加载器
    val_loader = None
    if LOCAL_RANK in {-1, 0}:  # 仅主进程加载验证集
        val_loader = create_dataloader(
            path=val_path,
            imgsz=imgsz,
            batch_size=opt.batch_size // WORLD_SIZE * 2,
            gs=gs,
            single_cls=opt.single_cls,
            hyp=opt.hyp,
            augment=False,
            cache=opt.cache,
            rect=True,
            rank=-1,
            workers=opt.workers * 2,
            pad=0.5,
            prefix=colorstr("val: ")
        )[0]

    # 4. 模型初始化与拓扑结构改造
    model = DetectionModel(opt.cfg, ch=3, nc=nc).to(device)  # 初始化检测模型
    # 替换1x1卷积为曲率感知卷积
    for i, m in enumerate(model.model):
        if isinstance(m, Conv) and m.stride == (1, 1) and m.kernel_size == (1, 1):
            c1, c2 = m.conv.in_channels, m.conv.out_channels
            model.model[i] = CurvatureAwareConv(c1, c2, k=1, r_base=8)  # 替换为自定义卷积

    # 5. 场景自适应拓扑参数调整
    model = adjust_topology_params(model, train_loader)

    # 6. 损失函数（使用YOLOv5原生ComputeLoss，扩展支持能量约束）
    compute_loss = ComputeLoss(
        model,  # 传入模型（原生损失类需要模型参数）
        energy_beta=opt.hyp.get('energy_beta', 0.1),  # 扩展参数：能量损失权重
        homotopy_tol=opt.hyp.get('homotopy_tol', 1e-3)
    )

    # 7. 优化器与学习率调度器
    optimizer = smart_optimizer(
        model,
        opt.optimizer,
        lr0=opt.hyp['lr0'],
        momentum=opt.hyp['momentum'],
        decay=opt.hyp['weight_decay']
    )
    lf = lambda x: (1 - x / opt.epochs) * (1 - opt.hyp['lrf']) + opt.hyp['lrf']  # 线性调度
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 8. 训练循环
    model.train()
    for epoch in range(opt.epochs):
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{opt.epochs}")
        for i, (imgs, targets, paths, _) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True) / 255.0
            targets = targets.to(device, non_blocking=True)

            # 前向传播（含特征提取）
            with torch.cuda.amp.autocast(enabled=device.type != 'cpu'):
                preds, feats = model(imgs)  # 假设模型输出(preds, 多尺度特征)
                loss, loss_items = compute_loss(preds, targets, feats)  # 传入特征计算能量损失

            # 拓扑监控（每10个batch）
            if i % 10 == 0 and opt.topo_monitor and LOCAL_RANK in {-1, 0}:
                curvature = torch.var(feats[-1]).item()
                energy = riemann_energy(feats[-1]).item()
                euler_chi = compute_euler_characteristic(feats[-1])

                # 日志记录
                if logger:
                    logger.log_scalars(
                        {
                            'topology/curvature': curvature,
                            'topology/energy': energy,
                            'topology/euler_chi': euler_chi,
                            'train/loss': loss.item()
                        },
                        step=epoch * len(train_loader) + i
                    )
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    curvature=f"{curvature:.4f}",
                    energy=f"{energy:.6f}"
                )

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # 梯度裁剪
            optimizer.step()

        # 每5个epoch验证（仅主进程）
        if (epoch + 1) % 5 == 0 and LOCAL_RANK in {-1, 0} and val_loader:
            metrics = validate.run(
                model=model,
                dataloader=val_loader,
                device=device,
                topo_metrics=True  # 需确保validate.run支持该参数
            )
            if logger:
                logger.log_scalars({**metrics, 'lr': optimizer.param_groups[0]['lr']}, step=epoch)

        scheduler.step()

    # 保存最终模型（仅主进程）
    if LOCAL_RANK in {-1, 0}:
        final_ckpt = {
            'model': model.state_dict(),
            'opt': vars(opt),
            'hyp': opt.hyp,
            'epoch': opt.epochs
        }
        torch.save(final_ckpt, save_dir / 'last_topo.pt')
        LOGGER.info(f"最终模型保存至 {save_dir / 'last_topo.pt'}")

def adjust_topology_params(model, train_loader):
    """基于训练集特征分布调整拓扑参数（简化版）"""
    # 随机采样10个batch计算特征统计量
    feats_list = []
    model.eval()
    with torch.no_grad():
        for imgs, _, _, _ in [next(iter(train_loader)) for _ in range(10)]:
            _, feats = model(imgs.to(next(model.parameters()).device))
            feats_list.extend(feats)
    model.train()

    # 计算特征平均方差（曲率基准）
    mean_var = torch.mean(torch.tensor([torch.var(f) for f in feats_list])).item()

    # 调整CurvatureAwareConv的曲率参数
    for m in model.modules():
        if isinstance(m, CurvatureAwareConv):
            m.curvature.data = torch.tensor(mean_var * 0.1, device=m.curvature.device)  # 动态更新
    return model


if __name__ == '__main__':
    opt = parse_opt()
    train_topo(opt)