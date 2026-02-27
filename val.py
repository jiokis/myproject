# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlpackage          # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.dataloaders import create_classification_dataloader
from utils.metrics import calculate_topo_metrics  # 新增导入

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_img_size,
    check_requirements,
    check_yaml,
    colorstr,
    increment_path,
    print_args,
    xyxy2xywh,
)
from utils.metrics import box_iou
from utils.plots import plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    """Saves one detection result to a txt file in normalized xywh format, optionally including confidence.

    Args:
        predn (torch.Tensor): Predicted bounding boxes and associated confidence scores and classes in xyxy format,
            tensor of shape (N, 6) where N is the number of detections.
        save_conf (bool): If True, saves the confidence scores along with the bounding box coordinates.
        shape (tuple): Shape of the original image as (height, width).
        file (str | Path): File path where the result will be saved.

    Returns:
        None

    Examples:
        ```python
        predn = torch.tensor([[10, 20, 30, 40, 0.9, 1]])  # example prediction
        save_one_txt(predn, save_conf=True, shape=(640, 480), file="output.txt")
        ```

    Notes:
        The xyxy bounding box format represents the coordinates (xmin, ymin, xmax, ymax).
        The xywh format represents the coordinates (center_x, center_y, width, height) and is normalized by the width and
        height of the image.
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    """Saves a single JSON detection result, including image ID, category ID, bounding box, and confidence score.

    Args:
        predn (torch.Tensor): Predicted detections in xyxy format with shape (n, 6) where n is the number of detections.
            The tensor should contain [x_min, y_min, x_max, y_max, confidence, class_id] for each detection.
        jdict (list[dict]): List to collect JSON formatted detection results.
        path (pathlib.Path): Path object of the image file, used to extract image_id.
        class_map (dict[int, int]): Mapping from model class indices to dataset-specific category IDs.

    Returns:
        None: Appends detection results as dictionaries to `jdict` list in-place.

    Examples:
        ```python
        predn = torch.tensor([[100, 50, 200, 150, 0.9, 0], [50, 30, 100, 80, 0.8, 1]])
        jdict = []
        path = Path("42.jpg")
        class_map = {0: 18, 1: 19}
        save_one_json(predn, jdict, path, class_map)
        ```
        This will append to `jdict`:
        ```
        [
            {'image_id': 42, 'category_id': 18, 'bbox': [125.0, 75.0, 100.0, 100.0], 'score': 0.9},
            {'image_id': 42, 'category_id': 19, 'bbox': [75.0, 55.0, 50.0, 50.0], 'score': 0.8}
        ]
        ```

    Notes:
        The `bbox` values are formatted as [x, y, width, height], where x and y represent the top-left corner of the box.
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(detections, labels, iouv):
    """Return a correct prediction matrix given detections and labels at various IoU thresholds.

    Args:
        detections (np.ndarray): Array of shape (N, 6) where each row corresponds to a detection with format [x1, y1,
            x2, y2, conf, class].
        labels (np.ndarray): Array of shape (M, 5) where each row corresponds to a ground truth label with format
            [class, x1, y1, x2, y2].
        iouv (np.ndarray): Array of IoU thresholds to evaluate at.

    Returns:
        correct (np.ndarray): A binary array of shape (N, len(iouv)) indicating whether each detection is a true
            positive for each IoU threshold. There are 10 IoU levels used in the evaluation.

    Examples:
        ```python
        detections = np.array([[50, 50, 200, 200, 0.9, 1], [30, 30, 150, 150, 0.7, 0]])
        labels = np.array([[1, 50, 50, 200, 200]])
        iouv = np.linspace(0.5, 0.95, 10)
        correct = process_batch(detections, labels, iouv)
        ```

    Notes:
        - This function is used as part of the evaluation pipeline for object detection models.
        - IoU (Intersection over Union) is a common evaluation metric for object detection performance.
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


"""Validates a YOLOv5 classification model on a dataset, computing metrics like top1 and top5 accuracy."""


@smart_inference_mode()
def run(
    data=ROOT / "../datasets/mnist",  # dataset dir
    weights=ROOT / "yolov5s-cls.pt",  # model.pt path(s)
    batch_size=128,  # batch size
    imgsz=224,  # inference size (pixels)
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    verbose=False,  # verbose output
    project=ROOT / "runs/val-cls",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    criterion=None,
    pbar=None,
    topo_metrics=False,  # 新增参数：是否计算拓扑指标
):
    """Validates a YOLOv5 classification model on a dataset, computing metrics like top1 and top5 accuracy."""
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Dataloader
        data = Path(data)
        test_dir = data / "test" if (data / "test").exists() else data / "val"  # data/test or data/val
        dataloader = create_classification_dataloader(
            path=test_dir, imgsz=imgsz, batch_size=batch_size, augment=False, rank=-1, workers=workers
        )

    model.eval()
    pred, targets, loss, dt = [], [], 0, (Profile(device=device), Profile(device=device), Profile(device=device))
    feats_list = []  # 新增：存储中间特征
    n = len(dataloader)  # number of batches
    action = "validating" if dataloader.dataset.root.stem == "val" else "testing"
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"
    bar = tqdm(dataloader, desc, n, not training, bar_format=TQDM_BAR_FORMAT, position=0)

    with torch.cuda.amp.autocast(enabled=device.type != "cpu"):
        for images, labels in bar:
            with dt[0]:
                images, labels = images.to(device, non_blocking=True), labels.to(device)

            with dt[1]:
                y = model(images)
                # 新增：若启用拓扑指标，获取中间特征（需模型支持特征提取）
                if topo_metrics:
                    # 假设模型推理时会存储中间特征到feats属性
                    # 实际使用时需确保模型forward方法正确设置了中间特征
                    if hasattr(model, "feats"):
                        feats_list.append(model.feats.detach().cpu())  # 存储特征并转移到CPU
                    else:
                        LOGGER.warning("Model does not have 'feats' attribute, cannot compute topological metrics")
                        topo_metrics = False  # 关闭指标计算

            with dt[2]:
                pred.append(y.argsort(1, descending=True)[:, :5])
                targets.append(labels)
                if criterion:
                    loss += criterion(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()

    if pbar:
        pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"
    if verbose:  # all classes
        LOGGER.info(f"{'Class':>24}{'Images':>12}{'top1_acc':>12}{'top5_acc':>12}")
        LOGGER.info(f"{'all':>24}{targets.shape[0]:>12}{top1:>12.3g}{top5:>12.3g}")
        for i, c in model.names.items():
            acc_i = acc[targets == i]
            top1i, top5i = acc_i.mean(0).tolist()
            LOGGER.info(f"{c:>24}{acc_i.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}")

        # Print results
        t = tuple(x.t / len(dataloader.dataset.samples) * 1e3 for x in dt)  # speeds per image
        shape = (1, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}" % t)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    # 新增：计算并返回拓扑指标
    if topo_metrics and feats_list:
        # 合并所有批次的特征
        feats = torch.cat(feats_list, dim=0)
        # 调用拓扑指标计算函数（需自行实现calculate_topo_metrics）
        topo_results = calculate_topo_metrics(pred=pred, feats=feats, targets=targets, imgsz=imgsz)
        return top1, top5, loss, topo_results
    else:
        return top1, top5, loss


def parse_opt():
    """Parse command-line options for configuring YOLOv5 model inference.

    Args:
        data (str, optional): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        weights (list[str], optional): List of paths to model weight files. Default is 'yolov5s.pt'.
        batch_size (int, optional): Batch size for inference. Default is 32.
        imgsz (int, optional): Inference image size in pixels. Default is 640.
        conf_thres (float, optional): Confidence threshold for predictions. Default is 0.001.
        iou_thres (float, optional): IoU threshold for Non-Max Suppression (NMS). Default is 0.6.
        max_det (int, optional): Maximum number of detections per image. Default is 300.
        task (str, optional): Task type - options are 'train', 'val', 'test', 'speed', or 'study'. Default is 'val'.
        device (str, optional): Device to run the model on. e.g., '0' or '0,1,2,3' or 'cpu'. Default is empty to let the
            system choose automatically.
        workers (int, optional): Maximum number of dataloader workers per rank in DDP mode. Default is 8.
        single_cls (bool, optional): If set, treats the dataset as a single-class dataset. Default is False.
        augment (bool, optional): If set, performs augmented inference. Default is False.
        verbose (bool, optional): If set, reports mAP by class. Default is False.
        save_txt (bool, optional): If set, saves results to *.txt files. Default is False.
        save_hybrid (bool, optional): If set, saves label+prediction hybrid results to *.txt files. Default is False.
        save_conf (bool, optional): If set, saves confidences in --save-txt labels. Default is False.
        save_json (bool, optional): If set, saves results to a COCO-JSON file. Default is False.
        project (str, optional): Project directory to save results to. Default is 'runs/val'.
        name (str, optional): Name of the directory to save results to. Default is 'exp'.
        exist_ok (bool, optional): If set, existing directory will not be incremented. Default is False.
        half (bool, optional): If set, uses FP16 half-precision inference. Default is False.
        dnn (bool, optional): If set, uses OpenCV DNN for ONNX inference. Default is False.

    Returns:
        argparse.Namespace: Parsed command-line options.

    Examples:
        To validate a trained YOLOv5 model on a COCO dataset:
        ```python
        $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640
        ```
        Different model formats could be used instead of `yolov5s.pt`:
        ```python
        $ python val.py --weights yolov5s.pt yolov5s.torchscript yolov5s.onnx yolov5s_openvino_model yolov5s.engine
        ```
        Additional options include saving results in different formats, selecting devices, and more.

    Notes:
        - The '--data' parameter is checked to ensure it ends with 'coco.yaml' if '--save-json' is set.
        - The '--save-txt' option is set to True if '--save-hybrid' is enabled.
        - Args are printed using `print_args` to facilitate debugging.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 tasks like training, validation, testing, speed, and study benchmarks based on provided options.

    Args:
        opt (argparse.Namespace): Parsed command-line options. This includes values for parameters like 'data',
            'weights', 'batch_size', 'imgsz', 'conf_thres', 'iou_thres', 'max_det', 'task', 'device', 'workers',
            'single_cls', 'augment', 'verbose', 'save_txt', 'save_hybrid', 'save_conf', 'save_json', 'project', 'name',
            'exist_ok', 'half', and 'dnn', essential for configuring the YOLOv5 tasks.

    Returns:
        None

    Examples:
        To validate a trained YOLOv5 model on the COCO dataset with a specific weights file, use:
        ```python
        $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640
        ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
