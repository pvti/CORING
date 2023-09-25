import os
import datetime
import torch
import argparse
from collections import OrderedDict
import cv2
import numpy as np

from models.imagenet.resnet import resnet_50
import utils.common as utils
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
    GradCAMElementWise,
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    preprocess_image,
)


parser = argparse.ArgumentParser("CAM visualization")

parser.add_argument("--input", type=str, default="~/sim3/", help="path to dataset")

parser.add_argument(
    "--job_dir", type=str, default="CAM", help="path for saving trained models"
)

parser.add_argument(
    "--compress_rate",
    type=str,
    default="[0.]+[0.5]*3+[0.6]*16",
    help="compress rate of each conv",
)

parser.add_argument(
    "--model-ori", type=str, default="checkpoint/resnet_50.pth", help="ori model path"
)
parser.add_argument(
    "--model-hrank", type=str, default="hrank-extreme.pt", help="hrank model path"
)
parser.add_argument(
    "--model-chip", type=str, default="chip-extreme.pt", help="chip model path"
)
parser.add_argument(
    "--model-coring", type=str, default="coring-extreme.pt", help="coring model path"
)

parser.add_argument("--batch_size", type=int, default=1, help="batch size")

parser.add_argument("--gpu", type=str, default="0", help="Select gpu to use")

parser.add_argument(
    "--aug-smooth",
    action="store_true",
    help="Apply test time augmentation to smooth the CAM",
)
parser.add_argument(
    "--eigen-smooth",
    action="store_true",
    help="Reduce noise by taking the first principle component"
    "of cam_weights*activations",
)
parser.add_argument(
    "--method",
    type=str,
    default="gradcam",
    choices=[
        "gradcam",
        "hirescam",
        "gradcam++",
        "scorecam",
        "xgradcam",
        "ablationcam",
        "eigencam",
        "eigengradcam",
        "layercam",
        "fullgrad",
        "gradcamelementwise",
    ],
    help="CAM method",
)

args = parser.parse_args()

if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logger = utils.get_logger(os.path.join(args.job_dir, "test_" + now + ".log"))


def main():
    logger.info("args = %s", args)

    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
    }

    cam_algorithm = methods[args.method]

    if args.compress_rate:
        import re

        cprate_str = args.compress_rate
        cprate_str_list = cprate_str.split("+")
        pat_cprate = re.compile(r"\d+\.\d*")
        pat_num = re.compile(r"\*\d+")
        cprate = []
        for x in cprate_str_list:
            num = 1
            find_num = re.findall(pat_num, x)
            if find_num:
                assert len(find_num) == 1
                num = int(find_num[0].replace("*", ""))
            find_cprate = re.findall(pat_cprate, x)
            assert len(find_cprate) == 1
            cprate += [float(find_cprate[0])] * num

        compress_rate = cprate

    logger.info("==> Building models..")
    model_ori = resnet_50(compress_rate=[0.0] * 100)
    model_hrank = resnet_50(compress_rate=compress_rate)
    model_chip = resnet_50(compress_rate=compress_rate)
    model_coring = resnet_50(compress_rate=compress_rate)

    logger.info("==> Loading checkpoints..")
    state_dict_ori = torch.load(args.model_ori)
    model_ori.load_state_dict(state_dict_ori)

    state_dict_hrank = torch.load(args.model_hrank)
    model_hrank.load_state_dict(state_dict_hrank["state_dict"], strict=False)

    state_dict_chip = torch.load(args.model_chip)
    model_chip.load_state_dict(state_dict_chip["state_dict"], strict=False)

    state_dict_coring = torch.load(args.model_coring)
    new_state_dict = OrderedDict()
    for k, v in state_dict_coring["state_dict"].items():
        name = k[7:]
        new_state_dict[name] = v
    model_coring.load_state_dict(new_state_dict)

    logger.info("==> Processing input..")
    class_name = args.input.split("/")[-1]
    output = os.path.join(args.job_dir, class_name)
    if not os.path.exists(output):
        os.mkdir(output)
    if os.path.exists(args.input):
        imgs = os.listdir(args.input)
        for img in imgs:
            img_path = os.path.join(args.input, img)
            rgb_img, input_tensor = preprocess(img_path)

            cam_image_ori = CAM(
                input_tensor=input_tensor,
                rgb_img=rgb_img,
                model=model_ori,
                cam_algorithm=cam_algorithm,
            )
            cam_image_hrank = CAM(
                input_tensor=input_tensor,
                rgb_img=rgb_img,
                model=model_hrank,
                cam_algorithm=cam_algorithm,
            )
            cam_image_chip = CAM(
                input_tensor=input_tensor,
                rgb_img=rgb_img,
                model=model_chip,
                cam_algorithm=cam_algorithm,
            )
            cam_image_coring = CAM(
                input_tensor=input_tensor,
                rgb_img=rgb_img,
                model=model_coring,
                cam_algorithm=cam_algorithm,
            )

            img_name = img.split(".")[0]
            cam_ori_path = os.path.join(output, f"{img_name}_ori.jpg")
            cam_hrank_path = os.path.join(output, f"{img_name}_hrank.jpg")
            cam_chip_path = os.path.join(output, f"{img_name}_chip.jpg")
            cam_coring_path = os.path.join(output, f"{img_name}_coring.jpg")

            cv2.imwrite(cam_ori_path, cam_image_ori)
            cv2.imwrite(cam_hrank_path, cam_image_hrank)
            cv2.imwrite(cam_chip_path, cam_image_chip)
            cv2.imwrite(cam_coring_path, cam_image_coring)


def preprocess(img_path):
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(
        rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return rgb_img, input_tensor


def CAM(input_tensor, rgb_img, model, cam_algorithm):
    target_layers = [model.layer4]
    targets = None
    with cam_algorithm(model=model, target_layers=target_layers, use_cuda=False) as cam:
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    return cam_image


if __name__ == "__main__":
    main()
