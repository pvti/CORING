import cv2
import time
import random
import numpy as np
import matplotlib as mpl
import torch
import argparse
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    keypointrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    KeypointRCNN_ResNet50_FPN_Weights,
)
import model
from utils import get_cpr


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Faster/Mask/KeypointRCNN_ResNet50_FPN Video Inference",
        add_help=add_help,
    )
    parser.add_argument(
        "--model", default="fasterrcnn_resnet50_fpn", type=str, help="model name"
    )
    parser.add_argument(
        "--custom", dest="custom", action="store_true", help="Use custom model"
    )
    parser.add_argument("--weight", help="Path to custom model weight file")
    parser.add_argument(
        "-cpr",
        "--compress_rate",
        type=str,
        default="[0.]+[0.5]*3+[0.6]*16",
        help="list of compress rate of each layer",
    )
    parser.add_argument("--input", default="input.mp4", help="Path to input video file")
    parser.add_argument(
        "--output", default="output.mp4", help="Path to output video file"
    )
    parser.add_argument("--fps", type=int, default=10, help="FPS to write output video")
    parser.add_argument(
        "--confidence", type=float, default=0.75, help="confidence threshold"
    )
    parser.add_argument("--device", default="cuda:0", help="Device to use (cuda/cpu)")

    return parser


def main(args):
    device = torch.device(args.device)

    if "faster" in args.model:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    elif "mask" in args.model:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    else:
        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    preprocess = weights.transforms()
    labels = weights.meta["categories"]
    # Create a color map for each class
    num_classes = len(labels)
    color_map = {}
    for i in range(num_classes):
        color = (int(i * 255 / num_classes), 64, 128)
        color_map[i] = color
    color_map[1] = (0, 255, 0)  # make person green

    if args.custom:
        compress_rate = get_cpr(args.compress_rate)
        model = eval(f"model.{args.model}")(
            weights=args.weight, compress_rate=compress_rate
        )
    else:
        model = eval(args.model)(weights=weights)
    model = model.to(device)
    model.eval()

    # Open the video capture
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Initialize VideoWriter for output video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        args.output, fourcc, fps=args.fps, frameSize=(frame_width, frame_height)
    )

    # Start the FPS measurement
    start_time = time.time()
    frame_count = 0

    # Pairs of edges for 17 of the keypoints detected
    # omit any of the undesired connecting points
    edges = [
        (0, 1),
        (0, 2),
        (2, 4),
        (1, 3),
        (6, 8),
        (8, 10),
        (5, 7),
        (7, 9),
        (5, 11),
        (11, 13),
        (13, 15),
        (6, 12),
        (12, 14),
        (14, 16),
        (5, 6),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_frame = to_pil_image(frame)
        input_image = preprocess(pil_frame).to(device)

        # Measure inference time
        inference_start = time.time()
        with torch.no_grad():
            prediction = model([input_image])[0]
        inference_end = time.time()

        # Calculate inference speed (FPS)
        inference_time = inference_end - inference_start
        fps = 1 / inference_time

        # Draw bounding boxes and class names on the frame
        for score, label, box in zip(
            prediction["scores"], prediction["labels"], prediction["boxes"]
        ):
            if score > args.confidence:  # Adjust the confidence threshold as needed
                box = [int(coord) for coord in box.tolist()]  # Convert to integers
                class_name = labels[label]
                color = color_map[label.item()]
                frame = cv2.rectangle(
                    frame, (box[0], box[1]), (box[2], box[3]), color, 2
                )
                text = f"{class_name} {score.item():.2f}"
                frame = cv2.putText(
                    frame,
                    text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Draw masks
        if "mask" in args.model:
            pred_score = list(prediction["scores"].cpu().numpy())
            pred_t = [pred_score.index(x) for x in pred_score if x > args.confidence]
            if len(pred_t) > 0:
                masks = (prediction["masks"] > 0.5).squeeze().detach().cpu().numpy()
                masks = masks[: pred_t[-1] + 1]
                for i in range(len(masks)):
                    rgb_mask = random_colour_masks(masks[i])
                    frame = cv2.addWeighted(frame, 1, rgb_mask, 0.5, 0)

        # Draw keypoints
        if "keypoint" in args.model:
            for i in range(len(prediction["keypoints"])):
                # get the detected keypoints
                keypoints = prediction["keypoints"][i].cpu().detach().numpy()
                # proceed to draw the lines
                if prediction["scores"][i] > args.confidence:
                    keypoints = keypoints[:, :].reshape(-1, 3)
                    for p in range(keypoints.shape[0]):
                        # draw the keypoints
                        cv2.circle(
                            frame,
                            (int(keypoints[p, 0]), int(keypoints[p, 1])),
                            3,
                            (0, 0, 255),
                            thickness=-1,
                            lineType=cv2.FILLED,
                        )
                    # draw the lines joining the keypoints
                    for ie, e in enumerate(edges):
                        # get different colors for the edges
                        rgb = mpl.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                        rgb = rgb * 255
                        # join the keypoint pairs to draw the skeletal structure
                        cv2.line(
                            frame,
                            (int(keypoints[e, 0][0]), int(keypoints[e, 1][0])),
                            (int(keypoints[e, 0][1]), int(keypoints[e, 1][1])),
                            tuple(rgb),
                            2,
                            lineType=cv2.LINE_AA,
                        )

        # Write FPS
        fps_text = f"FPS {fps:.2f}"
        frame = cv2.putText(
            frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        # Write the frame to the output video
        out.write(frame)

        frame_count += 1
        print(f"Processed frame {frame_count} FPS {fps}")

    # Release the video capture and writer
    cap.release()
    out.release()

    # Calculate and print FPS
    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    print(f"Processed {frame_count} frames at {fps:.2f} FPS")


def random_colour_masks(mask):
    """
    Apply random colors to mask regions.

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        np.ndarray: Mask with random colors applied.
    """

    # Define a list of predefined colors
    colors = [
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250],
        [50, 190, 190],
    ]

    # Initialize empty color channels
    r, g, b = (
        np.zeros_like(mask).astype(np.uint8),
        np.zeros_like(mask).astype(np.uint8),
        np.zeros_like(mask).astype(np.uint8),
    )

    # Assign random colors to mask regions
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colors[random.randrange(0, 10)]

    # Create a colored mask by stacking the color channels
    colored_mask = np.stack([r, g, b], axis=2)

    return colored_mask


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
