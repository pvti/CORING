import cv2
import time
import torch
import argparse
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from model import fasterrcnn_resnet50_fpn as custom_fasterrcnn_resnet50_fpn
from utils import get_cpr


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="FasterRCNN_ResNet50_FPN Video Inference", add_help=add_help
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
    parser.add_argument("--device", default="cuda:0", help="Device to use (cuda/cpu)")

    return parser


def main(args):
    device = torch.device(args.device)

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    preprocess = weights.transforms()
    labels = weights.meta["categories"]

    if args.custom:
        compress_rate = get_cpr(args.compress_rate)
        model = custom_fasterrcnn_resnet50_fpn(
            weights=args.weight, compress_rate=compress_rate
        )
    else:
        model = fasterrcnn_resnet50_fpn(weights=weights)
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
            if score > 0.5:  # Adjust the confidence threshold as needed
                box = [int(coord) for coord in box.tolist()]  # Convert to integers
                class_name = labels[label]
                frame = cv2.rectangle(
                    frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
                )
                text = f"{class_name}"
                frame = cv2.putText(
                    frame,
                    text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

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


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
