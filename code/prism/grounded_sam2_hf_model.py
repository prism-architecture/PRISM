import argparse
import os
# import cv2
import json
import torch
import numpy as np
# import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
# from supervision.draw.color import ColorPalette
# from gsam_utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from torchvision.ops import nms
from collections import defaultdict




def detect_objects(
    text_prompt: str = "left robot. right robot. red block. black block. blue block. yellow block.",
    img_path: str = None,
    canonical_labels: list = None,
    grounding_model: str = "IDEA-Research/grounding-dino-tiny",
    sam2_checkpoint: str = "/home/sujan/VoxPoser_Feb 28/VoxPoser/src/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
    sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    output_dir: str = "outputs/test_sam2.1",
    no_dump_json: bool = False,
    force_cpu: bool = False,
):

    GROUNDING_MODEL = grounding_model
    TEXT_PROMPT = text_prompt
    IMG_PATH = img_path
    SAM2_CHECKPOINT = sam2_checkpoint
    SAM2_MODEL_CONFIG = sam2_model_config
    DEVICE = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    OUTPUT_DIR = Path(output_dir)
    DUMP_JSON_RESULTS = not no_dump_json

    # create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # environment settings
    # use bfloat16
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino from huggingface
    model_id = GROUNDING_MODEL
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)


    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = TEXT_PROMPT
    img_path = IMG_PATH

    image = Image.open(img_path)

    sam2_predictor.set_image(np.array(image.convert("RGB")))

    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )



    # Extract relevant outputs from results[0]
    boxes = results[0]['boxes']
    scores = results[0]['scores']
    labels = results[0]['labels']



    # after performing non maximum supression to remove duplicate

    # Step 1: Apply NMS to keep non-overlapping boxes
    # iou_threshold = 0.5
    # keep_indices = nms(boxes, scores, iou_threshold)

    # # Filtered outputs
    # filtered_boxes = boxes[keep_indices]
    # filtered_scores = scores[keep_indices]
    # filtered_labels = [labels[i] for i in keep_indices]



    # Classes for which we skip NMS
    SKIP_NMS_CLASSES = {'small red block'}

    # Group indices by class label
    label_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[label.lower()].append(i)

    final_keep_indices = []

    # Apply NMS selectively
    for label, indices in label_to_indices.items():
        if not indices:
            continue
        label_boxes = boxes[indices]
        label_scores = scores[indices]

        # Skip NMS for certain classes
        if label in SKIP_NMS_CLASSES:
            final_keep_indices.extend(indices)
        else:
            keep = nms(label_boxes, label_scores, iou_threshold=0.5)
            final_keep_indices.extend([indices[i] for i in keep])

    # Final filtered outputs
    filtered_boxes = boxes[final_keep_indices]
    filtered_scores = scores[final_keep_indices]
    filtered_labels = [labels[i] for i in final_keep_indices]






    # Step 2: Identify robot candidates by label
    robot_indices = []
    for i, label in enumerate(filtered_labels):
        if 'robot' in label.lower():
            center_x = ((filtered_boxes[i][0] + filtered_boxes[i][2]) / 2).item()
            robot_indices.append((i, center_x))

    # Step 3: Identify left and right robots
    robot_indices.sort(key=lambda x: x[1])  # sort by x-center
    gripper_results = []

    if len(robot_indices) >= 1:
        left_idx = robot_indices[0][0]
        gripper_results.append({
            'label': 'gripper1',
            'box': filtered_boxes[left_idx].tolist()
        })

    if len(robot_indices) >= 2:
        right_idx = robot_indices[-1][0]
        if right_idx != left_idx:
            gripper_results.append({
                'label': 'gripper',
                'box': filtered_boxes[right_idx].tolist()
            })

    # Step 4: Add all other non-robot objects as-is (from NMS-filtered set)
    final_result = gripper_results.copy()
    robot_idx_set = {idx for idx, _ in robot_indices}
    for i in range(len(filtered_boxes)):
        if i not in [r['box'] for r in gripper_results] and i not in [idx for idx, _ in robot_indices]:
            final_result.append({
                'label': filtered_labels[i],
                'box': filtered_boxes[i].tolist()
            })

    from difflib import get_close_matches

    # Canonical object labels (from your original text prompt)
    canonical_labels = [
        "red block", "black block", "blue block", "yellow block"
    ]

    # Function to clean up and normalize label
    def normalize_label(label):
        if 'robot' in label.lower():
            return label  # don't modify robot labels
        match = get_close_matches(label.lower(), canonical_labels, n=1, cutoff=0.6)
        return f"{match[0]}" if match else f"{label}"

    # Apply normalization to final_result (non-robot objects only)
    for obj in final_result:
        if 'robot' not in obj['label'].lower():
            obj['label'] = normalize_label(obj['label'])

    # Display cleaned final result
    from pprint import pprint
    # pprint(final_result)

    return final_result


 
 # end of the function



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--text-prompt", default="left robot. right robot. red cup. green cup. magenta cup. small red block.")
    parser.add_argument("--img-path", default="/home/sujan/VoxPoser_Feb 28/VoxPoser/data/shell_game/episode1/front/front.png")
    parser.add_argument("--sam2-checkpoint", default="/home/sujan/VoxPoser_Feb 28/VoxPoser/src/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="outputs/test_sam2.1")
    parser.add_argument("--no-dump-json", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    GROUNDING_MODEL = args.grounding_model
    TEXT_PROMPT = args.text_prompt
    IMG_PATH = args.img_path
    SAM2_CHECKPOINT = args.sam2_checkpoint
    SAM2_MODEL_CONFIG = args.sam2_model_config
    DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    OUTPUT_DIR = Path(args.output_dir)
    DUMP_JSON_RESULTS = not args.no_dump_json

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)

    image = Image.open(IMG_PATH)
    sam2_predictor.set_image(np.array(image.convert("RGB")))

    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    input_boxes = results[0]["boxes"].cpu().numpy()
    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    class_ids = np.array(list(range(len(class_names))))
    labels = [f"{cls} {score:.2f}" for cls, score in zip(class_names, confidences)]

    img = cv2.imread(IMG_PATH)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

    if DUMP_JSON_RESULTS:
        def single_mask_to_rle(mask):
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        results = {
            "image_path": IMG_PATH,
            "annotations": [
                {
                    "class_name": cls,
                    "bbox": box.tolist(),
                    "segmentation": rle,
                    "score": score,
                }
                for cls, box, rle, score in zip(class_names, input_boxes, mask_rles, scores.tolist())
            ],
            "box_format": "xyxy",
            "img_width": image.width,
            "img_height": image.height,
        }

        with open(os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
            json.dump(results, f, indent=4)
