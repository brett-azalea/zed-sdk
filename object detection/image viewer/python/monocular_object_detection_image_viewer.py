########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
Modified from the original sample object_detection_image_viewer.py (3D bounding boxes)
Detect bags using pure YOLOv8 segmentation and draw 2D bounding boxes around them
Designed for ZED X One monocular camera with GMSL2 streaming
"""
import sys
import pyzed.sl as sl
import argparse
import cv2
import numpy as np
import time
from ultralytics import YOLO


# YOLO configuration for bag detection
YOLO_MODEL_PATH = "yolov8m-seg.pt"  # Same model as ZedCamera uses
CONFIDENCE_THRESHOLD = 0.4  # Confidence threshold (0.0 to 1.0)
IMAGE_SIZE = 640  # YOLO inference size
# COCO classes for bags: backpack=24, handbag=26, suitcase=28
BAG_CLASSES = [24, 26, 28]
COCO_CLASS_NAMES = {24: "backpack", 26: "handbag", 28: "suitcase"}


def parse_args(init, opt):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if (
            ip_str.replace(":", "").replace(".", "").isdigit()
            and len(ip_str.split(".")) == 4
            and len(ip_str.split(":")) == 2
        ):
            init.set_from_stream(ip_str.split(":")[0], int(ip_str.split(":")[1]))
            print("[Sample] Using Stream input, IP : ", ip_str)
        elif (
            ip_str.replace(":", "").replace(".", "").isdigit()
            and len(ip_str.split(".")) == 4
        ):
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ", ip_str)
        else:
            print("Invalid IP format. Using live stream")
    if "HD2K" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif "HD1200" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif "HD1080" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif "HD720" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif "SVGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif "VGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution) > 0:
        print("[Sample] No valid resolution entered. Using default")
    else:
        print("[Sample] Using default resolution")


def draw_yolo_detections(image, results):
    """
    Draw YOLOv8 detection results on the image
    """
    detection_count = 0

    if results and len(results.boxes) > 0:
        boxes = results.boxes.cpu().numpy()

        for i, box in enumerate(boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            # Get class name
            class_name = COCO_CLASS_NAMES.get(class_id, f"class_{class_id}")

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label with confidence
            label_text = f"{class_name} ({confidence:.2f})"
            cv2.putText(
                image,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)

            detection_count += 1

            # Debug info for first few detections
            if detection_count <= 3:
                print(
                    f"Detection {detection_count}: {class_name} at ({x1},{y1})-({x2},{y2}), conf: {confidence:.2f}"
                )

        # Draw segmentation masks if available
        # if hasattr(results, "masks") and results.masks is not None:
        #     masks = results.masks.cpu().numpy()
        #     for mask_data in masks:
        #         mask = (mask_data.data[0] * 255).astype(np.uint8)
        #         # Resize mask to image size
        #         mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        #         # Create colored mask overlay
        #         mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        #         # Blend with original image
        #         image = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

    return image, detection_count


def setup_yolo_detector(model_path=YOLO_MODEL_PATH):
    """
    Setup YOLOv8 detector
    """
    print(f"Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path)
        print("YOLO model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Make sure you have ultralytics installed: pip install ultralytics")
        return None


def main(opt):
    init_params = sl.InitParametersOne()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.set_from_stream(
        opt.ip_address.split(":")[0], int(opt.ip_address.split(":")[1])
    )
    init_params.camera_fps = 15
    init_params.sdk_verbose = 1
    parse_args(init_params, opt)

    zed = sl.CameraOne()

    print("Attempting to open ZED camera...")
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()

    print("Camera opened successfully!")

    # Wait a bit for camera to stabilize
    time.sleep(2)

    # Setup YOLO detector (no ZED SDK object detection needed!)
    yolo_model = setup_yolo_detector()
    if yolo_model is None:
        zed.close()
        exit(1)

    # Create ZED objects for image capture
    image = sl.Mat()

    print("Starting pure YOLO bag detection. Press 'q' to quit.")
    print(f"Using YOLO model: {YOLO_MODEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Target classes: {[COCO_CLASS_NAMES[c] for c in BAG_CLASSES]}")

    frame_count = 0
    total_detections = 0

    while True:
        grab_status = zed.grab()
        if grab_status == sl.ERROR_CODE.SUCCESS:
            frame_count += 1

            # Retrieve left image for display
            zed.retrieve_image(image)  # Changed: removed sl.VIEW.LEFT parameter

            # Convert ZED image to OpenCV format
            image_cv = image.get_data()

            # Convert BGRA to BGR for OpenCV and YOLO
            if image_cv.shape[2] == 4:  # BGRA
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2BGR)

            # Run YOLO detection (filter to bag classes only)
            try:
                results = yolo_model.predict(
                    image_cv,
                    classes=BAG_CLASSES,  # Only detect bags
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=IMAGE_SIZE,
                    verbose=False,
                )[0]

                # Draw detections on the image
                image_with_detections, detection_count = draw_yolo_detections(
                    image_cv.copy(), results
                )
                total_detections += detection_count

            except Exception as e:
                print(f"YOLO detection error: {e}")
                image_with_detections = image_cv.copy()
                detection_count = 0

            # Display detection info
            info_text = f"Bags detected: {detection_count} | Frame: {frame_count} | Total: {total_detections}"
            cv2.putText(
                image_with_detections,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Add model info
            model_text = f"YOLOv8-seg | Confidence: {CONFIDENCE_THRESHOLD}"
            cv2.putText(
                image_with_detections,
                model_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Show the image
            cv2.imshow("ZED X One - Pure YOLO Bag Detection", image_with_detections)

            # Check for exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                break
        else:
            print(f"Grab failed with status: {grab_status}")
            time.sleep(0.1)  # Wait a bit before retrying

    # Cleanup
    cv2.destroyAllWindows()
    image.free(memory_type=sl.MEM.CPU)

    # Close camera
    print("Shutting down...")
    zed.close()
    print("Camera closed.")
    print(
        f"Session summary: {frame_count} frames processed, {total_detections} total detections"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_svo_file",
        type=str,
        help="Path to an .svo file, if you want to replay it",
        default="",
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        help="IP Address, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup",
        default="",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA",
        default="",
    )
    opt = parser.parse_args()
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print(
            "Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program"
        )
        exit()
    main(opt)
