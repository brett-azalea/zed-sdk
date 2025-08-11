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
Detect bags and draw 2D bounding boxes around them
Inspired by zed_camera.py detection approach but minimal and standalone
"""
import sys
import pyzed.sl as sl
import argparse
import cv2
import numpy as np
import time


class DetectionModels:
    """Detection model options inspired by zed_camera.py"""

    MULTI_CLASS_BOX_ACCURATE = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE
    MULTI_CLASS_BOX_FAST = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
    MULTI_CLASS_BOX_MEDIUM = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM


# Detection configuration
DEFAULT_DETECTION_MODEL = DetectionModels.MULTI_CLASS_BOX_MEDIUM
CONFIDENCE_THRESHOLD = 40


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


def draw_detections(image, objects):
    """
    Draw 2D bounding boxes and labels on the image
    Enhanced version inspired by zed_camera.py bbox handling
    """
    detection_count = 0

    for obj in objects.object_list:
        # Robust bbox extraction inspired by zed_camera.py
        bbox_2d = (
            np.array(obj.bounding_box_2d, dtype=np.float32)
            if hasattr(obj, "bounding_box_2d") and obj.bounding_box_2d
            else np.zeros((4, 2), dtype=np.float32)
        )

        # Validate bbox shape and coordinates
        if bbox_2d.shape == (4, 2) and np.any(bbox_2d != 0):
            x_min, y_min = int(bbox_2d[0][0]), int(bbox_2d[0][1])
            x_max, y_max = int(bbox_2d[2][0]), int(bbox_2d[2][1])

            # Ensure valid bbox coordinates
            if x_max > x_min and y_max > y_min:
                # Draw bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Get label and confidence with defaults
                label = obj.label if hasattr(obj, "label") else "BAG"
                confidence = obj.confidence if hasattr(obj, "confidence") else 0.0

                # Draw label with confidence
                label_text = f"{label} ({confidence:.2f})"
                cv2.putText(
                    image,
                    label_text,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Draw center point
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)

                detection_count += 1

                # Debug info for first few detections
                if detection_count <= 3:
                    print(
                        f"Detection {detection_count}: {label} at ({x_min},{y_min})-({x_max},{y_max}), conf: {confidence:.2f}"
                    )

    return image, detection_count


def setup_object_detection(zed, detection_model=DEFAULT_DETECTION_MODEL):
    """
    Setup object detection with parameters inspired by zed_camera.py
    """
    print("Setting up object detection...")

    # Object detection parameters inspired by zed_camera.py
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = detection_model
    obj_param.enable_tracking = True  # Enable for better temporal consistency
    obj_param.enable_segmentation = True  # Enable for better detection quality

    # Enable positional tracking for object tracking
    tracking_params = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Warning: Positional tracking failed: {err}")

    # Enable object detection
    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Object Detection Enable Error: {err}")
        return False

    print("Object detection enabled successfully!")
    return True


def main(opt):
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # For ZED Box, use ULTRA depth mode instead of NONE to avoid GMSL issues
    # We'll just not use the depth data but keep the pipeline working
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA

    # ZED Box specific settings
    init_params.camera_fps = 15  # Conservative FPS for stability
    init_params.sdk_verbose = 1  # Enable verbose logging for debugging

    # Increase timeout for GMSL cameras
    init_params.open_timeout_sec = 60

    parse_args(init_params, opt)

    print("Attempting to open ZED camera...")
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open Error: {err}")
        print("Troubleshooting tips:")
        print("1. Make sure no other processes are using the camera")
        print("2. Try running: sudo systemctl restart nvargus-daemon")
        print("3. Check camera connection")
        print("4. Try rebooting the ZED Box")
        exit(1)

    print("Camera opened successfully!")

    # Wait a bit for camera to stabilize
    time.sleep(2)

    # Setup object detection with enhanced parameters
    if not setup_object_detection(zed, DEFAULT_DETECTION_MODEL):
        zed.close()
        exit(1)

    # Configure object detection runtime parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = CONFIDENCE_THRESHOLD
    obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.BAG]  # Only detect Bags

    # Create ZED objects filled in the main loop
    objects = sl.Objects()
    image = sl.Mat()

    # Set runtime parameters
    runtime_parameters = sl.RuntimeParameters()

    print("Starting enhanced bag detection. Press 'q' to quit.")
    print(f"Using detection model: {DEFAULT_DETECTION_MODEL}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    frame_count = 0
    total_detections = 0

    while True:
        # Grab an image
        grab_status = zed.grab(runtime_parameters)
        if grab_status == sl.ERROR_CODE.SUCCESS:
            frame_count += 1

            # Retrieve left image for display
            zed.retrieve_image(image, sl.VIEW.LEFT)

            # Retrieve objects
            zed.retrieve_objects(objects, obj_runtime_param)

            # Convert ZED image to OpenCV format
            image_cv = image.get_data()

            # Convert BGRA to BGR for OpenCV
            if image_cv.shape[2] == 4:  # BGRA
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2BGR)

            # Draw detections on the image with enhanced detection handling
            image_with_detections, detection_count = draw_detections(
                image_cv.copy(), objects
            )
            total_detections += detection_count

            # Display enhanced detection info
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
            model_text = (
                f"Model: {DEFAULT_DETECTION_MODEL} | Threshold: {CONFIDENCE_THRESHOLD}"
            )
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
            cv2.imshow("Enhanced ZED Bag Detection", image_with_detections)

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

    # Disable modules and close camera
    print("Shutting down...")
    zed.disable_object_detection()
    zed.disable_positional_tracking()
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
