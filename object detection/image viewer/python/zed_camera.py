import pyzed.sl as sl
from typing import List
import numpy as np
import pathlib
import yaml
from azaleacontrol.utils.custom_types import RGBDFrame, Detection
from azaleacontrol.perception.pick.detector import YOLODetector
import cv2
import time
import uuid
from pathlib import Path
from azaleacontrol.perception.common.utils import *


AZALEA_PARENT_DIR = pathlib.Path(__file__).parent.parent.parent.parent.parent


MAX_DEPTH_M = 100.0
STREAM_IP = "192.168.254.10"
CONFIDENCE_THRESHOLD = 35
CAMERA_TIMEOUT_SEC = 60
YOLO_DETECTOR_IMAGE_SIZE = 1280
EXTRINSICS_CALIBRATION_FILE = (
    AZALEA_PARENT_DIR
    / "python"
    / "azaleacontrol"
    / "perception"
    / "common"
    / "config"
    / "camera_data.yaml"
)


class DetectionModels:
    MULTI_CLASS_BOX_ACCURATE = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE
    MULTI_CLASS_BOX_FAST = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_FAST
    MULTI_CLASS_BOX_MEDIUM = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
    YOLO = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS


class ZedCamera:
    def __init__(
        self,
        target_serial_num: int,
        detection_model: DetectionModels = DetectionModels.YOLO,
        enable_visualization: bool = False,
        record: bool = False,
    ):
        self.camera_data = self.fetch_camera_data()
        self.extrinsics = self.camera_data.get("extrinsics", None)
        self.stream_port = self.camera_data.get("port", None)
        self.init_params = sl.InitParameters()
        self.serial_num = target_serial_num
        self.stream_ip = STREAM_IP
        self.detection_model_2d = detection_model
        if detection_model == DetectionModels.YOLO:
            path = (
                AZALEA_PARENT_DIR
                / "python"
                / "azaleacontrol"
                / "perception"
                / "common"
                / "models"
                / "yolov8m-seg.pt"
            )
            self.custom_detector = YOLODetector(
                path, YOLO_DETECTOR_IMAGE_SIZE, CONFIDENCE_THRESHOLD
            )
        self.enable_visualization = enable_visualization
        self.point_cloud_mat = sl.Mat()
        self.belt_mask_path = (
            Path(__file__).resolve().parent
            / "belt_masks"
            / f"belt_mask_{target_serial_num}.png"
        )
        self.record = record
        if self.record:
            timestamp = time.strftime("%H%M%d%m%y")
            self.record_path = (
                Path(__file__).resolve().parent
                / "recordings"
                / f"{self.serial_num}_{timestamp}.svo"
            )
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.record_path = None

    def close_camera(self):
        if self.record:
            self.zed.disable_recording()
        self.zed.close()
        print(f"Camera with serial ID {self.serial_num} closed")

    def fetch_camera_data(self):
        with open(EXTRINSICS_CALIBRATION_FILE, "r") as f:
            extrinsics_data = yaml.safe_load(f)
        return extrinsics_data["extrinsics"].get(self.serial_num, None)

    def print_camera_information(self, cam):
        cam_info = cam.get_camera_information()
        print("ZED Model                 : {0}".format(cam_info.camera_model))
        print("ZED Serial Number         : {0}".format(cam_info.serial_number))
        print(
            "ZED Camera Firmware       : {0}/{1}".format(
                cam_info.camera_configuration.firmware_version,
                cam_info.sensors_configuration.firmware_version,
            )
        )
        print(
            "ZED Camera Resolution     : {0}x{1}".format(
                round(cam_info.camera_configuration.resolution.width, 2),
                cam.get_camera_information().camera_configuration.resolution.height,
            )
        )
        print(
            "ZED Camera FPS            : {0}".format(
                int(cam_info.camera_configuration.fps)
            )
        )

    def find_and_open_camera_stream(self, enable_detection: bool) -> bool:
        # Retrofit to use stream IP and port number
        if self.stream_port is None:
            print(
                f"Stream port not found for camera with serial number {self.serial_num}"
            )
            return False
        print(f"Connecting to stream at {self.stream_ip}:{self.stream_port}")
        self.init_params.set_from_stream(self.stream_ip, self.stream_port)

        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.open_timeout_sec = CAMERA_TIMEOUT_SEC

        self.zed = sl.Camera()

        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Could not connect to stream at {self.stream_ip}:{self.stream_port}")
            return False

        self.print_camera_information(self.zed)

        time.sleep(1)

        if self.record:
            self.record_params = sl.RecordingParameters()
            self.record_params.compression_mode = sl.SVO_COMPRESSION_MODE.H264
            self.record_params.video_filename = str(self.record_path)
            err = self.zed.enable_recording(self.record_params)
            print(f"Recording to {self.record_path}")
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"Error enabling recording: {err}")
                return False

        camera_info = self.zed.get_camera_information()
        cam_config = camera_info.camera_configuration
        calibration_parameters = cam_config.calibration_parameters
        left_cam = calibration_parameters.left_cam
        self.intrinsics = {
            "fx": left_cam.fx,
            "fy": left_cam.fy,
            "cx": left_cam.cx,
            "cy": left_cam.cy,
        }

        if self.extrinsics == None:
            print(
                f"No extrinsics found for camera with serial number {self.serial_num}"
            )
            return False

        if enable_detection:
            detection_params = sl.ObjectDetectionParameters()
            detection_params.detection_model = self.detection_model_2d
            detection_params.enable_tracking = True
            detection_params.enable_segmentation = True
            tracking_params = sl.PositionalTrackingParameters()

            self.zed.enable_positional_tracking(tracking_params)
            err = self.zed.enable_object_detection(detection_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"Error enabling object detection: {err}")
            if self.belt_mask_path.exists():
                self.belt_mask_grayscale = cv2.imread(
                    str(self.belt_mask_path), cv2.IMREAD_GRAYSCALE
                )
                self.belt_mask = (self.belt_mask_grayscale > 0).astype(np.uint8)
                print(f"Successfully loaded belt mask from {self.belt_mask_path}")
            else:
                self.belt_mask = None
                print(f"Failed to load belt mask from {self.belt_mask_path}")
        return True

    def obj_mask_to_global_mask(
        self, object_mask: np.ndarray, bbox_2d: np.ndarray, image_shape: tuple
    ) -> np.ndarray:
        H, W = image_shape[0], image_shape[1]

        # get integer bbox coords
        x_min, y_min = int(bbox_2d[0, 0]), int(bbox_2d[0, 1])
        x_max, y_max = int(bbox_2d[2, 0]), int(bbox_2d[2, 1])
        w_box, h_box = x_max - x_min, y_max - y_min

        # resize cropped mask to bbox size
        if (object_mask.shape[1], object_mask.shape[0]) != (w_box, h_box):
            object_mask = cv2.resize(
                object_mask, (w_box, h_box), interpolation=cv2.INTER_NEAREST
            )

        # create full-frame blank mask and paste into the right location
        full_mask = np.zeros((H, W), dtype=np.uint8)
        y1, y2 = max(0, y_min), min(H, y_max)
        x1, x2 = max(0, x_min), min(W, x_max)
        crop_y1 = y1 - y_min if y1 > y_min else 0
        crop_x1 = x1 - x_min if x1 > x_min else 0
        crop_y2 = crop_y1 + (y2 - y1)
        crop_x2 = crop_x1 + (x2 - x1)

        full_mask[y1:y2, x1:x2] = object_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        full_mask = (full_mask // 255).astype(np.uint8)

        return full_mask

    def fetch_pointcloud(self) -> np.ndarray:

        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_measure(self.point_cloud_mat, sl.MEASURE.XYZ)
            points_raw = self.point_cloud_mat.get_data()
            return filter_raw_points(points_raw)
        return None

    def fetch_raw_images(
        self,
    ) -> tuple[
        bool,
        sl.Mat,
        sl.Mat,
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        image_grabbed = False
        if self.detection_model_2d == DetectionModels.YOLO:
            detection_runtime_params = sl.CustomObjectDetectionRuntimeParameters()
        else:
            detection_runtime_params = sl.ObjectDetectionRuntimeParameters(
                detection_confidence_threshold=CONFIDENCE_THRESHOLD,
                object_class_filter=[sl.OBJECT_CLASS.BAG],
            )
        objects = sl.Objects()
        best_bbox_2d = np.zeros((4, 2), dtype=np.float32)
        best_bbox_3d = np.zeros((8, 3), dtype=np.float32)
        best_object_position = np.zeros((3,), dtype=np.float32)
        best_object_velocity = np.zeros((3,), dtype=np.float32)
        best_object_mask = np.zeros((1, 1), dtype=np.uint8)
        img_mat, depth_mat = sl.Mat(), sl.Mat()
        rgb_timestamp = 0
        depth_timestamp = 0
        detection_success = False
        while not image_grabbed:
            runtime = sl.RuntimeParameters()
            if self.zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                print("Error grabbing image from ZED camera")
                continue
            image_grabbed = True
            img_mat, depth_mat = sl.Mat(), sl.Mat()
            self.zed.retrieve_image(img_mat, sl.VIEW.LEFT)
            rgb_timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns

            self.zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth_timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns

            img = img_mat.get_data()

            if self.detection_model_2d == DetectionModels.YOLO:
                detections = self.custom_detector.run_detection(img)
                self.zed.ingest_custom_mask_objects(detections)
                self.zed.retrieve_custom_objects(objects, detection_runtime_params)
            else:
                self.zed.retrieve_objects(objects, detection_runtime_params)

            max_intersection_pct = 0.4
            best_object_mask = np.zeros((1, 1), dtype=np.uint8)
            for obj in objects.object_list:
                bbox_2d = (
                    np.array(obj.bounding_box_2d, dtype=np.float32)
                    if hasattr(obj, "bounding_box_2d")
                    else np.zeros((4, 2), dtype=np.float32)
                )
                bbox_3d = (
                    np.array(obj.bounding_box, dtype=np.float32)
                    if hasattr(obj, "bounding_box")
                    else np.zeros((8, 3), dtype=np.float32)
                )
                object_position = (
                    np.array(obj.position, dtype=np.float32)
                    if hasattr(obj, "position")
                    else np.zeros((3,), dtype=np.float32)
                )
                object_velocity = (
                    np.array(obj.velocity, dtype=np.float32)
                    if hasattr(obj, "velocity")
                    else np.zeros((3,), dtype=np.float32)
                )
                if (
                    hasattr(obj, "mask")
                    and hasattr(obj.mask, "get_data")
                    and obj.mask.get_data() is not None
                    and obj.mask.get_data().shape[0] > 0
                    and obj.mask.get_data().shape[1] > 0
                ):
                    object_mask = np.array(obj.mask.get_data(), dtype=np.uint8)
                    global_mask = self.obj_mask_to_global_mask(
                        object_mask, bbox_2d, (img.shape[0], img.shape[1])
                    )
                    intersection_mask = global_mask & self.belt_mask

                    intersection_area = int(np.count_nonzero(intersection_mask))
                    object_area = int(np.count_nonzero(object_mask))
                    if object_area > 0:
                        intersection_pct = intersection_area / object_area

                        if intersection_pct > max_intersection_pct:
                            max_intersection_pct = intersection_pct
                            best_object_mask = object_mask
                            best_bbox_2d = bbox_2d
                            best_bbox_3d = bbox_3d
                            best_object_position = object_position
                            best_object_velocity = object_velocity
                            detection_success = True

                if self.enable_visualization and bbox_2d.shape == (4, 2):
                    x_min, y_min = int(bbox_2d[0][0]), int(bbox_2d[0][1])
                    x_max, y_max = int(bbox_2d[2][0]), int(bbox_2d[2][1])
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"{obj.label} ({obj.confidence:.2f})",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
        return (
            detection_success,
            img_mat,
            depth_mat,
            rgb_timestamp,
            depth_timestamp,
            best_bbox_2d,
            best_bbox_3d,
            best_object_position,
            best_object_velocity,
            best_object_mask,
        )

    def fetch_rgbd_image_with_detection(
        self,
        success,
        img_mat,
        depth_mat,
        rgb_timestamp,
        depth_timestamp,
        bbox_2d,
        bbox_3d,
        object_position,
        object_velocity,
        object_mask,
    ) -> tuple[RGBDFrame, Detection]:
        valid_depth = 0
        color_image = np.zeros((1, 1, 3), dtype=np.uint8)
        depth_data = np.zeros((1, 1), dtype=np.float32)
        while valid_depth == 0:
            h, w = img_mat.get_height(), img_mat.get_width()
            raw = img_mat.get_data()
            bgra = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
            color_image = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            depth_data = np.array(depth_mat.get_data(), copy=True)
            invalid = (
                (~np.isfinite(depth_data))
                | (depth_data <= 0)
                | (depth_data > MAX_DEPTH_M)
            )
            depth_data[invalid] = np.nan
            valid_depth = np.count_nonzero(~np.isnan(depth_data))

        frame_id = uuid.uuid4().hex
        rgbd_image = RGBDFrame(
            rgb_image=color_image,
            depth_image=depth_data,
            frame_id=frame_id,
            frame_timestamp=time.time_ns(),
            rgb_timestamp=rgb_timestamp,
            depth_timestamp=depth_timestamp,
            format="jpg",
        )

        detection = Detection(
            success=success,
            rgb_image=color_image,
            frame_id=frame_id,
            rgb_image_timestamp=rgb_timestamp,
            format="jpg",
            bbox_2d=bbox_2d,
            bbox_3d=bbox_3d,
            object_position=object_position,
            object_velocity=object_velocity,
            object_mask=object_mask,
        )

        return rgbd_image, detection

    def fetch_rgbd_image_and_detections(self) -> tuple[RGBDFrame, Detection]:
        (
            success,
            img_mat,
            depth_mat,
            rgb_timestamp,
            depth_timestamp,
            bbox_2d,
            bbox_3d,
            object_position,
            object_velocity,
            object_mask,
        ) = self.fetch_raw_images()
        rgbd_image, detection = self.fetch_rgbd_image_with_detection(
            success,
            img_mat,
            depth_mat,
            rgb_timestamp,
            depth_timestamp,
            bbox_2d,
            bbox_3d,
            object_position,
            object_velocity,
            object_mask,
        )

        return rgbd_image, detection


def main():
    serial_num = 47093072
    port = 30004
    zed_camera = ZedCamera(
        target_serial_num=serial_num,
        stream_port=port,
        enable_visualization=True,
        record=False,
    )
    if not zed_camera.find_and_open_camera_stream(enable_detection=True):
        print("Failed to open camera stream.")
        return

    while True:
        rgbd_frame, detection = zed_camera.fetch_rgbd_image_and_detections()

        object_mask = detection.object_mask
        if object_mask is not None:
            # Visualize rgb, depth, and mask overlay
            if zed_camera.enable_visualization:
                object_mask = detection.object_mask
                bbox_2d = detection.bbox_2d
                rgb_img = detection.rgb_image
                img_shape = (rgb_img.shape[0], rgb_img.shape[1])
                if object_mask.shape == (1, 1):
                    print("No object detected")
                    continue
                full_mask = zed_camera.obj_mask_to_global_mask(
                    object_mask, bbox_2d, img_shape
                )
                colored_mask = colorize_mask(full_mask)
                rgb_with_mask = rgbd_frame.rgb_image.copy()
                alpha, beta, gamma = 0.5, 1.0, 0.0
                overlay_image = cv2.addWeighted(
                    rgb_with_mask, beta, colored_mask, alpha, gamma
                )
                cv2.imshow("RGB Image", rgbd_frame.rgb_image)
                cv2.imshow("Depth Image", rgbd_frame.depth_image)
                cv2.imshow("Mask", colored_mask)
                cv2.imshow("RGB with Mask Overlay", overlay_image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break


if __name__ == "__main__":
    main()
