import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pyservicemaker import Pipeline, BatchMetadataOperator, Probe, osd
import json
import time

class TrackerIdPublisher(BatchMetadataOperator):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.publisher = ros_node.create_publisher(String, '/det', 10)
        self.fps_start_time = time.time()
        self.frame_count = 0

    def handle_metadata(self, batch_meta):
        dets = []
        for frame_meta in batch_meta.frame_items:
            # FPS calculation
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            if elapsed_time >= 1.0:
                fps = self.frame_count / elapsed_time
                print(f"Average FPS: {fps:.2f}")
                self.frame_count = 0
                self.fps_start_time = current_time

            for object_meta in frame_meta.object_items:
                tracker_id = getattr(object_meta, "object_id", None)
                left = getattr(object_meta, "rect_params", None)
                if left is not None:
                    x = max(0, int(getattr(left, "left", 0)))
                    y = max(0, int(getattr(left, "top", 0)))
                    w = float(getattr(left, "width", 0.0))
                    h = float(getattr(left, "height", 0.0))
                else:
                    x = 0
                    y = 0
                    w = 0.0
                    h = 0.0
                if tracker_id is not None:
                    dets.append({
                        "tracker_id": int(tracker_id),
                        "left": float(x),
                        "top": float(y),
                        "width": float(w),
                        "height": float(h),
                        "class_id": int(getattr(object_meta, "class_id", -1)),
                        "confidence": float(getattr(object_meta, "confidence", 0.0))
                    })
        msg = String()
        msg.data = json.dumps(dets)
        self.publisher.publish(msg)

CONFIG_FILE_PATH = "/root/DeepStream-Yolo/config_infer_primary_yolo11.txt"

class DeepStreamTrackerNode(Node):
    def __init__(self):
        super().__init__('deepstream_cam_tracker')
        self.pipeline = Pipeline("sample-pipeline")
        self.pipeline.add("v4l2src", "src" , {"device": "/dev/video0"})
        self.pipeline.add("capsfilter", "src_caps", {"caps": "video/x-raw, framerate=30/1"})
        self.pipeline.add("videoconvert", "convert")
        self.pipeline.add("nvvideoconvert", "nvconvert")
        self.pipeline.add("capsfilter", "nvconvert_caps", {"caps": "video/x-raw(memory:NVMM)"})
        self.pipeline.add("nvstreammux", "mux", {"batch-size": 1, "width": 1280, "height": 720})
        self.pipeline.add("nvinferbin", "infer", {"config-file-path": CONFIG_FILE_PATH})
        self.pipeline.add("nvtracker", "tracker", {
            "tracker-width": 640,
            "tracker-height": 384,
            "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
            "ll-config-file": "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
        })
        self.pipeline.add("nvosdbin", "osd").add("nveglglessink", "sink", {"sync": False})
        self.pipeline.link("src", "src_caps","convert", "nvconvert", "nvconvert_caps", "mux", "infer","tracker", "osd", "sink")
        self.pipeline.attach("tracker", Probe("tracker_id_pub", TrackerIdPublisher(self)))
        self.pipeline.start()

def main(args=None):
    rclpy.init(args=args)
    node = DeepStreamTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.pipeline.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()