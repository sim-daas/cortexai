import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pyservicemaker import Pipeline, BatchMetadataOperator, Probe, osd
import json
import time

class DetectionPublisher(BatchMetadataOperator):
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
                bbox = getattr(object_meta, "rect_params", None)
                if bbox is not None:
                    det = {
                        "class_id": int(getattr(object_meta, "class_id", -1)),
                        "left": float(getattr(bbox, "left", 0.0)),
                        "top": float(getattr(bbox, "top", 0.0)),
                        "width": float(getattr(bbox, "width", 0.0)),
                        "height": float(getattr(bbox, "height", 0.0)),
                        "confidence": float(getattr(object_meta, "confidence", 0.0))
                    }
                    dets.append(det)
        msg = String()
        msg.data = json.dumps(dets)
        self.publisher.publish(msg)

CONFIG_FILE_PATH = "/root/DeepStream-Yolo/config_infer_primary_yolo11.txt"

class DeepStreamNode(Node):
    def __init__(self):
        super().__init__('deepstream_cam_detector')
        self.pipeline = Pipeline("sample-pipeline")
        self.pipeline.add("v4l2src", "src" , {"device": "/dev/video0"})
        self.pipeline.add("capsfilter", "src_caps", {"caps": "video/x-raw, framerate=30/1"})
        self.pipeline.add("videoconvert", "convert")
        self.pipeline.add("nvvideoconvert", "nvconvert")
        self.pipeline.add("capsfilter", "nvconvert_caps", {"caps": "video/x-raw(memory:NVMM)"})
        self.pipeline.add("nvstreammux", "mux", {"batch-size": 1, "width": 1280, "height": 720})
        self.pipeline.add("nvinferbin", "infer", {"config-file-path": CONFIG_FILE_PATH})
        self.pipeline.add("nvosdbin", "osd").add("nveglglessink", "sink", {"sync": False})
        self.pipeline.link("src", "src_caps","convert", "nvconvert", "nvconvert_caps", "mux", "infer", "osd", "sink")
        self.pipeline.attach("infer", Probe("det_pub", DetectionPublisher(self)))
        self.pipeline.start()

def main(args=None):
    rclpy.init(args=args)
    node = DeepStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.pipeline.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()