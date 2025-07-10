import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pyservicemaker import Pipeline, BatchMetadataOperator, Probe, osd
import time
import json

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
                # Extract bounding box and class_id
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
        # Publish as JSON string for efficiency
        msg = String()
        msg.data = json.dumps(dets)
        self.publisher.publish(msg)

CONFIG_FILE_PATH = "/root/DeepStream-Yolo/config_infer_primary_yolo11.txt"
uri = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4"

class DeepStreamNode(Node):
    def __init__(self):
        super().__init__('deepstream_detector')
        self.pipeline = Pipeline("sample-pipeline")
        self.pipeline.add("nvurisrcbin", "src", {"uri": uri})
        self.pipeline.add("nvstreammux", "mux", {
            "batch-size": 1,
            "width": 1280,
            "height": 720,
            "live-source": False
        })
        self.pipeline.add("nvinferbin", "infer", {"config-file-path": CONFIG_FILE_PATH})
        self.pipeline.add("nvosdbin", "osd").add("nveglglessink", "sink", {
            "async": False,
            "sync": False,
            "qos": False
        })
        self.pipeline.link("src", "mux", "infer", "osd", "sink")
        self.pipeline.attach("infer", Probe("det_pub", DetectionPublisher(self)))
        # Start pipeline in a non-blocking way
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