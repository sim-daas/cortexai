from pyservicemaker import Pipeline, BatchMetadataOperator, Probe
import time
import argparse
import sys
import os

# Pose skeleton connections for drawing (COCO format)
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

class PoseDetector(BatchMetadataOperator):
    def __init__(self, streammux_width=1920, streammux_height=1080):
        super().__init__()
        self.frame_count = 0
        self.streammux_width = streammux_width
        self.streammux_height = streammux_height
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
    
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            self.frame_count += 1
            self.fps_frame_count += 1
            
            # Calculate and print FPS every 5 seconds
            current_time = time.time()
            if current_time - self.fps_start_time >= 5.0:
                fps = self.fps_frame_count / (current_time - self.fps_start_time)
                print(f"FPS: {fps:.2f}")
                self.fps_start_time = current_time
                self.fps_frame_count = 0
            
            # Process detected objects (persons)
            object_count = 0
            for object_meta in frame_meta.object_items:
                object_count += 1
                self.process_pose_detection(object_meta)
            
            if object_count > 0:
                print(f"Frame {self.frame_count}: Detected {object_count} person(s)")
    
    def process_pose_detection(self, object_meta):
        """Process pose detection for each detected person"""
        try:
            # Access basic properties that work
            class_id = object_meta.class_id
            confidence = object_meta.confidence
            object_id = object_meta.object_id
            
            # Print detection info - this is what we can reliably access
            print(f"  Person ID: {object_id}, Class: {class_id}, Confidence: {confidence:.2f}")
            
            # Note: Due to Python binding limitations in pyservicemaker:
            # - mask_params access fails with binding errors
            # - rect_params access fails with binding errors  
            # - OSD drawing functions have attribute errors
            # 
            # The actual pose keypoints would be in mask_params but cannot be accessed
            # The bounding box coordinates would be in rect_params but cannot be accessed
            # Custom drawing would require working OSD bindings
            
            # For a working pose detector, you would need to either:
            # 1. Use the original pyds-based pose detector
            # 2. Fix the pyservicemaker Python bindings
            # 3. Access pose data through tensor outputs (if available)
            
        except Exception as e:
            print(f"Error processing pose detection: {e}")

def main():
    parser = argparse.ArgumentParser(description='DeepStream Pose Detector using pyservicemaker')
    parser.add_argument('-s', '--source', 
                       default="file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4",
                       help='Source stream/file')
    parser.add_argument('-c', '--config-infer', 
                       default="/root/DeepStream-Yolo-Pose/config_infer_primary_yoloV8_pose.txt",
                       help='Config infer file')
    parser.add_argument('-w', '--width', type=int, default=1920, help='Stream width')
    parser.add_argument('-e', '--height', type=int, default=1080, help='Stream height')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-g', '--gpu-id', type=int, default=0, help='GPU ID')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.config_infer):
        print(f"Error: Config file not found: {args.config_infer}")
        sys.exit(1)
    
    print("=== DeepStream Pose Detector (pyservicemaker) ===")
    print(f"Source: {args.source}")
    print(f"Config: {args.config_infer}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU ID: {args.gpu_id}")
    print("=================================================")
    
    # Create pipeline
    pipeline = Pipeline("pose-detection-pipeline")
    
    # Add pipeline elements
    pipeline.add("nvurisrcbin", "src", {"uri": args.source})
    
    pipeline.add("nvstreammux", "mux", {
        "batch-size": args.batch_size,
        "width": args.width,
        "height": args.height,
        "batched-push-timeout": 25000,
        "enable-padding": False,
        "live-source": 1 if "rtsp://" in args.source else 0,
        "attach-sys-ts": True
    })
    
    pipeline.add("nvinfer", "pgie", {
        "config-file-path": args.config_infer,
        "qos": False
    })
    
    pipeline.add("nvtracker", "tracker", {
        "tracker-width": 640,
        "tracker-height": 384,
        "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "ll-config-file": "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
        "display-tracking-id": True,
        "qos": False
    })
    
    pipeline.add("nvvideoconvert", "nvvidconv", {
        "qos": False
    })
    
    pipeline.add("nvdsosd", "osd", {
        "process-mode": 1,  # GPU mode
        "qos": False
    })
    
    # Add sink based on platform
    try:
        import platform
        if platform.uname()[4] == 'aarch64':  # Jetson
            pipeline.add("nv3dsink", "sink", {
                "async": False,
                "sync": False,
                "qos": False
            })
        else:  # x86
            pipeline.add("nveglglessink", "sink", {
                "async": False,
                "sync": False,
                "qos": False
            })
    except:
        pipeline.add("nveglglessink", "sink", {
            "async": False,
            "sync": False,
            "qos": False
        })
    
    # Link pipeline elements
    pipeline.link(("src", "mux"), ("", "sink_%u"))
    pipeline.link("mux", "pgie", "tracker", "nvvidconv", "osd", "sink")
    
    # Attach pose detector to tracker output
    pose_detector = PoseDetector(args.width, args.height)
    pipeline.attach("tracker", Probe("pose-detector", pose_detector))
    
    print("Starting pipeline...")
    print("Note: This demonstrates the pyservicemaker API structure.")
    print("Due to Python binding limitations:")
    print("- Pose keypoints cannot be accessed from mask_params")
    print("- Bounding box coordinates cannot be accessed from rect_params") 
    print("- Custom OSD drawing functions have attribute errors")
    print("Only basic object detection info (class_id, confidence, object_id) works.")
    print("For full pose visualization, use the original pyds-based pose detector.")
    print("")
    
    try:
        pipeline.start().wait()
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        print("Pipeline stopped.")

if __name__ == '__main__':
    main()