from pyservicemaker import Pipeline, BatchMetadataOperator, Probe, osd
import time

class TrackerIdVisualizer(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            for object_meta in frame_meta.object_items:
                tracker_id = getattr(object_meta, "object_id", None)
                left = getattr(object_meta, "rect_params", None)
                # Ensure x/y offsets are non-negative integers
                if left is not None:
                    x = max(0, int(getattr(left, "left", 0)))
                    y = max(0, int(getattr(left, "top", 0)) - 10)
                else:
                    x = 0
                    y = 0
                if tracker_id is not None:
                    display_meta = batch_meta.acquire_display_meta()
                    label = osd.Text()
                    label.display_text = f"ID: {tracker_id}".encode('ascii')
                    label.x_offset = x
                    label.y_offset = y
                    label.font.name = osd.FontFamily.Serif
                    label.font.size = 14
                    label.font.color = osd.Color(1.0, 1.0, 0.0, 1.0)
                    label.set_bg_color = True
                    label.bg_color = osd.Color(0.0, 0.0, 0.0, 1.0)
                    display_meta.add_text(label)
                    frame_meta.append(display_meta)

CONFIG_FILE_PATH = "/root/DeepStream-Yolo/config_infer_primary_yolo11.txt"

if __name__ == '__main__':
    pipeline = Pipeline("sample-pipeline")
    pipeline.add("v4l2src", "src" , {"device": "/dev/video0"})
    pipeline.add("capsfilter", "src_caps", {"caps": "video/x-raw, framerate=30/1"})
    pipeline.add("videoconvert", "convert")
    pipeline.add("nvvideoconvert", "nvconvert")
    pipeline.add("capsfilter", "nvconvert_caps", {"caps": "video/x-raw(memory:NVMM)"})
    pipeline.add("nvstreammux", "mux", {"batch-size": 1, "width": 1280, "height": 720})
    pipeline.add("nvinferbin", "infer", {"config-file-path": CONFIG_FILE_PATH})
    pipeline.add("nvtracker", "tracker", {
        "tracker-width": 640,
        "tracker-height": 384,
        "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "ll-config-file": "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
    })
    pipeline.add("nvosdbin", "osd").add("nveglglessink", "sink", {"sync": False})
    pipeline.link("src", "src_caps","convert", "nvconvert", "nvconvert_caps", "mux", "infer","tracker", "osd", "sink")
    pipeline.attach("tracker", Probe("tracker_id", TrackerIdVisualizer()))
    pipeline.start().wait()