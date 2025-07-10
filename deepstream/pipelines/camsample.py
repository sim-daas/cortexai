from pyservicemaker import Pipeline, BatchMetadataOperator, Probe, osd
import time

class ObjectCounterMarker(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            vehcle_count = 0
            person_count = 0
            for object_meta in frame_meta.object_items:
                class_id = object_meta.class_id
                if class_id == 0:
                    vehcle_count += 1
                elif class_id == 2:
                    person_count += 1
            print(f"Object Counter: Pad Idx={frame_meta.pad_index},"
                f"Frame Number={frame_meta.frame_number},"
                f"Vehicle Count={vehcle_count}, Person Count={person_count}")
            text = f"Person={person_count},Vehicle={vehcle_count}"
            display_meta = batch_meta.acquire_display_meta()
            label = osd.Text()
            label.display_text = text.encode('ascii')
            label.x_offset = 10
            label.y_offset = 12
            label.font.name = osd.FontFamily.Serif
            label.font.size = 12
            label.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
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
    pipeline.add("nvosdbin", "osd").add("nveglglessink", "sink", {"sync": False})
    pipeline.link("src", "src_caps","convert", "nvconvert", "nvconvert_caps", "mux", "infer", "osd", "sink")
    pipeline.attach("infer", Probe("counter", ObjectCounterMarker()))
    pipeline.start().wait()