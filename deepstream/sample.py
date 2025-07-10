from pyservicemaker import Pipeline, BatchMetadataOperator, Probe, osd
import time

class ObjectCounterMarker(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        self.fps_start_time = time.time()
        self.frame_count = 0

    def handle_metadata(self, batch_meta):
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

            vehcle_count = 0
            person_count = 0
            for object_meta in frame_meta.object_items:
                class_id = object_meta.class_id
                print(object_meta.class_id)
                if class_id == 0:
                    vehcle_count += 1
                elif class_id == 2:
                    person_count += 1
         #   print(f"Object Counter: Pad Idx={frame_meta.pad_index},"
          #      f"Frame Number={frame_meta.frame_number},"
           #     f"Vehicle Count={vehcle_count}, Person Count={person_count}")
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

CONFIG_FILE_PATH = "/root/DeepStream-Yolo/config_infer_yolov11x.txt"
uri = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4"

if __name__ == '__main__':
    pipeline = Pipeline("sample-pipeline")
    pipeline.add("nvurisrcbin", "src", {"uri": uri})
    pipeline.add("nvstreammux", "mux", {
        "batch-size": 1, 
        "width": 1280, 
        "height": 720,
        "live-source": False
    })
    pipeline.add("nvinferbin", "infer", {"config-file-path": CONFIG_FILE_PATH})
    pipeline.add("nvosdbin", "osd").add("nveglglessink", "sink", {
        "async": False,
        "sync": False,
        "qos": False
    })
   # pipeline.link(("src", "mux"), ("", "sink_%u")).link("mux", "infer", "osd", "sink")
    pipeline.link("src", "mux", "infer", "osd", "sink")
    pipeline.attach("infer", Probe("counter", ObjectCounterMarker()))
    pipeline.start().wait()