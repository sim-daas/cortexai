from pyservicemaker import Pipeline
import sys

CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/nvinfer/config_infer_primary_trafficcamnet.yml"

if __name__ == '__main__':
    pipeline = Pipeline("sample-pipeline")
    pipeline.add("nvurisrcbin", "src", {"uri": sys.argv[1]})
    pipeline.add("nvstreammux", "mux", {"batch-size": 1, "width": 1280, "height": 720})
    pipeline.add("nvinferbin", "infer", {"config-file-path": CONFIG_FILE_PATH})
    pipeline.add("nvosdbin", "osd").add("nveglglessink", "sink")
    pipeline.link(("src", "mux"), ("", "sink_%u")).link("mux", "infer", "osd", "sink")
    pipeline.start().wait()
