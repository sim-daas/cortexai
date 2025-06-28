from pyservicemaker import Pipeline, Flow

pipeline = Pipeline("detector")
infer_config = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"
video_file = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
Flow(pipeline).batch_capture([video_file]).infer(infer_config).render()()
