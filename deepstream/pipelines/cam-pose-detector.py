import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import os
import sys
import time
import argparse
import platform
from ctypes import *

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds

MAX_ELEMENTS_IN_DISPLAY_META = 16

SOURCE = ''
CONFIG_INFER = ''
STREAMMUX_BATCH_SIZE = 1
STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080
GPU_ID = 0
PERF_MEASUREMENT_INTERVAL_SEC = 5

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

start_time = time.time()
fps_streams = {}


def set_custom_bbox(obj_meta):
    border_width = 6
    font_size = 18
    x_offset = int(min(STREAMMUX_WIDTH - 1, max(0, obj_meta.rect_params.left - (border_width / 2))))
    y_offset = int(min(STREAMMUX_HEIGHT - 1, max(0, obj_meta.rect_params.top - (font_size * 2) + 1)))

    obj_meta.rect_params.border_width = border_width
    obj_meta.rect_params.border_color.red = 0.0
    obj_meta.rect_params.border_color.green = 0.0
    obj_meta.rect_params.border_color.blue = 1.0
    obj_meta.rect_params.border_color.alpha = 1.0
    obj_meta.text_params.font_params.font_name = 'Ubuntu'
    obj_meta.text_params.font_params.font_size = font_size
    obj_meta.text_params.x_offset = x_offset
    obj_meta.text_params.y_offset = y_offset
    obj_meta.text_params.font_params.font_color.red = 1.0
    obj_meta.text_params.font_params.font_color.green = 1.0
    obj_meta.text_params.font_params.font_color.blue = 1.0
    obj_meta.text_params.font_params.font_color.alpha = 1.0
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.red = 0.0
    obj_meta.text_params.text_bg_clr.green = 0.0
    obj_meta.text_params.text_bg_clr.blue = 1.0
    obj_meta.text_params.text_bg_clr.alpha = 1.0


def parse_pose_from_meta(frame_meta, obj_meta):
    try:
        # Defensive: check mask_params exists and is valid
        if not hasattr(obj_meta, "mask_params"):
            return
        mask_params = obj_meta.mask_params
        if not hasattr(mask_params, "size") or mask_params.size == 0:
            return
        if not hasattr(mask_params, "get_mask_array"):
            return

        num_joints = int(mask_params.size / (sizeof(c_float) * 3))
        if num_joints == 0:
            return

        gain = min(mask_params.width / STREAMMUX_WIDTH, mask_params.height / STREAMMUX_HEIGHT)
        pad_x = (mask_params.width - STREAMMUX_WIDTH * gain) / 2.0
        pad_y = (mask_params.height - STREAMMUX_HEIGHT * gain) / 2.0

        batch_meta = frame_meta.base_meta.batch_meta
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        data = mask_params.get_mask_array()
        # ...no debug prints...

        for i in range(num_joints):
            xc = int((data[i * 3 + 0] - pad_x) / gain)
            yc = int((data[i * 3 + 1] - pad_y) / gain)
            confidence = data[i * 3 + 2]
            if confidence < 0.5 or not (0 <= xc < STREAMMUX_WIDTH and 0 <= yc < STREAMMUX_HEIGHT):
                continue

            if display_meta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            circle_params = display_meta.circle_params[display_meta.num_circles]
            circle_params.xc = xc
            circle_params.yc = yc
            circle_params.radius = 6
            circle_params.circle_color.red = 1.0
            circle_params.circle_color.green = 1.0
            circle_params.circle_color.blue = 1.0
            circle_params.circle_color.alpha = 1.0
            circle_params.has_bg_color = 1
            circle_params.bg_color.red = 0.0
            circle_params.bg_color.green = 0.0
            circle_params.bg_color.blue = 1.0
            circle_params.bg_color.alpha = 1.0
            display_meta.num_circles += 1

        for i in range(len(skeleton)):
            idx1 = skeleton[i][0] - 1
            idx2 = skeleton[i][1] - 1
            if idx1 >= num_joints or idx2 >= num_joints:
                continue
            x1 = int((data[idx1 * 3 + 0] - pad_x) / gain)
            y1 = int((data[idx1 * 3 + 1] - pad_y) / gain)
            confidence1 = data[idx1 * 3 + 2]
            x2 = int((data[idx2 * 3 + 0] - pad_x) / gain)
            y2 = int((data[idx2 * 3 + 1] - pad_y) / gain)
            confidence2 = data[idx2 * 3 + 2]
            if (confidence1 < 0.5 or confidence2 < 0.5 or
                not (0 <= x1 < STREAMMUX_WIDTH and 0 <= y1 < STREAMMUX_HEIGHT) or
                not (0 <= x2 < STREAMMUX_WIDTH and 0 <= y2 < STREAMMUX_HEIGHT)):
                continue

            if display_meta.num_lines == MAX_ELEMENTS_IN_DISPLAY_META:
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            line_params = display_meta.line_params[display_meta.num_lines]
            line_params.x1 = x1
            line_params.y1 = y1
            line_params.x2 = x2
            line_params.y2 = y2
            line_params.line_width = 6
            line_params.line_color.red = 0.0
            line_params.line_color.green = 0.0
            line_params.line_color.blue = 1.0
            line_params.line_color.alpha = 1.0
            display_meta.num_lines += 1

    except Exception as e:
        pass  # suppress all debug output


class GETFPS:
    def __init__(self, stream_id):
        global start_time
        self.start_time = start_time
        self.is_first = True
        self.frame_count = 0
        self.stream_id = stream_id
        self.total_fps_time = 0
        self.total_frame_count = 0

    def get_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        current_time = end_time - self.start_time
        if current_time > PERF_MEASUREMENT_INTERVAL_SEC:
            self.total_fps_time = self.total_fps_time + current_time
            self.total_frame_count = self.total_frame_count + self.frame_count
            current_fps = float(self.frame_count) / current_time
            avg_fps = float(self.total_frame_count) / self.total_fps_time
            print('FPS: %.2f (avg: %.2f)' % (current_fps, avg_fps))
            self.start_time = end_time
            self.frame_count = 0
        else:
            self.frame_count = self.frame_count + 1


def pgie_src_pad_buffer_probe(pad, info, user_data):
    buf = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        current_index = frame_meta.source_id

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            parse_pose_from_meta(frame_meta, obj_meta)
            set_custom_bbox(obj_meta)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Fix: Always provide a default GETFPS for the stream if missing
        stream_key = f'stream{current_index}'
        if stream_key not in fps_streams:
            fps_streams[stream_key] = GETFPS(current_index)
        fps_streams[stream_key].get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find('decodebin') != -1:
        Object.connect('child-added', decodebin_child_added, user_data)
    if name.find('nvv4l2decoder') != -1:
        Object.set_property('drop-frame-interval', 0)
        Object.set_property('num-extra-surfaces', 1)
        if is_aarch64():
            Object.set_property('enable-max-performance', 1)
        else:
            Object.set_property('cudadec-memtype', 0)
            Object.set_property('gpu-id', GPU_ID)


def cb_newpad(decodebin, pad, user_data):
    streammux_sink_pad = user_data
    caps = pad.get_current_caps()
    if not caps:
        caps = pad.query_caps()
    structure = caps.get_structure(0)
    name = structure.get_name()
    features = caps.get_features(0)
    if name.find('video') != -1:
        if features.contains('memory:NVMM'):
            if pad.link(streammux_sink_pad) != Gst.PadLinkReturn.OK:
                sys.stderr.write('ERROR: Failed to link source to streammux sink pad\n')
        else:
            sys.stderr.write('ERROR: decodebin did not pick NVIDIA decoder plugin')


def create_uridecode_bin(stream_id, uri, streammux):
    bin_name = 'source-bin-%04d' % stream_id
    bin = Gst.ElementFactory.make('uridecodebin', bin_name)
    if 'rtsp://' in uri:
        pyds.configure_source_for_ntp_sync(bin)
    bin.set_property('uri', uri)
    pad_name = 'sink_%u' % stream_id
    streammux_sink_pad = streammux.get_request_pad(pad_name)
    bin.connect('pad-added', cb_newpad, streammux_sink_pad)
    bin.connect('child-added', decodebin_child_added, 0)
    fps_streams['stream{0}'.format(stream_id)] = GETFPS(stream_id)
    return bin


def bus_call(bus, message, user_data):
    loop = user_data
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write('DEBUG: EOS\n')
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write('WARNING: %s: %s\n' % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write('ERROR: %s: %s\n' % (err, debug))
        loop.quit()
    return True


def is_aarch64():
    return platform.uname()[4] == 'aarch64'


def main():
    Gst.init(None)

    loop = GLib.MainLoop()

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write('ERROR: Failed to create pipeline\n')
        sys.exit(1)

    # --- Camera input elements before nvstreammux, matching camsample.py ---
    print("Creating camera source pipeline elements")
    source = Gst.ElementFactory.make("v4l2src", "src")
    if not source:
        sys.stderr.write(" Unable to create Source \n")
        sys.exit(1)

    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "src_caps")
    if not caps_v4l2src:
        sys.stderr.write(" Unable to create v4l2src capsfilter \n")
        sys.exit(1)

    vidconvsrc = Gst.ElementFactory.make("videoconvert", "convert")
    if not vidconvsrc:
        sys.stderr.write(" Unable to create videoconvert \n")
        sys.exit(1)

    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "nvconvert")
    if not nvvidconvsrc:
        sys.stderr.write(" Unable to create Nvvideoconvert \n")
        sys.exit(1)

    caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvconvert_caps")
    if not caps_vidconvsrc:
        sys.stderr.write(" Unable to create capsfilter \n")
        sys.exit(1)

    # Set camera device and caps (as in camsample.py)
    source.set_property('device', '/dev/video0')
    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))

    # --- End camera input elements ---

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make('nvstreammux', 'mux')
    if not streammux:
        sys.stderr.write('ERROR: Failed to create nvstreammux\n')
        sys.exit(1)

    pgie = Gst.ElementFactory.make('nvinfer', 'infer')
    if not pgie:
        sys.stderr.write('ERROR: Failed to create nvinfer\n')
        sys.exit(1)

    converter = Gst.ElementFactory.make('nvvideoconvert', 'nvvidconv')
    if not converter:
        sys.stderr.write('ERROR: Failed to create nvvideoconvert\n')
        sys.exit(1)

    osd = Gst.ElementFactory.make('nvdsosd', 'osd')
    if not osd:
        sys.stderr.write('ERROR: Failed to create nvdsosd\n')
        sys.exit(1)

    sink = Gst.ElementFactory.make('nveglglessink', 'sink')
    if not sink:
        sys.stderr.write('ERROR: Failed to create nveglglessink\n')
        sys.exit(1)
    sink.set_property('sync', False)

    # Add elements to pipeline (camera input first)
    pipeline.add(source)
    pipeline.add(caps_v4l2src)
    pipeline.add(vidconvsrc)
    pipeline.add(nvvidconvsrc)
    pipeline.add(caps_vidconvsrc)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(converter)
    pipeline.add(osd)
    pipeline.add(sink)

    # Link camera input elements
    source.link(caps_v4l2src)
    caps_v4l2src.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_vidconvsrc)

    # Link camera pipeline to nvstreammux
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
        sys.exit(1)
    srcpad = caps_vidconvsrc.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
        sys.exit(1)
    srcpad.link(sinkpad)

    # ...existing code for linking downstream elements...
    streammux.set_property('batch-size', 1)
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('live-source', True)
    streammux.set_property('batched-push-timeout', 25000)
    streammux.set_property('enable-padding', 0)
    streammux.set_property('attach-sys-ts', 1)

    pgie.set_property('config-file-path', CONFIG_INFER)
    pgie.set_property('qos', 0)
    osd.set_property('process-mode', int(pyds.MODE_GPU))
    osd.set_property('qos', 0)

    # Link downstream elements
    streammux.link(pgie)
    pgie.link(converter)
    converter.link(osd)
    osd.link(sink)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', bus_call, loop)

    pgie_src_pad = pgie.get_static_pad('src')
    if not pgie_src_pad:
        sys.stderr.write('ERROR: Failed to get pgie src pad\n')
        sys.exit(1)
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    pipeline.set_state(Gst.State.PLAYING)

    sys.stdout.write('\n')

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)

    sys.stdout.write('\n')


def parse_args():
    global SOURCE, CONFIG_INFER, STREAMMUX_BATCH_SIZE, STREAMMUX_WIDTH, STREAMMUX_HEIGHT, GPU_ID, \
        PERF_MEASUREMENT_INTERVAL_SEC

    parser = argparse.ArgumentParser(description='DeepStream')
    parser.add_argument('-s', '--source', help='Source stream/file', default="file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4")
    parser.add_argument('-c', '--config-infer', help='Config infer file', default="/root/DeepStream-Yolo-Pose/config_infer_primary_yoloV8_pose.txt")
    parser.add_argument('-b', '--streammux-batch-size', type=int, default=1, help='Streammux batch-size (default: 1)')
    parser.add_argument('-w', '--streammux-width', type=int, default=1280, help='Streammux width (default: 1280)')
    parser.add_argument('-e', '--streammux-height', type=int, default=720, help='Streammux height (default: 720)')
    parser.add_argument('-g', '--gpu-id', type=int, default=0, help='GPU id (default: 0)')
    parser.add_argument('-f', '--fps-interval', type=int, default=5, help='FPS measurement interval (default: 5)')
    args = parser.parse_args()
    if args.source == '':
        sys.stderr.write('ERROR: Source not found\n')
        sys.exit(1)
    if args.config_infer == '' or not os.path.isfile(args.config_infer):
        sys.stderr.write('ERROR: Config infer not found\n')
        sys.exit(1)

    SOURCE = args.source
    CONFIG_INFER = args.config_infer
    STREAMMUX_BATCH_SIZE = args.streammux_batch_size
    STREAMMUX_WIDTH = args.streammux_width
    STREAMMUX_HEIGHT = args.streammux_height
    GPU_ID = args.gpu_id
    PERF_MEASUREMENT_INTERVAL_SEC = args.fps_interval


if __name__ == '__main__':
    parse_args()
    sys.exit(main())