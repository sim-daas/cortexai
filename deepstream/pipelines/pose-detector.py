import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import os
import sys
import time
import platform
from ctypes import *
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds

MAX_ELEMENTS_IN_DISPLAY_META = 16

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


class DeepStreamPoseNode(Node):
    def __init__(self):
        super().__init__('deepstream_pose_detector')
        # Configurable parameters
        self.STREAMMUX_WIDTH = 1920
        self.STREAMMUX_HEIGHT = 1080
        self.CONFIG_INFER = "/root/DeepStream-Yolo-Pose/config_infer_primary_yoloV8_pose.txt"
        self.PERF_MEASUREMENT_INTERVAL_SEC = 5
        self.fps_start_time = time.time()
        self.frame_count = 0

        # ROS2 publisher
        self.publisher = self.create_publisher(String, '/det', 10)

        # DeepStream pipeline setup
        Gst.init(None)
        self.loop = GLib.MainLoop()
        self.pipeline = Gst.Pipeline()

        # Streammux and source bin
        streammux = Gst.ElementFactory.make('nvstreammux', 'nvstreammux')
        self.pipeline.add(streammux)

        # Use file source for demo, can be replaced with camera/RTSP
        source_bin = self.create_uridecode_bin(0, "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4", streammux)
        self.pipeline.add(source_bin)

        pgie = Gst.ElementFactory.make('nvinfer', 'pgie')
        tracker = Gst.ElementFactory.make('nvtracker', 'nvtracker')
        converter = Gst.ElementFactory.make('nvvideoconvert', 'nvvideoconvert')
        osd = Gst.ElementFactory.make('nvdsosd', 'nvdsosd')
        sink = Gst.ElementFactory.make('nveglglessink', 'nveglglessink')

        # Set properties
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', 25000)
        streammux.set_property('width', self.STREAMMUX_WIDTH)
        streammux.set_property('height', self.STREAMMUX_HEIGHT)
        streammux.set_property('enable-padding', 0)
        streammux.set_property('live-source', 0)
        streammux.set_property('attach-sys-ts', 1)
        pgie.set_property('config-file-path', self.CONFIG_INFER)
        pgie.set_property('qos', 0)
        tracker.set_property('tracker-width', 640)
        tracker.set_property('tracker-height', 384)
        tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
        tracker.set_property('ll-config-file',
                             '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml')
        tracker.set_property('display-tracking-id', 1)
        tracker.set_property('qos', 0)
        osd.set_property('process-mode', int(pyds.MODE_GPU))
        osd.set_property('qos', 0)
        sink.set_property('async', 0)
        sink.set_property('sync', 0)
        sink.set_property('qos', 0)

        # Add and link elements
        for elem in [pgie, tracker, converter, osd, sink]:
            self.pipeline.add(elem)
        streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(converter)
        converter.link(osd)
        osd.link(sink)

        # Attach probe for pose publishing
        tracker_src_pad = tracker.get_static_pad('src')
        tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.tracker_src_pad_buffer_probe_ros, self)

        # Bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.bus_call, self.loop)

        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)

    def create_uridecode_bin(self, stream_id, uri, streammux):
        bin_name = 'source-bin-%04d' % stream_id
        bin = Gst.ElementFactory.make('uridecodebin', bin_name)
        bin.set_property('uri', uri)
        pad_name = 'sink_%u' % stream_id
        streammux_sink_pad = streammux.get_request_pad(pad_name)
        bin.connect('pad-added', self.cb_newpad, streammux_sink_pad)
        bin.connect('child-added', self.decodebin_child_added, 0)
        return bin

    def cb_newpad(self, decodebin, pad, user_data):
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
                sys.stderr.write('ERROR: decodebin did not pick NVIDIA decoder plugin\n')

    def decodebin_child_added(self, child_proxy, Object, name, user_data):
        if name.find('decodebin') != -1:
            Object.connect('child-added', self.decodebin_child_added, user_data)
        if name.find('nvv4l2decoder') != -1:
            Object.set_property('drop-frame-interval', 0)
            Object.set_property('num-extra-surfaces', 1)

    def parse_pose_from_meta(self, frame_meta, obj_meta):
        try:
            num_joints = int(obj_meta.mask_params.size / (sizeof(c_float) * 3))
            gain = min(obj_meta.mask_params.width / self.STREAMMUX_WIDTH,
                       obj_meta.mask_params.height / self.STREAMMUX_HEIGHT)
            pad_x = (obj_meta.mask_params.width - self.STREAMMUX_WIDTH * gain) / 2.0
            pad_y = (obj_meta.mask_params.height - self.STREAMMUX_HEIGHT * gain) / 2.0

            batch_meta = frame_meta.base_meta.batch_meta
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            data = obj_meta.mask_params.get_mask_array()

            for i in range(num_joints):
                xc = int((data[i * 3 + 0] - pad_x) / gain)
                yc = int((data[i * 3 + 1] - pad_y) / gain)
                confidence = data[i * 3 + 2]
                if confidence < 0.5:
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
                data = obj_meta.mask_params.get_mask_array()
                x1 = int((data[(skeleton[i][0] - 1) * 3 + 0] - pad_x) / gain)
                y1 = int((data[(skeleton[i][0] - 1) * 3 + 1] - pad_y) / gain)
                confidence1 = data[(skeleton[i][0] - 1) * 3 + 2]
                x2 = int((data[(skeleton[i][1] - 1) * 3 + 0] - pad_x) / gain)
                y2 = int((data[(skeleton[i][1] - 1) * 3 + 1] - pad_y) / gain)
                confidence2 = data[(skeleton[i][1] - 1) * 3 + 2]

                if confidence1 < 0.5 or confidence2 < 0.5:
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
        except Exception:
            pass

    def publish_poses(self, frame_meta):
        poses = []
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            except Exception:
                l_obj = l_obj.next if l_obj else None
                continue
            if not hasattr(obj, "mask_params") or not hasattr(obj.mask_params, "size") or obj.mask_params.size == 0:
                l_obj = l_obj.next if l_obj else None
                continue
            mask_params = obj.mask_params
            num_joints = int(mask_params.size / (sizeof(c_float) * 3))
            data = mask_params.get_mask_array()
            keypoints = []
            for i in range(num_joints):
                x = float(data[i * 3 + 0])
                y = float(data[i * 3 + 1])
                conf = float(data[i * 3 + 2])
                keypoints.append([x, y, conf])
            poses.append({
                "object_id": int(getattr(obj, "object_id", -1)),
                "keypoints": keypoints
            })
            l_obj = l_obj.next if l_obj else None
        msg = String()
        msg.data = json.dumps(poses)
        self.publisher.publish(msg)

    def update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.fps_start_time
        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            print(f"Average FPS: {fps:.2f}")
            self.frame_count = 0
            self.fps_start_time = current_time

    def tracker_src_pad_buffer_probe_ros(self, pad, info, user_data):
        buf = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                self.parse_pose_from_meta(frame_meta, obj_meta)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            self.publish_poses(frame_meta)
            self.update_fps()

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def bus_call(self, bus, message, user_data):
        loop = user_data
        t = message.type
        if t == Gst.MessageType.EOS:
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write('WARNING: %s: %s\n' % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write('ERROR: %s: %s\n' % (err, debug))
            loop.quit()
        return True


def main(args=None):
    rclpy.init(args=args)
    node = DeepStreamPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.pipeline.set_state(Gst.State.NULL)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()