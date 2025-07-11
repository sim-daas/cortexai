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
        super().__init__('deepstream_cam_pose_detector')
        # Configurable parameters
        self.STREAMMUX_WIDTH = 1280  # Match nvstreammux width
        self.STREAMMUX_HEIGHT = 720  # Match nvstreammux height
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

        # Camera input elements
        source = Gst.ElementFactory.make("v4l2src", "src")
        caps_v4l2src = Gst.ElementFactory.make("capsfilter", "src_caps")
        vidconvsrc = Gst.ElementFactory.make("videoconvert", "convert")
        nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "nvconvert")
        caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvconvert_caps")
        streammux = Gst.ElementFactory.make('nvstreammux', 'mux')
        pgie = Gst.ElementFactory.make('nvinfer', 'infer')
        converter = Gst.ElementFactory.make('nvvideoconvert', 'nvvidconv')
        osd = Gst.ElementFactory.make('nvdsosd', 'osd')
        sink = Gst.ElementFactory.make('nveglglessink', 'sink')

        # Set properties
        source.set_property('device', '/dev/video0')
        caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
        caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
        streammux.set_property('batch-size', 1)
        streammux.set_property('width', self.STREAMMUX_WIDTH)
        streammux.set_property('height', self.STREAMMUX_HEIGHT)
        streammux.set_property('live-source', True)
        streammux.set_property('batched-push-timeout', 25000)
        streammux.set_property('enable-padding', 0)
        streammux.set_property('attach-sys-ts', 1)
        pgie.set_property('config-file-path', self.CONFIG_INFER)
        pgie.set_property('qos', 0)
        osd.set_property('process-mode', int(pyds.MODE_GPU))
        osd.set_property('qos', 0)
        sink.set_property('sync', False)

        # Add elements to pipeline
        for elem in [source, caps_v4l2src, vidconvsrc, nvvidconvsrc, caps_vidconvsrc, streammux, pgie, converter, osd, sink]:
            self.pipeline.add(elem)

        # Link camera input elements
        source.link(caps_v4l2src)
        caps_v4l2src.link(vidconvsrc)
        vidconvsrc.link(nvvidconvsrc)
        nvvidconvsrc.link(caps_vidconvsrc)

        # Link camera pipeline to nvstreammux
        sinkpad = streammux.get_request_pad("sink_0")
        srcpad = caps_vidconvsrc.get_static_pad("src")
        srcpad.link(sinkpad)

        # Link downstream elements
        streammux.link(pgie)
        pgie.link(converter)
        converter.link(osd)
        osd.link(sink)

        # Attach probe for pose publishing
        pgie_src_pad = pgie.get_static_pad('src')
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.pgie_src_pad_buffer_probe_ros, self)

        # Bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.bus_call, self.loop)

        # Start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)

    def set_custom_bbox(self, obj_meta):
        border_width = 6
        font_size = 18
        x_offset = int(min(self.STREAMMUX_WIDTH - 1, max(0, obj_meta.rect_params.left - (border_width / 2))))
        y_offset = int(min(self.STREAMMUX_HEIGHT - 1, max(0, obj_meta.rect_params.top - (font_size * 2) + 1)))
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

    def parse_pose_from_meta(self, frame_meta, obj_meta):
        try:
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

            # --- Use the exact logic from pose-detector.py for scaling ---
            gain = min(mask_params.width / self.STREAMMUX_WIDTH,
                       mask_params.height / self.STREAMMUX_HEIGHT)
            pad_x = (mask_params.width - self.STREAMMUX_WIDTH * gain) / 2.0
            pad_y = (mask_params.height - self.STREAMMUX_HEIGHT * gain) / 2.0

            batch_meta = frame_meta.base_meta.batch_meta
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            data = mask_params.get_mask_array()

            for i in range(num_joints):
                xc = int((data[i * 3 + 0] - pad_x) / gain)
                yc = int((data[i * 3 + 1] - pad_y) / gain)
                confidence = data[i * 3 + 2]
                if confidence < 0.5 or not (0 <= xc < self.STREAMMUX_WIDTH and 0 <= yc < self.STREAMMUX_HEIGHT):
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
                    not (0 <= x1 < self.STREAMMUX_WIDTH and 0 <= y1 < self.STREAMMUX_HEIGHT) or
                    not (0 <= x2 < self.STREAMMUX_WIDTH and 0 <= y2 < self.STREAMMUX_HEIGHT)):
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

    def pgie_src_pad_buffer_probe_ros(self, pad, info, user_data):
        buf = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch_meta.frame_meta_list
        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            # Visualize and publish
            l_obj = frame_meta.obj_meta_list
            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                self.parse_pose_from_meta(frame_meta, obj_meta)
                self.set_custom_bbox(obj_meta)
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