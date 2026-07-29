#!/usr/bin/env python3
import rclpy
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import signal
import sys
import os

Gst.init(None)

class GstImagePublisher(Node):
    def __init__(self):
        super().__init__('gst_publisher_node')

        # Camera/pipeline parameters
        self.declare_parameter('device_id', 0)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('desired_fps', 30)
        self.declare_parameter('output_width', 640)
        self.declare_parameter('output_height', 360)
        self.declare_parameter('allow_fallback_devices', True)
        self.declare_parameter('use_nv16_first', False)

        self.device_id = self.get_parameter('device_id').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.desired_fps = int(self.get_parameter('desired_fps').value)
        self.output_width = int(self.get_parameter('output_width').value)
        self.output_height = int(self.get_parameter('output_height').value)
        self.allow_fallback_devices = bool(self.get_parameter('allow_fallback_devices').value)
        self.use_nv16_first = bool(self.get_parameter('use_nv16_first').value)

        if self.desired_fps <= 0:
            self.desired_fps = 30

        self.device_path = f"/dev/video{self.device_id}"
        self.pipeline = None
        self.pub = self.create_publisher(Image, self.camera_topic, 10)
        self.loop = GLib.MainLoop()
        self._publish_info_logged = False
        self._bus = None

        # Find and set up an available camera
        self.camera_found = False
        if os.path.exists(self.device_path):
            self.get_logger().info(f'Trying primary camera device: {self.device_path}')
            if self.setup_pipeline():
                self.camera_found = True
                self.get_logger().info(f'Successfully connected to primary camera: {self.device_path}')

        # Try other cameras if primary not available
        if not self.camera_found and self.allow_fallback_devices:
            self.get_logger().warning(f'Camera device {self.device_path} not available, trying others...')
            for i in range(10):  # Try video0 through video9
                if i == self.device_id:
                    continue
                self.device_id = i
                self.device_path = f"/dev/video{self.device_id}"
                if os.path.exists(self.device_path):
                    self.get_logger().info(f'Trying alternate camera device: {self.device_path}')
                    if self.setup_pipeline():
                        self.camera_found = True
                        self.get_logger().info(f'Successfully connected to camera: {self.device_path}')
                        break

        # Exit if no camera found
        if not self.camera_found:
            self.get_logger().error('Could not find any available camera device. Exiting.')
            sys.exit(1)
        
    def setup_pipeline(self):
        try:
            self._cleanup_pipeline()
            
            # Camera supports UYVY/NV16 at 1920x1080 and 1920x1536 up to 60fps.
            # Prefer 1920x1080 first for lower processing overhead.
            format_order = ['NV16', 'UYVY'] if self.use_nv16_first else ['UYVY', 'NV16']
            fps_order = [self.desired_fps, 60, 30]
            # Keep unique FPS values in order.
            fps_order = list(dict.fromkeys([fps for fps in fps_order if fps > 0]))
            formats_to_try = []
            for fmt in format_order:
                for width, height in [(1920, 1080), (1920, 1536)]:
                    for fps in fps_order:
                        formats_to_try.append((fmt, width, height, fps))
            
            for fmt, width, height, fps in formats_to_try:
                # Jetson-optimized pipeline:
                # v4l2src -> nvvidconv (GPU scale/convert) -> videoconvert -> appsink(BGR)
                # A single CPU conversion stage remains to feed OpenCV/ROS bgr8.
                pipeline_str = (
                    f'v4l2src device={self.device_path} io-mode=2 do-timestamp=true ! '
                    f'video/x-raw,format={fmt},width={width},height={height},framerate={fps}/1 ! '
                    'queue max-size-buffers=2 leaky=downstream ! '
                    f'nvvidconv ! video/x-raw(memory:NVMM),width={self.output_width},height={self.output_height},format=NV12 ! '
                    'nvvidconv ! video/x-raw,format=BGRx ! '
                    'videoconvert n-threads=2 ! '
                    'video/x-raw,format=BGR ! '
                    'appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true'
                )
                
                self.get_logger().info(
                    f'Trying pipeline: {fmt} {width}x{height}@{fps} -> {self.output_width}x{self.output_height}'
                )
                
                try:
                    self.pipeline = Gst.parse_launch(pipeline_str)
                    appsink = self.pipeline.get_by_name('sink')
                    appsink.set_property('emit-signals', True)
                    appsink.set_property('sync', False)
                    appsink.set_property('max-buffers', 1)
                    appsink.set_property('drop', True)
                    appsink.connect('new-sample', self.on_new_sample)
                    self.pipeline.set_state(Gst.State.PLAYING)
                    
                    # Wait up to 5 seconds for pipeline to start (camera needs time to initialize)
                    state_change_return, state, pending = self.pipeline.get_state(5 * Gst.SECOND)
                    self.get_logger().info(f'Pipeline state: {state.value_nick}')
                    
                    # Connect to bus for error messages
                    self._bus = self.pipeline.get_bus()
                    self._bus.add_signal_watch()
                    self._bus.connect('message', self.on_bus_message)
                    
                    # Check if pipeline is in playing state
                    if state == Gst.State.PLAYING:
                        self.get_logger().info(
                            f'Successfully connected with {fmt} at {width}x{height}@{fps}; output {self.output_width}x{self.output_height}'
                        )
                        self.width = self.output_width
                        self.height = self.output_height
                        self.format = 'BGR'  # Output format after conversion
                        return True
                    else:
                        self.get_logger().warning(f'Failed with format {fmt} at {width}x{height}@{fps}')
                        self._cleanup_pipeline()
                
                except Exception as e:
                    self.get_logger().warning(f'Error trying format {fmt} {width}x{height}@{fps}: {e}')
                    self._cleanup_pipeline()
                        
            self.get_logger().error('All format combinations failed')
            return False
            
        except Exception as e:
            self.get_logger().error(f'Error setting up pipeline: {e}')
            self._cleanup_pipeline()
            return False

    def _cleanup_pipeline(self):
        if self._bus is not None:
            try:
                self._bus.remove_signal_watch()
            except Exception:
                pass
            self._bus = None

        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.get_logger().error(f'GStreamer ERROR: {err}, debug: {debug}')
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            self.get_logger().warning(f'GStreamer WARNING: {err}, debug: {debug}')
        elif t == Gst.MessageType.EOS:
            self.get_logger().info('GStreamer: End of stream')
            self.shutdown(None, None)

    def on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        if sample is None:
            self.get_logger().warning('No sample received from GStreamer pipeline.')
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')

        try:
            # Map buffer directly to avoid an extra extract/copy before ROS message creation.
            ok, map_info = buf.map(Gst.MapFlags.READ)
            if not ok:
                self.get_logger().warning('Failed to map GStreamer buffer.')
                return Gst.FlowReturn.ERROR
            
            # Get the pixel format from GStreamer caps
            fmt = caps.get_structure(0).get_value('format')
            if fmt not in ('BGR', 'bgr'):
                self.get_logger().error(f'Unexpected format {fmt}; expected BGR from pipeline.')
                buf.unmap(map_info)
                return Gst.FlowReturn.ERROR
            expected = width * height * 3
            if map_info.size < expected:
                self.get_logger().error(
                    f'Invalid buffer size {map_info.size}, expected at least {expected}.'
                )
                buf.unmap(map_info)
                return Gst.FlowReturn.ERROR
            
            # Create ROS Image message with BGR8 encoding
            img_msg = Image()
            img_msg.header = Header()
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.height = height
            img_msg.width = width
            img_msg.encoding = 'bgr8'
            img_msg.is_bigendian = 0
            img_msg.step = width * 3
            img_msg.data = bytes(map_info.data[:expected])
            buf.unmap(map_info)
            
            self.pub.publish(img_msg)
            if not self._publish_info_logged:
                self.get_logger().info(
                    f'Publishing BGR images: {width}x{height} from {self.device_path} to {self.camera_topic}'
                )
                self._publish_info_logged = True
            
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return Gst.FlowReturn.ERROR
            
        return Gst.FlowReturn.OK

    def shutdown(self, signum, frame):
        self.get_logger().info('Shutdown signal received. Stopping pipeline and exiting...')
        self._cleanup_pipeline()
        self.loop.quit()

    def run(self):
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        self.get_logger().info('GstImagePublisher node started. Press Ctrl+C to exit.')
        try:
            self.loop.run()
        except Exception as e:
            self.get_logger().error(f'Exception in main loop: {e}')
        finally:
            self._cleanup_pipeline()
            self.get_logger().info('Pipeline stopped. Exiting node.')


def main(args=None):
    rclpy.init(args=args)
    node = GstImagePublisher()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
