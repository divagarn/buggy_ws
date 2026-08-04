# Saved maps

`ros2 launch buggy_mapping mapping.launch.py` (or `real_mapping.launch.py`)
builds a map here as you drive. Once you've covered the loop, save it:

```bash
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
  "{filename: '/home/divagar/ros2_r_ws/buggy_demo_01/src/buggy_mapping/maps/<name>'}"
```

This is slam_toolbox's own serialization (`<name>.data` + `<name>.posegraph`)
- what `closed_loop_navigate.launch.py`'s `map_name` argument points at for
localization mode. It is NOT the same thing as `nav2_map_server`'s
`map_saver_cli` (`.yaml`/`.pgm`) - that produces a plain image-based map for
tools like AMCL, which this project isn't using; slam_toolbox's own
localization mode needs its own serialized pose graph instead, so use the
service call above, not `map_saver_cli`.

Both mapping.launch.py and closed_loop_navigate.launch.py get `/map`
(nav_msgs/OccupancyGrid) directly from slam_toolbox itself in either mode
- no separate map_server node is used anywhere in this package.
