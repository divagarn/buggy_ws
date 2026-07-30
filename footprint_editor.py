#!/usr/bin/env python3
"""Interactive footprint editor for the local_nav costmap configs.

Loads the current footprint polygon from local_costmap.yaml (and confirms
planner_costmap.yaml matches), shows it as a draggable rectangle with a
grid for scale reference, and writes the updated polygon back to BOTH
files (they must stay in sync - local and global costmaps both need the
same real vehicle footprint) when you click Apply. Edits the footprint
line in place with a targeted string replace, not a full YAML re-dump -
this preserves every comment/explanation already in those files.

Usage:
    python3 footprint_editor.py

Drag any of the 4 corner points with the mouse. The polygon updates live.
Click "Apply" to write the new footprint to both config files (requires a
colcon build afterward to take effect, though symlink-install workspaces
pick up the plain text change immediately on next node restart - no
rebuild strictly needed, but running colcon build is still recommended
to catch anything else that might depend on it).
Click "Reset" to revert to the values that were loaded at startup.
"""

import ast
import re

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

LOCAL_COSTMAP_YAML = '/home/divagar/ros2_r_ws/buggy_demo_01/src/local_nav/config/local_costmap.yaml'
PLANNER_COSTMAP_YAML = '/home/divagar/ros2_r_ws/buggy_demo_01/src/local_nav/config/planner_costmap.yaml'

FOOTPRINT_RE = re.compile(r'(footprint:\s*")(\[.*?\])(")')


def read_footprint(path):
    with open(path) as f:
        content = f.read()
    m = FOOTPRINT_RE.search(content)
    if not m:
        raise RuntimeError(f"No footprint line found in {path}")
    return ast.literal_eval(m.group(2))


def write_footprint(path, points):
    with open(path) as f:
        content = f.read()
    new_str = '[' + ', '.join(f'[{x:.4f}, {y:.4f}]' for x, y in points) + ']'
    new_content = FOOTPRINT_RE.sub(lambda m: m.group(1) + new_str + m.group(3), content, count=1)
    with open(path, 'w') as f:
        f.write(new_content)


def main():
    local_fp = read_footprint(LOCAL_COSTMAP_YAML)
    planner_fp = read_footprint(PLANNER_COSTMAP_YAML)
    if local_fp != planner_fp:
        print("WARNING: local_costmap.yaml and planner_costmap.yaml footprints "
              "currently differ - loading local_costmap.yaml's values as the "
              "starting point. Applying will make both match.")
    initial_points = list(local_fp)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('x (m, forward +)')
    ax.set_ylabel('y (m, left +)')
    ax.set_title('Drag corners to resize footprint, then click Apply')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2)  # base_link origin

    points = [list(p) for p in initial_points]
    n = len(points)

    closed_x = [p[0] for p in points] + [points[0][0]]
    closed_y = [p[1] for p in points] + [points[0][1]]
    (polygon_line,) = ax.plot(closed_x, closed_y, 'b-', linewidth=2)
    scatter = ax.scatter([p[0] for p in points], [p[1] for p in points],
                          s=120, c='red', zorder=5, picker=8)

    dims_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update_dims_text():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        length = max(xs) - min(xs)
        width = max(ys) - min(ys)
        lines = [f'length={length:.3f}m  width={width:.3f}m']
        for i, (x, y) in enumerate(points):
            lines.append(f'  p{i}: ({x:.3f}, {y:.3f})')
        dims_text.set_text('\n'.join(lines))

    def redraw():
        closed_x = [p[0] for p in points] + [points[0][0]]
        closed_y = [p[1] for p in points] + [points[0][1]]
        polygon_line.set_data(closed_x, closed_y)
        scatter.set_offsets(points)
        update_dims_text()
        margin = 1.5
        all_x = closed_x
        all_y = closed_y
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        fig.canvas.draw_idle()

    update_dims_text()
    redraw()

    dragging = {'index': None}

    def on_press(event):
        if event.inaxes != ax:
            return
        offsets = scatter.get_offsets()
        for i, (x, y) in enumerate(offsets):
            if abs(event.xdata - x) < 0.3 and abs(event.ydata - y) < 0.3:
                dragging['index'] = i
                return

    def on_motion(event):
        if dragging['index'] is None or event.inaxes != ax:
            return
        points[dragging['index']][0] = event.xdata
        points[dragging['index']][1] = event.ydata
        redraw()

    def on_release(event):
        dragging['index'] = None

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    status_text = fig.text(0.5, 0.02, '', ha='center', fontsize=10, color='green')

    def on_apply(event):
        write_footprint(LOCAL_COSTMAP_YAML, points)
        write_footprint(PLANNER_COSTMAP_YAML, points)
        status_text.set_text(f'Applied - wrote footprint to both config files at {LOCAL_COSTMAP_YAML.split("/")[-1]} and {PLANNER_COSTMAP_YAML.split("/")[-1]}')
        status_text.set_color('green')
        print("Applied. New footprint:", points)
        print("Written to:")
        print(" ", LOCAL_COSTMAP_YAML)
        print(" ", PLANNER_COSTMAP_YAML)
        fig.canvas.draw_idle()

    def on_reset(event):
        for i, p in enumerate(initial_points):
            points[i][0], points[i][1] = p[0], p[1]
        status_text.set_text('Reset to values loaded at startup (not yet applied).')
        status_text.set_color('orange')
        redraw()

    ax_apply = plt.axes([0.55, 0.02, 0.15, 0.06])
    ax_reset = plt.axes([0.75, 0.02, 0.15, 0.06])
    btn_apply = Button(ax_apply, 'Apply', color='lightgreen', hovercolor='limegreen')
    btn_reset = Button(ax_reset, 'Reset', color='lightcoral', hovercolor='indianred')
    btn_apply.on_clicked(on_apply)
    btn_reset.on_clicked(on_reset)

    plt.show()


if __name__ == '__main__':
    main()
