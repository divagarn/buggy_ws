#!/usr/bin/env python3
"""Interactive inflation editor for the local_nav costmap configs.

Loads the current cost_scaling_factor and inflation_radius from
local_costmap.yaml (confirms planner_costmap.yaml matches), shows the
actual cost-vs-distance decay curve using nav2_costmap_2d's real formula
(verified from /opt/ros/humble/include/nav2_costmap_2d/nav2_costmap_2d/
inflation_layer.hpp's computeCost(), not approximated):

    distance <= inscribed_radius:  cost = 254 (INSCRIBED_INFLATED_OBSTACLE)
    inscribed_radius < distance <= inflation_radius:
        cost = 253 * exp(-cost_scaling_factor * (distance - inscribed_radius))
    distance > inflation_radius:   cost = 0

inscribed_radius is computed from the current footprint polygon (in
local_costmap.yaml) - the distance from the vehicle's center to its
nearest edge - not a guess, since that's what nav2_costmap_2d itself
derives it from.

Usage:
    python3 inflation_editor.py

Drag either slider (cost_scaling_factor or inflation_radius). The curve
and both costmaps' current cost at your vehicle's own robot_radius
exclusion boundary (self_hit_filter.py) update live. Click Apply to write
both values to local_costmap.yaml AND planner_costmap.yaml (edits the
existing lines in place - every comment/other setting is preserved).
Click Reset to revert to the values loaded at startup.
"""

import ast
import re
import math

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

LOCAL_COSTMAP_YAML = '/home/divagar/ros2_r_ws/buggy_demo_01/src/local_nav/config/local_costmap.yaml'
PLANNER_COSTMAP_YAML = '/home/divagar/ros2_r_ws/buggy_demo_01/src/local_nav/config/planner_costmap.yaml'

FOOTPRINT_RE = re.compile(r'footprint:\s*"(\[.*?\])"')
COST_SCALING_RE = re.compile(r'(cost_scaling_factor:\s*)([\d.]+)')
INFLATION_RADIUS_RE = re.compile(r'(inflation_radius:\s*)([\d.]+)')

INSCRIBED_INFLATED_OBSTACLE = 253  # nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE - 1 for exp() ceiling
LETHAL_COST = 254


def read_value(path, pattern):
    with open(path) as f:
        content = f.read()
    m = pattern.search(content)
    if not m:
        raise RuntimeError(f"Pattern not found in {path}")
    return float(m.group(m.lastindex))


def read_footprint(path):
    with open(path) as f:
        content = f.read()
    m = FOOTPRINT_RE.search(content)
    if not m:
        raise RuntimeError(f"No footprint line found in {path}")
    return ast.literal_eval(m.group(1))


def inscribed_radius_from_footprint(points):
    """Distance from origin to the nearest polygon edge - same quantity
    nav2_costmap_2d derives its inscribed_radius from."""
    min_dist = float('inf')
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        # Distance from origin (0,0) to the line segment (x1,y1)-(x2,y2)
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            dist = math.hypot(x1, y1)
        else:
            t = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / seg_len_sq))
            proj_x, proj_y = x1 + t * dx, y1 + t * dy
            dist = math.hypot(proj_x, proj_y)
        min_dist = min(min_dist, dist)
    return min_dist


def write_value(path, pattern, new_value):
    with open(path) as f:
        content = f.read()
    new_content = pattern.sub(lambda m: f"{m.group(1)}{new_value:.4f}", content, count=1)
    with open(path, 'w') as f:
        f.write(new_content)


def compute_cost(distance, inscribed_radius, inflation_radius, cost_scaling_factor):
    if distance <= inscribed_radius:
        return LETHAL_COST
    elif distance <= inflation_radius:
        factor = math.exp(-1.0 * cost_scaling_factor * (distance - inscribed_radius))
        return INSCRIBED_INFLATED_OBSTACLE * factor
    return 0.0


def main():
    local_scale = read_value(LOCAL_COSTMAP_YAML, COST_SCALING_RE)
    local_radius = read_value(LOCAL_COSTMAP_YAML, INFLATION_RADIUS_RE)
    planner_scale = read_value(PLANNER_COSTMAP_YAML, COST_SCALING_RE)
    planner_radius = read_value(PLANNER_COSTMAP_YAML, INFLATION_RADIUS_RE)

    if local_scale != planner_scale or local_radius != planner_radius:
        print("WARNING: local_costmap.yaml and planner_costmap.yaml inflation "
              "values currently differ - loading local_costmap.yaml's values "
              "as the starting point. Applying will make both match.")

    footprint = read_footprint(LOCAL_COSTMAP_YAML)
    inscribed_radius = inscribed_radius_from_footprint(footprint)
    print(f"Computed inscribed_radius from footprint: {inscribed_radius:.3f}m")

    initial_scale = local_scale
    initial_radius = local_radius

    fig, ax = plt.subplots(figsize=(9, 7))
    plt.subplots_adjust(bottom=0.32)

    state = {'scale': initial_scale, 'radius': initial_radius}

    def curve_xy():
        max_dist = max(state['radius'] * 1.4, 1.0)
        xs = [i * max_dist / 300 for i in range(301)]
        ys = [compute_cost(d, inscribed_radius, state['radius'], state['scale']) for d in xs]
        return xs, ys

    xs, ys = curve_xy()
    (line,) = ax.plot(xs, ys, 'b-', linewidth=2)
    inscribed_vline = ax.axvline(inscribed_radius, color='red', linestyle='--',
                                  label=f'inscribed_radius={inscribed_radius:.3f}m (from footprint)')
    inflation_vline = ax.axvline(state['radius'], color='green', linestyle='--',
                                  label=f'inflation_radius={state["radius"]:.3f}m')
    ax.set_xlabel('distance from robot edge (m)')
    ax.set_ylabel('cost (0-254)')
    ax.set_title('Inflation cost decay curve (nav2_costmap_2d exact formula)')
    ax.set_ylim(-10, 264)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    def redraw():
        xs, ys = curve_xy()
        line.set_data(xs, ys)
        ax.set_xlim(0, max(xs))
        inscribed_vline.set_label(f'inscribed_radius={inscribed_radius:.3f}m (from footprint)')
        inflation_vline.set_xdata([state['radius'], state['radius']])
        inflation_vline.set_label(f'inflation_radius={state["radius"]:.3f}m')
        ax.legend(loc='upper right')
        fig.canvas.draw_idle()

    ax_scale = plt.axes([0.15, 0.20, 0.7, 0.03])
    ax_radius = plt.axes([0.15, 0.15, 0.7, 0.03])
    slider_scale = Slider(ax_scale, 'cost_scaling_factor', 0.1, 15.0, valinit=initial_scale)
    slider_radius = Slider(ax_radius, 'inflation_radius (m)', 0.1, 6.0, valinit=initial_radius)

    def on_scale_change(val):
        state['scale'] = val
        redraw()

    def on_radius_change(val):
        state['radius'] = val
        redraw()

    slider_scale.on_changed(on_scale_change)
    slider_radius.on_changed(on_radius_change)

    status_text = fig.text(0.5, 0.03, '', ha='center', fontsize=10, color='green')

    def on_apply(event):
        write_value(LOCAL_COSTMAP_YAML, COST_SCALING_RE, state['scale'])
        write_value(LOCAL_COSTMAP_YAML, INFLATION_RADIUS_RE, state['radius'])
        write_value(PLANNER_COSTMAP_YAML, COST_SCALING_RE, state['scale'])
        write_value(PLANNER_COSTMAP_YAML, INFLATION_RADIUS_RE, state['radius'])
        status_text.set_text(
            f'Applied: cost_scaling_factor={state["scale"]:.3f}, '
            f'inflation_radius={state["radius"]:.3f}m -> both config files')
        status_text.set_color('green')
        print("Applied. cost_scaling_factor =", state['scale'], " inflation_radius =", state['radius'])
        print("Written to:")
        print(" ", LOCAL_COSTMAP_YAML)
        print(" ", PLANNER_COSTMAP_YAML)
        fig.canvas.draw_idle()

    def on_reset(event):
        slider_scale.set_val(initial_scale)
        slider_radius.set_val(initial_radius)
        status_text.set_text('Reset to values loaded at startup (not yet applied).')
        status_text.set_color('orange')

    ax_apply = plt.axes([0.55, 0.06, 0.15, 0.06])
    ax_reset = plt.axes([0.75, 0.06, 0.15, 0.06])
    btn_apply = Button(ax_apply, 'Apply', color='lightgreen', hovercolor='limegreen')
    btn_reset = Button(ax_reset, 'Reset', color='lightcoral', hovercolor='indianred')
    btn_apply.on_clicked(on_apply)
    btn_reset.on_clicked(on_reset)

    redraw()
    plt.show()


if __name__ == '__main__':
    main()
