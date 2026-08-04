#!/usr/bin/env python3
"""Standalone (no ROS2 required) sandbox for tuning carrot_path_publisher's
goal-heading-picking algorithm by drawing a map instead of running Gazebo.

This duplicates clear_distance()/pick_goal_heading() below line-for-line
from local_nav/local_nav/carrot_path_publisher.py's _clear_distance()/
_pick_goal_heading() (same math, same constants, same offset spread, same
smoothing/snap logic) - NOT imported from there, since that module pulls in
rclpy/nav2 message types and this tool is meant to run with nothing but
matplotlib+numpy, fast iteration, no ROS environment to source. If you tune
a parameter here and like the result, copy the value into the relevant
launch file's carrot_path_publisher parameters (or teb_controller.yaml,
for anything downstream of the picked heading) - this tool does not talk
to the real node at all.

What this does NOT model: SmacPlannerHybrid's actual path search, TEB's
trajectory optimization/obstacle avoidance, or the vehicle's real Ackermann
kinematics (min turning radius, acceleration limits). The simulated
"vehicle" here just teleports `step_size` meters along whatever heading
pick_goal_heading() returns each cycle - a deliberately simplified
kinematic proxy so the heading-pick algorithm itself (the thing the
sliders below actually tune) is isolated and exercised exactly as
carrot_path_publisher.py calls it, without the rest of the stack's
behavior blurring the picture. Real turning will be smoother/slower than
what you see here (TEB enforces min_turning_radius; this tool doesn't).

Usage:
    python3 carrot_heading_tuner.py

Controls:
    - Radio buttons select what a mouse click/drag does: paint obstacles,
      erase them, place the start pose, or place the goal point.
    - Sliders tune the same parameters carrot_path_publisher.py exposes as
      ROS params (scan_distance, carrot_distance, heading_switch_margin_m,
      goal_heading_alpha, goal_heading_snap_deg) plus one simulation-only
      parameter (step_size - how far the vehicle advances per replan
      cycle, standing in for resend_period_sec * actual speed).
    - "Compute Path" (or just changing a slider) re-runs the simulation
      and redraws the resulting path over the drawn map.
    - "Clear Obstacles" wipes the drawn map back to free space.
"""

import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

# --- Constants copied from carrot_path_publisher.py (keep in sync) --------
LETHAL_THRESHOLD = 90  # nav2_costmap_2d: 253 lethal / 254 inscribed / 0-252 inflation decay
RAYCAST_STEP = 0.2  # matches _clear_distance's default `step` argument

# --- Drawing-canvas-only settings (no equivalent in the real node) --------
GRID_RESOLUTION = 0.1  # m/cell, matches local_costmap.yaml's resolution
GRID_EXTENT_M = 24.0  # square world size (m) the canvas covers
OBSTACLE_BRUSH_RADIUS_M = 0.4
GOAL_TOLERANCE_M = 0.5
MAX_ITERATIONS = 500
STUCK_PATIENCE = 20  # iterations without progress toward the goal before giving up
LANE_SIMPLIFY_EPSILON_M = 0.3  # Douglas-Peucker tolerance for the "Lane (skeleton)" algorithm


class Costmap:
    """Minimal stand-in for the fields _clear_distance actually reads off
    a nav_msgs/OccupancyGrid (info.width/height/resolution,
    info.origin.position.x/y, data as a flat row-major int array) - just
    enough structure to reuse the exact same indexing math unmodified."""

    def __init__(self, width, height, resolution, origin_x, origin_y):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.data = np.zeros(width * height, dtype=np.int8)

    def grid_2d(self):
        return self.data.reshape(self.height, self.width)


class Params:
    def __init__(self):
        self.scan_distance = 13.0
        self.carrot_distance = 13.0
        self.heading_switch_margin_m = 1.5
        self.goal_heading_alpha = 0.4
        self.goal_heading_snap_deg = 25.0
        self.step_size = 1.0
        # How much a candidate's side clearance counts against its
        # distance-to-goal when scoring (goal-seeking mode only) - see
        # pick_goal_heading's docstring. 0 disables centering entirely
        # (original behavior: shortest technically-clear route wins, even
        # if it grazes a wall).
        self.centering_weight = 2.0
        # How far sideways to check for centering, capped so a candidate
        # in a very wide-open area doesn't keep scoring better forever
        # just for being in the middle of nowhere - reward stops accruing
        # past this much clearance on the tighter side.
        self.corridor_check_distance = 3.0


def clear_distance(costmap, x0, y0, heading, max_dist, step=RAYCAST_STEP):
    """Verbatim port of carrot_path_publisher.CarrotPathPublisher._clear_distance."""
    w, h = costmap.width, costmap.height
    res = costmap.resolution
    ox, oy = costmap.origin_x, costmap.origin_y
    data = costmap.data
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    d = 0.0
    while d < max_dist:
        d += step
        gx = int((x0 + d * cos_h - ox) / res)
        gy = int((y0 + d * sin_h - oy) / res)
        if gx < 0 or gx >= w or gy < 0 or gy >= h:
            return d  # ran off the known window - treat edge as the limit
        val = data[gy * w + gx]
        if val < 0 or val >= LETHAL_THRESHOLD:
            return d
    return max_dist


GOAL_SEEKING_OFFSETS_DEG = (
    15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90,
    105, -105, 120, -120, 135, -135, 150, -150, 165, -165)
REACTIVE_OFFSETS_DEG = (15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90)


def pick_goal_heading(costmap, x0, y0, current_yaw, smoothed_yaw, params, goal_xy=None):
    """Port of carrot_path_publisher.CarrotPathPublisher._pick_goal_heading.
    smoothed_yaw is threaded through explicitly (in place of self._smoothed_yaw)
    since this is a free function, not a method on a long-lived node.

    goal_xy=None reproduces the original purely-reactive behavior (scan
    around current_yaw, no notion of a destination, pick whichever
    candidate has the longest raw clear_distance). goal_xy set switches to
    goal-seeking mode - and that mode canNOT just be "same clearance-max
    scan, centered on the goal bearing instead of current heading": raw
    clearance has no idea a candidate is useful, only that it's open, so
    it will happily prefer heading almost straight back down the empty
    corridor you already drove through over continuing toward the goal,
    since "already-driven-through empty corridor" reads as more clear than
    "corridor ahead, not yet explored, curving toward the goal". Confirmed
    live on this tool: with a raw-clearance scan merely re-centered on the
    goal bearing, the simulated vehicle oscillated in place, repeatedly
    "deciding" a ~180deg reversal was the best option, and never reached an
    L-shaped corridor's goal at all.

    Goal-seeking mode instead scores each candidate by the resulting
    distance to the goal after advancing along it (by however far
    clear_distance says is safe, capped at carrot_distance/remaining
    distance) - not by raw clearance. A direct, unobstructed shot at the
    goal scores 0 (lands exactly on it) and always wins. When blocked,
    whichever candidate's reachable endpoint is closest to the goal wins -
    reversing away from the goal always scores worse than continuing
    forward, no matter how much open corridor is behind you, which is
    exactly the missing constraint above.

    That alone still isn't enough to get a good-looking path, just a
    legal one: pure "closest to goal" has no preference for staying away
    from walls, so around a bend it'll happily hug the inside edge as
    tight as clear_distance allows, since that's the shortest technically-
    clear route - confirmed live by drawing an actual curved corridor in
    this tool and watching the picked path graze the inner wall the whole
    way around instead of favoring the middle. candidate_centering adds a
    second term: two perpendicular raycasts at (x0,y0) relative to the
    candidate heading (not world-frame left/right - "left of travel" at
    whatever heading is being scored), capped at corridor_check_distance,
    scored by whichever SIDE is tighter (min, not average - a candidate
    that's wide open on one side and grazing a wall on the other should
    still be penalized for the grazing side, not let the open side hide
    it). Weighted by centering_weight and added directly to the goal-
    distance score (both already in meters), so heading_switch_margin_m's
    existing "how much better before switching" logic keeps working
    unchanged on the combined score."""
    if goal_xy is not None:
        gx, gy = goal_xy
        dist_to_goal = math.hypot(gx - x0, gy - y0)
        base_heading = math.atan2(gy - y0, gx - x0)
        offsets_deg = GOAL_SEEKING_OFFSETS_DEG
    else:
        dist_to_goal = None
        base_heading = current_yaw
        offsets_deg = REACTIVE_OFFSETS_DEG

    def candidate_advance(clear):
        if goal_xy is not None:
            return max(min(clear, params.carrot_distance, dist_to_goal), 0.0)
        return max(min(clear, params.carrot_distance), 1.0)

    def candidate_centering(heading, ex, ey):
        # Checked at the candidate's prospective ENDPOINT (ex,ey), not at
        # (x0,y0) - (x0,y0) is the same fixed point for every candidate,
        # so checking there mostly measures "how open is it here" rather
        # than "does THIS candidate's own endpoint sit near a wall",
        # which barely distinguishes between candidates at all.
        if params.centering_weight == 0.0:
            return 0.0
        left = clear_distance(costmap, ex, ey, heading + math.pi / 2, params.corridor_check_distance)
        right = clear_distance(costmap, ex, ey, heading - math.pi / 2, params.corridor_check_distance)
        return min(left, right)

    def candidate_score(heading, clear):
        if goal_xy is None:
            return clear  # original behavior: bigger clearance wins outright
        advance = candidate_advance(clear)
        ex = x0 + advance * math.cos(heading)
        ey = y0 + advance * math.sin(heading)
        goal_term = -math.hypot(gx - ex, gy - ey)  # bigger (closer to 0) is better, like clear
        return goal_term + params.centering_weight * candidate_centering(heading, ex, ey)

    best_heading = base_heading
    best_clear = clear_distance(costmap, x0, y0, base_heading, params.scan_distance)
    best_score = candidate_score(base_heading, best_clear)

    for offset_deg in offsets_deg:
        heading = base_heading + math.radians(offset_deg)
        clear = clear_distance(costmap, x0, y0, heading, params.scan_distance)
        score = candidate_score(heading, clear)
        if score > best_score + params.heading_switch_margin_m:
            best_score = score
            best_clear = clear
            best_heading = heading

    goal_dist = candidate_advance(best_clear)

    if smoothed_yaw is None:
        smoothed_yaw = best_heading
    else:
        diff = math.atan2(
            math.sin(best_heading - smoothed_yaw),
            math.cos(best_heading - smoothed_yaw))
        if abs(math.degrees(diff)) >= params.goal_heading_snap_deg:
            smoothed_yaw = best_heading  # real corner - commit now, don't lag
        else:
            smoothed_yaw += params.goal_heading_alpha * diff

    return smoothed_yaw, goal_dist


MIN_STEP_M = 0.5


def split_into_clear_segments(costmap, x0, y0, gx, gy, params, max_steps, step_m):
    """Port of carrot_path_publisher.CarrotPathPublisher._split_into_clear_segments:
    builds a polyline of waypoints from (x0,y0) to (gx,gy) by repeatedly
    calling pick_goal_heading and advancing step_m at a time - i.e. this
    IS simulate_path's own loop, reused to produce a waypoint list to hand
    to the controller directly instead of only a live/visual simulation.

    Two earlier versions of this function tried to be a proper geometric
    "insert one detour waypoint, recurse" splitter with its own scoring
    loop, and both were wrong in different ways, found only by running
    them on this tool's own L/T-junction test (not by inspection):
      1. Scored each candidate by the goal-distance reached after
         advancing the candidate's own FULL clear_distance. A diagonal
         heading with only ~6m of clearance can score better (ends up
         numerically closer to the goal) than continuing another ~12m down
         the actual corridor at ~0deg, because the diagonal points more
         directly at the goal - so one greedy jump locks onto the
         diagonal, quickly runs out of room hugging the inside wall, and
         never finds its way through.
      2. Capped each step's advance at a small fixed distance to force
         many small steps instead of one large one (the right instinct),
         but then floored that capped advance at a minimum step so the
         vehicle was never allowed to stay still - which, once genuinely
         cornered (clearance near zero), forced it forward anyway,
         punching straight through the wall in small increments and never
         recovering (every step after that started from an already-inside-
         a-wall position, so its own clear_distance readings were
         meaningless from then on).

    pick_goal_heading doesn't have either problem: its own goal_dist is
    already bounded by real measured clearance (never forced past what's
    actually open), and calling it repeatedly with a small step_m re-
    evaluates from a fresh position often enough that a momentarily-
    attractive diagonal never gets to commit the vehicle too far into a
    corner before the next call corrects course - confirmed live, this is
    exactly the version of the algorithm that already reached the goal
    cleanly on this tool's L/T-junction test."""
    x, y = x0, y0
    yaw = math.atan2(gy - y0, gx - x0)
    smoothed_yaw = None
    waypoints = [(x, y)]
    for _ in range(max_steps):
        dist = math.hypot(gx - x, gy - y)
        if dist < MIN_STEP_M:
            break
        smoothed_yaw, goal_dist = pick_goal_heading(
            costmap, x, y, yaw, smoothed_yaw, params, goal_xy=(gx, gy))
        advance = min(step_m, goal_dist, dist)
        x = x + advance * math.cos(smoothed_yaw)
        y = y + advance * math.sin(smoothed_yaw)
        yaw = smoothed_yaw
        waypoints.append((x, y))
        if math.hypot(gx - x, gy - y) < MIN_STEP_M:
            break
    return waypoints


def simulate_path(costmap, start_xy, goal_xy, params):
    """Repeatedly calls pick_goal_heading like carrot_path_publisher's
    request_plan() timer does, advancing a simplified point-vehicle
    step_size meters per cycle along whichever heading gets picked (see
    module docstring for what this does and doesn't model). Returns
    (waypoints, status) where waypoints is a list of (x, y, yaw, goal_dist)."""
    x, y = start_xy
    goal_x, goal_y = goal_xy
    initial_yaw = math.atan2(goal_y - y, goal_x - x)
    yaw = initial_yaw
    smoothed_yaw = None
    waypoints = [(x, y, yaw, None)]

    best_remaining = math.hypot(goal_x - x, goal_y - y)
    stuck_counter = 0
    status = f"stopped after {MAX_ITERATIONS} iterations without reaching the goal"

    for i in range(MAX_ITERATIONS):
        smoothed_yaw, goal_dist = pick_goal_heading(
            costmap, x, y, yaw, smoothed_yaw, params, goal_xy=goal_xy)
        advance = min(params.step_size, goal_dist)
        x = x + advance * math.cos(smoothed_yaw)
        y = y + advance * math.sin(smoothed_yaw)
        yaw = smoothed_yaw
        waypoints.append((x, y, yaw, goal_dist))

        remaining = math.hypot(goal_x - x, goal_y - y)
        if remaining <= GOAL_TOLERANCE_M:
            status = f"reached goal in {i + 1} steps"
            break
        if remaining < best_remaining - 0.1:
            best_remaining = remaining
            stuck_counter = 0
        else:
            stuck_counter += 1
            if stuck_counter >= STUCK_PATIENCE:
                status = (
                    f"stuck after {i + 1} steps - no progress toward goal in "
                    f"{STUCK_PATIENCE} cycles (boxed in, or parameters too conservative "
                    f"to commit to the opening)")
                break

    return waypoints, status


def build_lane_path(costmap, x0, y0, gx, gy, lane_simplify_epsilon):
    """Port of carrot_path_publisher.CarrotPathPublisher._build_lane_path:
    medial-axis (skeleton) of the free space, graph search along it,
    Douglas-Peucker simplified. See that method's docstring - the greedy
    "shortcut to whichever point is still technically clear" alternative
    was tried first and throws away the whole centering benefit, producing
    the same corner-hugging diagonal the other two techniques already had
    a problem with. Returns a list of (x, y) waypoints, or None if no
    skeleton path connects start to goal."""
    free = (costmap.data.reshape(costmap.height, costmap.width) >= 0) & \
           (costmap.data.reshape(costmap.height, costmap.width) < LETHAL_THRESHOLD)
    skeleton = skeletonize(free)
    ys, xs = np.nonzero(skeleton)
    if len(ys) == 0:
        return None

    pts = np.stack([ys, xs], axis=1)
    tree = cKDTree(pts)
    res = costmap.resolution
    ox, oy = costmap.origin_x, costmap.origin_y

    graph = nx.Graph()
    idx_of = {(int(y), int(x)): True for y, x in pts}
    for y, x in pts:
        y, x = int(y), int(x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                if (y + dy, x + dx) in idx_of:
                    graph.add_edge((y, x), (y + dy, x + dx), weight=math.hypot(dy, dx) * res)

    def nearest_skeleton_cell(wx, wy):
        cell_y = (wy - oy) / res
        cell_x = (wx - ox) / res
        _, i = tree.query([cell_y, cell_x])
        return (int(pts[i][0]), int(pts[i][1]))

    start_cell = nearest_skeleton_cell(x0, y0)
    goal_cell = nearest_skeleton_cell(gx, gy)

    try:
        path_cells = nx.shortest_path(graph, start_cell, goal_cell, weight='weight')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    path_xy = [(ox + (cx + 0.5) * res, oy + (cy + 0.5) * res) for cy, cx in path_cells]
    return simplify_path_dp(path_xy, lane_simplify_epsilon)


def simplify_path_dp(points, epsilon):
    """Ramer-Douglas-Peucker: keeps the path within epsilon meters
    (perpendicular distance) of the original polyline's shape - see
    build_lane_path's docstring for why this, not greedy shortcutting."""
    if len(points) < 3:
        return points

    def perpendicular_distance(p, a, b):
        (px, py), (ax, ay), (bx, by) = p, a, b
        dx, dy = bx - ax, by - ay
        if dx == 0.0 and dy == 0.0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        proj_x, proj_y = ax + t * dx, ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        left = simplify_path_dp(points[:index + 1], epsilon)
        right = simplify_path_dp(points[index:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


class Tuner:
    def __init__(self):
        self.params = Params()
        n = int(round(GRID_EXTENT_M / GRID_RESOLUTION))
        half = GRID_EXTENT_M / 2.0
        self.costmap = Costmap(n, n, GRID_RESOLUTION, -half, -half)

        self.start_xy = (-half + 2.0, 0.0)
        self.goal_xy = (half - 2.0, 0.0)
        self.mode = "Draw Obstacle"
        self._mouse_down = False

        self.fig = plt.figure(figsize=(13, 8))
        self.fig.suptitle(
            "carrot_path_publisher heading-pick tuner "
            "(draws with _clear_distance/_pick_goal_heading, not the real controller)")
        self.ax = self.fig.add_axes([0.05, 0.08, 0.62, 0.84])
        self.ax.set_xlim(-half, half)
        self.ax.set_ylim(-half, half)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")

        self.im = self.ax.imshow(
            self.costmap.grid_2d(), origin="lower", cmap="Greys", vmin=0, vmax=100,
            extent=[-half, half, -half, half])

        (self.path_line,) = self.ax.plot([], [], "-o", color="tab:blue", markersize=3, linewidth=1.5)
        self.start_marker = self.ax.scatter([], [], c="tab:green", s=120, marker="o", zorder=5, label="start")
        self.goal_marker = self.ax.scatter([], [], c="tab:red", s=160, marker="*", zorder=5, label="goal")
        self.heading_quiver = None
        self.ax.legend(loc="upper right")
        self.status_text = self.ax.text(
            0.02, 0.02, "", transform=self.ax.transAxes, fontsize=9,
            va="bottom", ha="left", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        self._build_controls()
        self._redraw_map()
        self.recompute()

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # ---- controls ----------------------------------------------------

    def _build_controls(self):
        radio_ax = self.fig.add_axes([0.71, 0.72, 0.26, 0.18])
        radio_ax.set_title("Click/drag action", fontsize=9)
        self.radio = RadioButtons(
            radio_ax, ("Draw Obstacle", "Erase", "Set Start", "Set Goal"))
        self.radio.on_clicked(self._on_mode_change)

        slider_specs = [
            ("scan_distance", 1.0, 25.0, self.params.scan_distance),
            ("carrot_distance", 1.0, 25.0, self.params.carrot_distance),
            ("heading_switch_margin_m", 0.0, 5.0, self.params.heading_switch_margin_m),
            ("goal_heading_alpha", 0.05, 1.0, self.params.goal_heading_alpha),
            ("goal_heading_snap_deg", 5.0, 60.0, self.params.goal_heading_snap_deg),
            ("step_size", 0.2, 5.0, self.params.step_size),
            ("centering_weight", 0.0, 5.0, self.params.centering_weight),
            ("corridor_check_distance", 0.5, 8.0, self.params.corridor_check_distance),
        ]
        self.sliders = {}
        top = 0.66
        for i, (name, lo, hi, default) in enumerate(slider_specs):
            ax = self.fig.add_axes([0.71, top - i * 0.055, 0.26, 0.025])
            slider = Slider(ax, name, lo, hi, valinit=default)
            slider.on_changed(self._on_slider_change)
            self.sliders[name] = slider

        algo_ax = self.fig.add_axes([0.71, 0.15, 0.26, 0.09])
        algo_ax.set_title("Path algorithm", fontsize=9)
        self.algo_radio = RadioButtons(algo_ax, ("Greedy (step-by-step)", "Lane (skeleton)"))
        self.algo_radio.on_clicked(self._on_algo_change)
        self.algorithm = "Greedy (step-by-step)"

        compute_ax = self.fig.add_axes([0.71, 0.08, 0.26, 0.05])
        self.compute_btn = Button(compute_ax, "Compute Path")
        self.compute_btn.on_clicked(lambda _event: self.recompute())

        clear_ax = self.fig.add_axes([0.71, 0.02, 0.26, 0.05])
        self.clear_btn = Button(clear_ax, "Clear Obstacles")
        self.clear_btn.on_clicked(lambda _event: self._clear_obstacles())

    def _on_mode_change(self, label):
        self.mode = label

    def _on_algo_change(self, label):
        self.algorithm = label
        self.recompute()

    def _on_slider_change(self, _value):
        self.params.scan_distance = self.sliders["scan_distance"].val
        self.params.carrot_distance = self.sliders["carrot_distance"].val
        self.params.heading_switch_margin_m = self.sliders["heading_switch_margin_m"].val
        self.params.goal_heading_alpha = self.sliders["goal_heading_alpha"].val
        self.params.goal_heading_snap_deg = self.sliders["goal_heading_snap_deg"].val
        self.params.step_size = self.sliders["step_size"].val
        self.params.centering_weight = self.sliders["centering_weight"].val
        self.params.corridor_check_distance = self.sliders["corridor_check_distance"].val
        self.recompute()

    # ---- mouse handling ------------------------------------------------

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._mouse_down = True
        if self.mode == "Set Start":
            self.start_xy = (event.xdata, event.ydata)
            self.recompute()
        elif self.mode == "Set Goal":
            self.goal_xy = (event.xdata, event.ydata)
            self.recompute()
        else:
            self._paint(event.xdata, event.ydata)

    def _on_release(self, _event):
        if self._mouse_down and self.mode in ("Draw Obstacle", "Erase"):
            self.recompute()
        self._mouse_down = False

    def _on_motion(self, event):
        if not self._mouse_down or event.inaxes != self.ax or event.xdata is None:
            return
        if self.mode in ("Draw Obstacle", "Erase"):
            self._paint(event.xdata, event.ydata)

    def _paint(self, wx, wy):
        res = self.costmap.resolution
        radius_cells = max(1, int(round(OBSTACLE_BRUSH_RADIUS_M / res)))
        gx = int((wx - self.costmap.origin_x) / res)
        gy = int((wy - self.costmap.origin_y) / res)
        value = 100 if self.mode == "Draw Obstacle" else 0
        grid = self.costmap.grid_2d()
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                yy, xx = gy + dy, gx + dx
                if 0 <= yy < self.costmap.height and 0 <= xx < self.costmap.width:
                    grid[yy, xx] = value
        self._redraw_map()

    def _clear_obstacles(self):
        self.costmap.data[:] = 0
        self._redraw_map()
        self.recompute()

    def _redraw_map(self):
        self.im.set_data(self.costmap.grid_2d())
        self.fig.canvas.draw_idle()

    # ---- simulation + redraw -------------------------------------------

    def recompute(self):
        if self.algorithm == "Lane (skeleton)":
            lane_xy = build_lane_path(
                self.costmap, self.start_xy[0], self.start_xy[1],
                self.goal_xy[0], self.goal_xy[1], LANE_SIMPLIFY_EPSILON_M)
            if lane_xy is None:
                waypoints = [(self.start_xy[0], self.start_xy[1], 0.0, None)]
                status = "no skeleton path connects start to goal"
            else:
                waypoints = [(x, y, 0.0, None) for x, y in lane_xy]
                # Fill in headings (toward the next waypoint) for the quiver display only.
                for i in range(len(waypoints) - 1):
                    x0, y0, _, _ = waypoints[i]
                    x1, y1, _, _ = waypoints[i + 1]
                    waypoints[i] = (x0, y0, math.atan2(y1 - y0, x1 - x0), None)
                final = math.hypot(self.goal_xy[0] - lane_xy[-1][0], self.goal_xy[1] - lane_xy[-1][1])
                status = "reached goal" if final < GOAL_TOLERANCE_M else "skeleton path found (not touching goal exactly)"
        else:
            waypoints, status = simulate_path(self.costmap, self.start_xy, self.goal_xy, self.params)

        xs = [w[0] for w in waypoints]
        ys = [w[1] for w in waypoints]
        self.path_line.set_data(xs, ys)

        self.start_marker.set_offsets([self.start_xy])
        self.goal_marker.set_offsets([self.goal_xy])

        if self.heading_quiver is not None:
            self.heading_quiver.remove()
            self.heading_quiver = None
        stride = max(1, len(waypoints) // 25)
        sampled = waypoints[::stride]
        if sampled:
            qx = [w[0] for w in sampled]
            qy = [w[1] for w in sampled]
            qu = [math.cos(w[2]) for w in sampled]
            qv = [math.sin(w[2]) for w in sampled]
            self.heading_quiver = self.ax.quiver(
                qx, qy, qu, qv, color="tab:orange", scale=20, width=0.004, zorder=4)

        final_remaining = math.hypot(
            self.goal_xy[0] - waypoints[-1][0], self.goal_xy[1] - waypoints[-1][1])
        self.status_text.set_text(
            f"{status}\n"
            f"waypoints: {len(waypoints)}   final distance to goal: {final_remaining:.2f} m")
        self.fig.canvas.draw_idle()


def main():
    Tuner()
    plt.show()


if __name__ == "__main__":
    main()
