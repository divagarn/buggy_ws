#!/usr/bin/env python3
"""
GPU-accelerated ground segmentation (PyTorch CUDA) with CPU fallback.
Optimised for NVIDIA Jetson AGX Orin 64 GB (JetPack 6) and desktop GPUs.

Pipeline (identical logic on GPU and CPU):
  1. Distance / height pre-filter
  2. Voxel downsample seed candidates
  3. Height-band seed selection
  4. Batched RANSAC plane fit (coarse)
  5. Refine plane on coarse inliers (fine)
  6. Slope-aware adaptive threshold - classify all points
  7. Statistical outlier rejection on far-range ground (k-NN)
  8. Publish ground / non-ground (zero-copy binary PointCloud2)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import time
from sensor_msgs_py import point_cloud2 as pc2

# -- Try CUDA via PyTorch --
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        _DEV = torch.device('cuda')
        torch.zeros(1, device=_DEV)  # warm-up
    else:
        _DEV = torch.device('cpu')
except ImportError:
    HAS_CUDA = False
    torch = None
    _DEV = None

# -- Constants --
VOXEL_SIZE              = 0.15
MAX_DISTANCE            = 25.0
LIDAR_HEIGHT            = 1.00
MIN_HEIGHT              = -1.60
MAX_HEIGHT              = 2.00
SEED_Z_MIN              = -1.45
SEED_Z_MAX              = -0.45
MAX_GROUND_ANGLE        = 15.0
RANSAC_ITERATIONS       = 300
RANSAC_THRESHOLD_COARSE = 0.20
RANSAC_THRESHOLD_FINE   = 0.12
MIN_GROUND_POINTS       = 50
BASE_GROUND_THRESH      = 0.15
DISTANCE_SCALE          = 0.012
MAX_ADAPTIVE_THRESH     = 0.35
GROUND_MAX_Z_ABOVE      = 0.25
GROUND_MAX_Z_BELOW      = 0.30
STAT_OUTLIER_K          = 15
STAT_OUTLIER_STD        = 2.0
RANSAC_BATCH            = 64  # GPU hypotheses evaluated per loop


# ====================================================================
#  GPU helpers
# ====================================================================

def _g(arr):
    """NumPy -> contiguous GPU tensor."""
    return torch.from_numpy(np.ascontiguousarray(arr)).to(_DEV, non_blocking=True)


def g_voxel_ds(pts, vs):
    """Voxel downsample on GPU using scatter_add."""
    vox = torch.floor(pts / vs).to(torch.int64)
    keys = vox[:, 0] * 2_000_000 + vox[:, 1] * 2_000 + vox[:, 2]
    u, inv = torch.unique(keys, return_inverse=True)
    nv = len(u)
    s = torch.zeros(nv, 3, dtype=torch.float64, device=_DEV)
    s.scatter_add_(0, inv.unsqueeze(1).expand(-1, 3), pts.double())
    c = torch.zeros(nv, dtype=torch.float64, device=_DEV)
    c.scatter_add_(0, inv, torch.ones(len(pts), dtype=torch.float64, device=_DEV))
    return (s / c.unsqueeze(1)).float()


def g_ransac(pts, dt, mi, ma, batch=RANSAC_BATCH):
    """Batched RANSAC plane fit on GPU."""
    n = pts.shape[0]
    if n < 3:
        return torch.tensor([0., 0., 1., 0.], device=_DEV), \
               torch.zeros(n, dtype=torch.bool, device=_DEV)
    cl = float(np.cos(np.radians(ma)))
    bc = 0
    beq = torch.tensor([0., 0., 1., 0.], device=_DEV)
    bm = torch.zeros(n, dtype=torch.bool, device=_DEV)
    for _ in range(max(1, mi // batch)):
        ix = torch.randint(0, n, (batch, 3), device=_DEV)
        p1, p2, p3 = pts[ix[:, 0]], pts[ix[:, 1]], pts[ix[:, 2]]
        nm = torch.linalg.cross(p2 - p1, p3 - p1)
        nl = torch.linalg.norm(nm, dim=1, keepdim=True)
        ok = nl.squeeze(1) > 1e-8
        nm = nm / nl.clamp(min=1e-8)
        # Flip normals so z > 0
        flip = nm[:, 2] < 0
        nm[flip] *= -1
        ok = ok & (nm[:, 2].abs() > cl)
        d = -(nm * p1).sum(1)
        # Compute distances for all points against all batch hypotheses: (n, batch)
        dist = torch.abs(pts @ nm.T + d.unsqueeze(0))
        ic = (dist < dt).sum(0)
        ic[~ok] = 0
        bi = torch.argmax(ic)
        cv = ic[bi].item()
        if cv > bc:
            bc = cv
            nr = nm[bi]
            beq = torch.tensor([nr[0].item(), nr[1].item(), nr[2].item(),
                                d[bi].item()], device=_DEV)
            bm = dist[:, bi] < dt
    return beq, bm


def g_sor(pts, k, sm):
    """Statistical outlier rejection via batched k-NN on GPU."""
    n = pts.shape[0]
    if n < k + 1:
        return torch.ones(n, dtype=torch.bool, device=_DEV)
    md = torch.empty(n, device=_DEV)
    ch = min(n, 8192)
    for s in range(0, n, ch):
        e = min(s + ch, n)
        d = torch.cdist(pts[s:e].unsqueeze(0), pts.unsqueeze(0)).squeeze(0)
        tv, _ = torch.topk(d, k + 1, dim=1, largest=False)
        md[s:e] = tv[:, 1:].mean(1)
    mu, sig = md.mean(), md.std()
    return md < (mu + sm * sig)


# ====================================================================
#  CPU fallback helpers
# ====================================================================

def c_voxel_ds(pts, vs):
    """Voxel downsample on CPU."""
    vi = np.floor(pts / vs).astype(np.int32)
    keys = vi[:, 0] * 2_000_000 + vi[:, 1] * 2_000 + vi[:, 2]
    _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    s = np.zeros((len(cnt), 3), dtype=np.float64)
    np.add.at(s, inv, pts)
    return (s / cnt[:, None]).astype(np.float32)


def c_ransac(pts, dt, mi, ma):
    """Sequential RANSAC plane fit on CPU."""
    n = len(pts)
    if n < 3:
        return np.array([0., 0., 1., 0.]), np.zeros(n, dtype=bool)
    cl = np.cos(np.radians(ma))
    bc = 0
    beq = np.array([0., 0., 1., 0.])
    bm = np.zeros(n, dtype=bool)
    for _ in range(mi):
        ix = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = pts[ix]
        nm = np.cross(p2 - p1, p3 - p1)
        nl = np.linalg.norm(nm)
        if nl < 1e-8:
            continue
        nm /= nl
        a, b, c = nm
        if c < 0:
            a, b, c = -a, -b, -c
        if abs(c) < cl:
            continue
        d = -(a * p1[0] + b * p1[1] + c * p1[2])
        ds = np.abs(pts[:, 0] * a + pts[:, 1] * b + pts[:, 2] * c + d)
        m = ds < dt
        cv = int(m.sum())
        if cv > bc:
            bc = cv
            beq = np.array([a, b, c, d])
            bm = m
    return beq, bm


def c_sor(pts, k, sm):
    """Statistical outlier rejection on CPU via chunked distance."""
    n = len(pts)
    if n < k + 1:
        return np.ones(n, dtype=bool)
    md = np.empty(n, dtype=np.float64)
    ch = 4000
    for s in range(0, n, ch):
        e = min(s + ch, n)
        diff = pts[None, :, :] - pts[s:e, None, :]
        d2 = (diff ** 2).sum(2)
        kp = np.partition(d2, k, axis=1)[:, :k + 1]
        ks = np.sort(kp, axis=1)[:, 1:k + 1]
        md[s:e] = np.mean(np.sqrt(ks), axis=1)
    mu, sig = md.mean(), md.std()
    return md < (mu + sm * sig)


# ====================================================================
#  Zero-copy binary PointCloud2 publisher
# ====================================================================

_PC_FIELDS = [
    PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.UINT32,  count=1),
]


def fast_pub(header, pts, pub, is_ground):
    """Build PointCloud2 binary data in NumPy - zero per-point Python loops."""
    n = len(pts)
    if n == 0:
        return
    if is_ground:
        d = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        mx = max(d.max(), 1e-6)
        g = (255 * (0.5 + 0.5 * (1.0 - d / mx))).astype(np.uint32)
        rgb = g << 8
    else:
        h = pts[:, 2]
        hmn, hmx = h.min(), h.max()
        rng = max(hmx - hmn, 0.01)
        nr = (h - hmn) / rng
        r = (255 * (1.0 - nr)).astype(np.uint32)
        g = (255 * np.abs(nr - 0.5) * 2).astype(np.uint32)
        b = (255 * nr).astype(np.uint32)
        rgb = (r << 16) | (g << 8) | b
    buf = np.empty((n, 4), dtype=np.float32)
    buf[:, :3] = pts[:, :3]
    buf[:, 3] = rgb.view(np.float32)
    m = PointCloud2()
    m.header = header
    m.height = 1
    m.width = n
    m.fields = _PC_FIELDS
    m.is_bigendian = False
    m.point_step = 16
    m.row_step = 16 * n
    m.data = buf.tobytes()
    m.is_dense = True
    pub.publish(m)


# ====================================================================
#  ROS 2 Node
# ====================================================================

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('segment_ground')
        self.gpub = self.create_publisher(PointCloud2, '/ground_points', 2)
        self.ngpub = self.create_publisher(PointCloud2, '/non_ground_points', 2)
        self.tt = 0.0
        self.cnt = 0
        self.use_gpu = HAS_CUDA

        if self.use_gpu:
            self.last_eq = torch.tensor([0., 0., 1., -LIDAR_HEIGHT], device=_DEV)
        else:
            self.last_eq = np.array([0., 0., 1., -LIDAR_HEIGHT])

        self.create_subscription(PointCloud2, '/velodyne_points', self.callback, 1)

        if self.use_gpu:
            be = "CUDA ({})".format(torch.cuda.get_device_name(0))
        else:
            be = "CPU (NumPy)"
        self.get_logger().info("Ground segmentation -- backend: {}".format(be))
        self.get_logger().info(
            "RANSAC coarse={} fine={} angle<{}deg iters={} batch={}".format(
                RANSAC_THRESHOLD_COARSE, RANSAC_THRESHOLD_FINE,
                MAX_GROUND_ANGLE, RANSAC_ITERATIONS, RANSAC_BATCH))

    def callback(self, msg):
        try:
            t0 = time.time()
            # Fast binary decode — structured dtype, no per-point Python loop
            n = msg.width * msg.height
            if n < MIN_GROUND_POINTS:
                return
            step = msg.point_step
            off = {}
            for f in msg.fields:
                if f.name in ('x', 'y', 'z'):
                    off[f.name] = f.offset
            if len(off) < 3:
                return
            dt = np.dtype({'names': ['x', 'y', 'z'],
                           'formats': ['<f4', '<f4', '<f4'],
                           'offsets': [off['x'], off['y'], off['z']],
                           'itemsize': step})
            raw = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
            pts_np = np.column_stack([raw['x'], raw['y'], raw['z']])
            valid = np.isfinite(pts_np).all(axis=1)
            if not valid.all():
                pts_np = pts_np[valid]
            nt = len(pts_np)
            if nt < MIN_GROUND_POINTS:
                return
            if self.use_gpu:
                self._gpu(msg, pts_np, nt, t0)
            else:
                self._cpu(msg, pts_np, nt, t0)
        except Exception as e:
            self.get_logger().error("Ground seg error: {}".format(e))
            import traceback
            self.get_logger().error(traceback.format_exc())

    # -- GPU path --
    def _gpu(self, msg, pts_np, nt, t0):
        p = _g(pts_np)

        # 1 - pre-filter
        rad = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        keep = (rad < MAX_DISTANCE) & (p[:, 2] > MIN_HEIGHT) & (p[:, 2] < MAX_HEIGHT)
        f = p[keep]
        if f.shape[0] < MIN_GROUND_POINTS:
            return

        # 2 - seeds
        sm = (f[:, 2] >= SEED_Z_MIN) & (f[:, 2] <= SEED_Z_MAX)
        seeds = f[sm]
        if seeds.shape[0] < MIN_GROUND_POINTS:
            zt = torch.quantile(f[:, 2], 0.20)
            seeds = f[f[:, 2] <= zt]
        if seeds.shape[0] < MIN_GROUND_POINTS:
            return

        # 3 - downsample seeds
        sd = g_voxel_ds(seeds, VOXEL_SIZE) if seeds.shape[0] > 5000 else seeds

        # 4 - coarse RANSAC
        eq, im = g_ransac(sd, RANSAC_THRESHOLD_COARSE, RANSAC_ITERATIONS, MAX_GROUND_ANGLE)
        if im.sum().item() < MIN_GROUND_POINTS:
            eq = self.last_eq
        else:
            # 5 - refine
            ci = sd[im]
            if ci.shape[0] > 3:
                eq2, _ = g_ransac(ci, RANSAC_THRESHOLD_FINE,
                                  RANSAC_ITERATIONS // 2, MAX_GROUND_ANGLE)
                if abs(eq2[2].item()) > np.cos(np.radians(MAX_GROUND_ANGLE)):
                    eq = eq2
            self.last_eq = eq

        a, b, c, d = eq

        # 6 - classify all points
        signed = p[:, 0] * a + p[:, 1] * b + p[:, 2] * c + d
        absd = signed.abs()
        ra = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        nb = 0.08 * torch.clamp(1.0 - ra / 3.0, min=0.0)
        adapt = torch.clamp(BASE_GROUND_THRESH + DISTANCE_SCALE * ra + nb,
                            max=MAX_ADAPTIVE_THRESH)
        gm = (absd < adapt) & (signed < GROUND_MAX_Z_ABOVE) & \
             (signed > -GROUND_MAX_Z_BELOW) & (p[:, 2] < MAX_HEIGHT)

        # 7 - SOR (far only)
        gi = torch.where(gm)[0]
        if gi.shape[0] > STAT_OUTLIER_K + 1:
            gp = p[gi]
            gr = torch.sqrt(gp[:, 0] ** 2 + gp[:, 1] ** 2)
            far = gr > 4.0
            fp = gp[far]
            if fp.shape[0] > STAT_OUTLIER_K + 1:
                sk = min(STAT_OUTLIER_K, 10) if fp.shape[0] > 8000 else STAT_OUTLIER_K
                smf = g_sor(fp, sk, STAT_OUTLIER_STD)
                sf = torch.ones(gi.shape[0], dtype=torch.bool, device=_DEV)
                fi = torch.where(far)[0]
                sf[fi[~smf]] = False
                gm[gi[~sf]] = False

        gm_np = gm.cpu().numpy()
        fast_pub(msg.header, pts_np[gm_np], self.gpub, True)
        fast_pub(msg.header, pts_np[~gm_np], self.ngpub, False)
        self._perf(t0, int(gm_np.sum()), nt - int(gm_np.sum()), nt)

    # -- CPU path --
    def _cpu(self, msg, pts_np, nt, t0):
        p = pts_np

        rad = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        keep = (rad < MAX_DISTANCE) & (p[:, 2] > MIN_HEIGHT) & (p[:, 2] < MAX_HEIGHT)
        f = p[keep]
        if len(f) < MIN_GROUND_POINTS:
            return

        sm = (f[:, 2] >= SEED_Z_MIN) & (f[:, 2] <= SEED_Z_MAX)
        seeds = f[sm]
        if len(seeds) < MIN_GROUND_POINTS:
            zt = np.percentile(f[:, 2], 20)
            seeds = f[f[:, 2] <= zt]
        if len(seeds) < MIN_GROUND_POINTS:
            return

        sd = c_voxel_ds(seeds, VOXEL_SIZE) if len(seeds) > 5000 else seeds

        eq, im = c_ransac(sd, RANSAC_THRESHOLD_COARSE, RANSAC_ITERATIONS, MAX_GROUND_ANGLE)
        if im.sum() < MIN_GROUND_POINTS:
            eq = self.last_eq
        else:
            ci = sd[im]
            if len(ci) > 3:
                eq2, _ = c_ransac(ci, RANSAC_THRESHOLD_FINE,
                                  RANSAC_ITERATIONS // 2, MAX_GROUND_ANGLE)
                if abs(eq2[2]) > np.cos(np.radians(MAX_GROUND_ANGLE)):
                    eq = eq2
            self.last_eq = eq

        a, b, c, d = eq
        signed = p[:, 0] * a + p[:, 1] * b + p[:, 2] * c + d
        absd = np.abs(signed)
        ra = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        nb = 0.08 * np.maximum(1.0 - ra / 3.0, 0.0)
        adapt = np.minimum(BASE_GROUND_THRESH + DISTANCE_SCALE * ra + nb,
                           MAX_ADAPTIVE_THRESH)
        gm = (absd < adapt) & (signed < GROUND_MAX_Z_ABOVE) & \
             (signed > -GROUND_MAX_Z_BELOW) & (p[:, 2] < MAX_HEIGHT)

        gi = np.where(gm)[0]
        if len(gi) > STAT_OUTLIER_K + 1:
            gp = p[gi]
            gr = np.sqrt(gp[:, 0] ** 2 + gp[:, 1] ** 2)
            far = gr > 4.0
            fp = gp[far]
            if len(fp) > STAT_OUTLIER_K + 1:
                sk = min(STAT_OUTLIER_K, 10) if len(fp) > 8000 else STAT_OUTLIER_K
                smf = c_sor(fp, sk, STAT_OUTLIER_STD)
                sf = np.ones(len(gi), dtype=bool)
                fi = np.where(far)[0]
                sf[fi[~smf]] = False
                gm[gi[~sf]] = False

        fast_pub(msg.header, p[gm], self.gpub, True)
        fast_pub(msg.header, p[~gm], self.ngpub, False)
        self._perf(t0, int(gm.sum()), nt - int(gm.sum()), nt)

    def _perf(self, t0, ng, nng, nt):
        dt = time.time() - t0
        self.tt += dt
        self.cnt += 1
        if self.cnt % 10 == 0:
            avg = self.tt / self.cnt
            hz = 1.0 / avg if avg > 0 else 0
            pct = 100.0 * ng / nt if nt else 0
            self.get_logger().info(
                "Ground: {} ({:.1f}%) | {} non-gnd | {:.1f}ms (avg {:.1f}ms {:.1f}Hz)".format(
                    ng, pct, nng, dt * 1000, avg * 1000, hz))


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
