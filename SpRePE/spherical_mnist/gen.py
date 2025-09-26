#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate spherical MNIST from IDX (no lie_learn), save as .npy
- output:
    out_dir/
      train/data.npy   # [N,1,2b,2b]
      train/label.npy  # [N]
      test/data.npy
      test/label.npy
"""

import os
import math
import struct
import argparse
import numpy as np

# ---------------- MNIST IDX loader ----------------
def load_mnist_idx(images_path: str, labels_path: str):
    labels = np.load(labels_path)
    images = np.load(images_path).reshape(-1, 64, 64)
    return images, labels

# ---------------- random SO(3) rotation ----------------
def rand_rotation_matrix(deflection: float = 1.0) -> np.ndarray:
    """
    0=no rotation, 1≈uniform over SO(3)
    """
    u1, u2, u3 = np.random.uniform(size=(3,))
    theta = u1 * 2.0 * deflection * np.pi
    phi   = u2 * 2.0 * np.pi
    z     = u3 * 2.0 * deflection

    r = np.sqrt(z)
    V = np.array([np.sin(phi)*r, np.cos(phi)*r, np.sqrt(max(0.0, 2.0 - z))], dtype=np.float64)
    st, ct = np.sin(theta), np.cos(theta)
    Rz = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)), dtype=np.float64)
    M = (np.outer(V, V) - np.eye(3)) @ Rz
    return M

# ---------------- spherical grid (ERP-like), H=W=2*bandwidth ----------------
def get_projection_grid(bandwidth: int):
    H = W = 2 * bandwidth
    L = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float64)   # 经度
    P = np.linspace( np.pi/2, -np.pi/2, H, dtype=np.float64)              # 纬度
    L, P = np.meshgrid(L, P)
    # 正确的球面到笛卡尔
    x = np.cos(P) * np.cos(L)
    y = np.cos(P) * np.sin(L)
    z = np.sin(P)
    return x, y, z

def rotate_grid(rot: np.ndarray, grid):
    x, y, z = grid
    xyz = np.array((x, y, z))  # [3,H,W]
    x_r, y_r, z_r = np.einsum('ij,jab->iab', rot, xyz)
    return x_r, y_r, z_r

# ---------------- project to plane z=0 from a perspective origin, normalize to [0,1]^2 ----------------
def project_sphere_on_xy_plane(grid, projection_origin=(0.0, 0.0, 2.0)):
    sx, sy, sz = projection_origin
    x, y, z = grid
    z_shift = z + 1.0

    t  = -z_shift / (z_shift - sz + 1e-7)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 0.5 * (-1 - sx) + -1
    ymin = 0.5 * (-1 - sy) + -1

    rx = (qx - xmin) / (2 * abs(xmin))
    ry = (qy - ymin) / (2 * abs(ymin))
    return rx, ry

# ---------------- bilinear sampler ----------------
def bilinear_sample_xrow(gray_img: np.ndarray, xs_row: np.ndarray, ys_col: np.ndarray) -> np.ndarray:
    """
    gray_img: [H0, W0]
    xs_row:   用于“行”的坐标（由 x 归一化映射而来）  -> 与 H0 尺寸匹配
    ys_col:   用于“列”的坐标（由 y 归一化映射而来）  -> 与 W0 尺寸匹配
    返回与 xs_row/ys_col 同形的采样值
    """
    H0, W0 = gray_img.shape

    r0 = np.floor(xs_row).astype(np.int64)         # row 
    c0 = np.floor(ys_col).astype(np.int64)         # col 
    r1 = np.clip(r0 + 1, 0, H0 - 1); r0 = np.clip(r0, 0, H0 - 1)
    c1 = np.clip(c0 + 1, 0, W0 - 1); c0 = np.clip(c0, 0, W0 - 1)

    wx = xs_row - r0   
    wy = ys_col - c0   

    I00 = gray_img[r0, c0]; I10 = gray_img[r1, c0]
    I01 = gray_img[r0, c1]; I11 = gray_img[r1, c1]

    top = I00 * (1 - wx) + I10 * wx   #
    bot = I01 * (1 - wx) + I11 * wx
    val = top * (1 - wy) + bot * wy   #
    return val.astype(np.float64)

# ---------------- project a batch onto sphere (south hemisphere) ----------------
def project_2d_on_sphere(signals: np.ndarray, grid):
    """
    signals: [B, Hs, Ws] float64
    grid:    (x,y,z) each [H,W]
    return:  uint8 [B,H,W]
    """
    x, y, z = grid
    H, W = x.shape
    rx, ry = project_sphere_on_xy_plane(grid, projection_origin=(0, 0, 2.0))

    mask = (z <= 0)
    K = int(mask.sum())
    if K == 0:
        return np.zeros((signals.shape[0], H, W), dtype=np.uint8)

    Hs, Ws = signals.shape[1], signals.shape[2]
    rx_use = np.clip(rx, 0.0, 1.0 - 1e-8)
    ry_use = np.clip(ry, 0.0, 1.0 - 1e-8)

    rows = rx_use[mask] * (Hs - 1)
    cols = ry_use[mask] * (Ws - 1)

    out = np.zeros((signals.shape[0], H * W), dtype=np.float64)
    flat_mask = mask.ravel()
    for i in range(signals.shape[0]):
        vals = bilinear_sample_xrow(signals[i], rows, cols)  # [K]
        tmp = np.zeros((H * W,), dtype=np.float64)
        tmp[flat_mask] = vals
        out[i] = tmp
    out = out.reshape(signals.shape[0], H, W)

    vmin = out.reshape(signals.shape[0], -1).min(axis=1)[:, None, None]
    vmax = out.reshape(signals.shape[0], -1).max(axis=1)[:, None, None]
    denom = np.clip(vmax - vmin, 1e-6, None)
    out = (out - vmin) / denom
    return (out * 255.0).astype(np.uint8)

# ---------------- save helper ----------------
def save_split_npy(out_dir, images_u8: np.ndarray, labels_i64: np.ndarray, out_dtype: str):
    """
    images_u8: [N,H,W] uint8
    labels_i64: [N]
    写出:
      out_dir/data.npy  -> [N,1,H,W]（float32:[0,1] 或 uint8:[0..255]）
      out_dir/label.npy -> [N]
    """
    os.makedirs(out_dir, exist_ok=True)
    if out_dtype.lower() == "float32":
        data = images_u8.astype(np.float32) / 255.0
    elif out_dtype.lower() == "uint8":
        data = images_u8
    else:
        raise ValueError("out_dtype must be 'float32' or 'uint8'")
    data = data[:, None, :, :]
    np.save(os.path.join(out_dir, "images.npy"), data)
    np.save(os.path.join(out_dir, "labels.npy"), labels_i64.astype(np.int64))

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Spherical MNIST from IDX (no lie_learn), output .npy")

    parser.add_argument("--train_images", type=str, required=True, help="train-images-idx3-ubyte")
    parser.add_argument("--train_labels", type=str, required=True, help="train-labels-idx1-ubyte")
    parser.add_argument("--test_images",  type=str, required=True, help="t10k-images-idx3-ubyte")
    parser.add_argument("--test_labels",  type=str, required=True, help="t10k-labels-idx1-ubyte")

    parser.add_argument("--out_dir", type=str, default="./sph_mnist_out_npy")
    parser.add_argument("--out_dtype", type=str, choices=["float32", "uint8"], default="uint8")

    parser.add_argument("--bandwidth", type=int, default=32, help="output H=W=2*bandwidth")
    parser.add_argument("--noise", type=float, default=1.0, help="rotational noise 0..1")
    parser.add_argument("--chunk_size", type=int, default=1, help="samples per rotation")
    parser.add_argument("--no_rotate_train", dest="no_rotate_train", action="store_true")
    parser.add_argument("--no_rotate_test",  dest="no_rotate_test",  action="store_true")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 读取 IDX
    print("Loading MNIST IDX ...")
    trX_u8, trY = load_mnist_idx(args.train_images, args.train_labels)
    teX_u8, teY = load_mnist_idx(args.test_images,  args.test_labels)

    # 生成球面网格
    grid = get_projection_grid(args.bandwidth)
    H = W = 2 * args.bandwidth

    def process_split(name: str, imgs_u8: np.ndarray, labs: np.ndarray):
        print(f"Projecting {name} ...")
        N = imgs_u8.shape[0]
        projections_u8 = np.zeros((N, H, W), dtype=np.uint8)
        current = 0
        while current < N:
            end = min(N, current + args.chunk_size)

            # 本 chunk 的球面旋转
            if (name == "train" and not args.no_rotate_train) or (name == "test" and not args.no_rotate_test):
                rot = rand_rotation_matrix(deflection=args.noise)
                rotated_grid = rotate_grid(rot, grid)
            else:
                rotated_grid = grid

            # 投影（批处理）
            batch = imgs_u8[current:end].astype(np.float64)  # [B,28,28]
            proj = project_2d_on_sphere(batch, rotated_grid) # [B,H,W] uint8
            projections_u8[current:end] = proj

            current = end
            print(f"\r{name}: {current}/{N}", end="")
        print("")
        return projections_u8, labs.astype(np.int64)

    # 处理并保存
    train_imgs_u8, train_labels = process_split("train", trX_u8, trY)
    test_imgs_u8,  test_labels  = process_split("test",  teX_u8, teY)

    save_split_npy(os.path.join(args.out_dir, "train"), train_imgs_u8, train_labels, args.out_dtype)
    save_split_npy(os.path.join(args.out_dir, "test"),  test_imgs_u8,  test_labels,  args.out_dtype)

    print("Saved .npy to:", args.out_dir)

if __name__ == "__main__":
    main()
