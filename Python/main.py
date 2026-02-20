import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 0) Load image

img_path = "original_img.JPG"
img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)  # (rows, cols, 3)

rows, cols, ch = img.shape
N = rows * cols


def entropy_manual(channel_uint8: np.ndarray) -> float:
    counts = np.bincount(channel_uint8.ravel(), minlength=256).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def adjacent_corr_horizontal(channel_uint8: np.ndarray) -> float:
    A = channel_uint8.astype(np.float64)
    x = A[:, :-1].ravel()
    y = A[:,  1:].ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    return 0.0 if denom == 0 else float((x * y).sum() / denom)

def scatter_adjacent(channel_uint8: np.ndarray, n_points: int, rng: np.random.Generator):
    A = channel_uint8.astype(np.float64)
    x = A[:, :-1].ravel()
    y = A[:,  1:].ravel()
    M = x.size
    n = min(n_points, M)
    idx = rng.permutation(M)[:n]
    plt.plot(x[idx], y[idx], ".", markersize=2)

def plot_hist_color(channel_uint8: np.ndarray, color):
    counts = np.bincount(channel_uint8.ravel(), minlength=256)
    plt.bar(np.arange(256), counts, color=color, edgecolor=color, linewidth=0.2)
    plt.xlim(0, 255)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")


# 1) Original Image + Histogram

plt.figure("Original Image")
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")

plt.figure("Histogram of Original Image")
plt.subplot(2, 2, 1)
plt.imshow(img); plt.axis("off"); plt.title("Original Image")

plt.subplot(2, 2, 2)
plot_hist_color(img[:, :, 0], (1, 0, 0))
plt.title("Histogram - Red")

plt.subplot(2, 2, 3)
plot_hist_color(img[:, :, 1], (0, 1, 0))
plt.title("Histogram - Green")

plt.subplot(2, 2, 4)
plot_hist_color(img[:, :, 2], (0, 0, 1))
plt.title("Histogram - Blue")


# 2) Confusion Stage (Scrambling Logistic Map)

r = 3.99
x0_scramble = 0.7

img1d = img.reshape(-1, ch)  # (N, 3)

seq = np.empty(N, dtype=np.float64)
x = x0_scramble
for i in range(N):
    x = r * x * (1 - x)
    seq[i] = x

idx = np.argsort(seq)  # permutation indices
scrambled = img1d[idx, :]
scrambled_img = scrambled.reshape(rows, cols, ch).astype(np.uint8)

plt.figure("Scrambled Image - Confusion Stage")
plt.imshow(scrambled_img)
plt.axis("off")
plt.title("Scrambled Image (Confusion)")


# 3) Encrypted Image - Diffusion Stage (XOR on bits)

x0_diffusion = 0.5
x = x0_diffusion

flat_vals = scrambled_img.ravel()  # uint8 array length = rows*cols*3

# Convert to bits (big-endian like dec2bin 8)
bits = np.unpackbits(flat_vals, bitorder="big")  # shape (totalBits,)
total_bits = bits.size

chaos_bits = np.empty(total_bits, dtype=np.uint8)
for i in range(total_bits):
    x = r * x * (1 - x)
    chaos_bits[i] = (int(np.floor(x * 1e14)) & 1)

xored_bits = np.bitwise_xor(bits, chaos_bits).astype(np.uint8)
dec_vals = np.packbits(xored_bits, bitorder="big")  # back to bytes

encrypted_img = dec_vals.reshape(rows, cols, ch).astype(np.uint8)

plt.figure("Encrypted Image - Diffusion Stage")
plt.imshow(encrypted_img)
plt.axis("off")
plt.title("Encrypted Image (Diffusion)")


# 5) Histogram After Encryption

plt.figure("Histogram After Encryption")
plt.subplot(1, 3, 1)
plot_hist_color(encrypted_img[:, :, 0], (1, 0, 0))
plt.title("Encrypted Histogram - Red")

plt.subplot(1, 3, 2)
plot_hist_color(encrypted_img[:, :, 1], (0, 1, 0))
plt.title("Encrypted Histogram - Green")

plt.subplot(1, 3, 3)
plot_hist_color(encrypted_img[:, :, 2], (0, 0, 1))
plt.title("Encrypted Histogram - Blue")


# 6) Side-by-Side Comparison

plt.figure("Side-by-Side Comparison", figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(img); plt.axis("off"); plt.title("Original Image")
plt.subplot(1, 3, 2); plt.imshow(scrambled_img); plt.axis("off"); plt.title("Scrambled Image (Confusion)")
plt.subplot(1, 3, 3); plt.imshow(encrypted_img); plt.axis("off"); plt.title("Encrypted Image (Diffusion)")

# 7) Entropy Analysis (Per Channel)

H_R_orig = entropy_manual(img[:, :, 0])
H_G_orig = entropy_manual(img[:, :, 1])
H_B_orig = entropy_manual(img[:, :, 2])

H_R_enc  = entropy_manual(encrypted_img[:, :, 0])
H_G_enc  = entropy_manual(encrypted_img[:, :, 1])
H_B_enc  = entropy_manual(encrypted_img[:, :, 2])

print("\n7. Entropy Analysis (Per Channel)")
print(f"Original Entropy  (R,G,B) = {H_R_orig:.5f}  {H_G_orig:.5f}  {H_B_orig:.5f}")
print(f"Encrypted Entropy (R,G,B) = {H_R_enc:.5f}  {H_G_enc:.5f}  {H_B_enc:.5f}")


# 8) Correlation Analysis


def adjacent_pairs(channel_uint8, direction):
    A = channel_uint8.astype(np.float64)
    if direction == "horizontal":
        x = A[:, :-1].ravel()
        y = A[:,  1:].ravel()
    elif direction == "vertical":
        x = A[:-1, :].ravel()
        y = A[ 1:, :].ravel()
    elif direction == "diagonal":
        x = A[:-1, :-1].ravel()
        y = A[ 1:,  1:].ravel()
    return x, y

def adjacent_corr(channel_uint8, direction):
    x, y = adjacent_pairs(channel_uint8, direction)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    return 0 if denom == 0 else (x*y).sum() / denom

def scatter_direction(channel_uint8, direction, color, title_text):
    rng = np.random.default_rng(1)
    x, y = adjacent_pairs(channel_uint8, direction)
    idx = rng.permutation(len(x))[:8000]
    plt.scatter(x[idx], y[idx], s=6, c=color, alpha=0.35)
    r = adjacent_corr(channel_uint8, direction)
    plt.title(f"{title_text}\nr = {r:.4f}")
    plt.xlabel("Pixel(i)")
    plt.ylabel("Neighbor")
    plt.xlim(0,255)
    plt.ylim(0,255)
    plt.grid(True, alpha=0.3)

directions = ["horizontal", "vertical", "diagonal"]
dir_titles = ["Horizontal", "Vertical", "Diagonal"]

channels = [
    ("Red",   0, "red"),
    ("Green", 1, "green"),
    ("Blue",  2, "blue")
]

for name, idx, color in channels:
    
    plt.figure(f"Correlation - {name} Channel (Before & After)", figsize=(12,6))
    
    # ===== Original (Row 1)
    for i, d in enumerate(directions):
        plt.subplot(2,3,i+1)
        scatter_direction(img[:,:,idx], d, color, f"Original - {dir_titles[i]}")
    
    # ===== Encrypted (Row 2)
    for i, d in enumerate(directions):
        plt.subplot(2,3,i+4)
        scatter_direction(encrypted_img[:,:,idx], d, color, f"Encrypted - {dir_titles[i]}")
    
    plt.tight_layout()

plt.show()