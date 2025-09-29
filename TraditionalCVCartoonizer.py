import cv2
import numpy as np
import streamlit as st

# ====================================================
# 📌 Utility functions (scratch implementations)
# ====================================================

def to_gray(img):
    
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def median_blur(img, k=3):

    pad = k // 2
    padded = np.pad(img, pad, mode='edge')
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+k, j:j+k].flatten()
            out[i, j] = np.median(region)
    return out

def sobel_edges(img_gray, thresh=100):
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    Gx = cv2.filter2D(img_gray, -1, Kx)
    Gy = cv2.filter2D(img_gray, -1, Ky)
    mag = np.sqrt(Gx**2 + Gy**2).astype(np.uint8)
    _, edge = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY_INV)
    return edge

def kmeans_quantize(img, k=8, max_iter=10):
    data = img.reshape((-1, 3)).astype(np.float32)
    # Randomly pick centers
    centers = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(max_iter):
        # Compute distances
        dists = np.linalg.norm(data[:, None] - centers[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Update centers
        new_centers = np.array([data[labels==i].mean(axis=0) if np.any(labels==i) else centers[i] for i in range(k)])
        if np.allclose(centers, new_centers, atol=1.0):
            break
        centers = new_centers
    quantized = centers[labels].reshape(img.shape).astype(np.uint8)
    return quantized

def boost_saturation_contrast(img, sat_mult=1.3, val_mult=1.2):
    img = img.astype(np.float32) / 255.0
    maxc = img.max(axis=2)
    minc = img.min(axis=2)
    v = maxc
    s = (maxc - minc) / (maxc + 1e-6)
    h = np.zeros_like(v)

    # Hue calc 
    mask = (maxc == img[...,0])
    h[mask] = (60 * (img[...,1][mask] - img[...,2][mask]) / (maxc[mask] - minc[mask] + 1e-6)) % 360
    mask = (maxc == img[...,1])
    h[mask] = (120 + 60 * (img[...,2][mask] - img[...,0][mask]) / (maxc[mask] - minc[mask] + 1e-6)) % 360
    mask = (maxc == img[...,2])
    h[mask] = (240 + 60 * (img[...,0][mask] - img[...,1][mask]) / (maxc[mask] - minc[mask] + 1e-6)) % 360

    # Boost
    s = np.clip(s * sat_mult, 0, 1)
    v = np.clip(v * val_mult, 0, 1)

    # Convert back (HSV → RGB )
    c = v * s
    x = c * (1 - np.abs(((h / 60) % 2) - 1))
    m = v - c
    rgb = np.zeros_like(img)

    h_idx = (h < 60)
    rgb[h_idx] = np.stack([c[h_idx], x[h_idx], np.zeros_like(c[h_idx])], axis=1)
    h_idx = (h >= 60) & (h < 120)
    rgb[h_idx] = np.stack([x[h_idx], c[h_idx], np.zeros_like(c[h_idx])], axis=1)
    h_idx = (h >= 120) & (h < 180)
    rgb[h_idx] = np.stack([np.zeros_like(c[h_idx]), c[h_idx], x[h_idx]], axis=1)
    h_idx = (h >= 180) & (h < 240)
    rgb[h_idx] = np.stack([np.zeros_like(c[h_idx]), x[h_idx], c[h_idx]], axis=1)
    h_idx = (h >= 240) & (h < 300)
    rgb[h_idx] = np.stack([x[h_idx], np.zeros_like(c[h_idx]), c[h_idx]], axis=1)
    h_idx = (h >= 300)
    rgb[h_idx] = np.stack([c[h_idx], np.zeros_like(c[h_idx]), x[h_idx]], axis=1)

    rgb = (rgb + m[...,None]) * 255
    return rgb.astype(np.uint8)

# ====================================================
# 📌 Cartoonizer Pipeline
# ====================================================

def cartoonize(img, mode="Cartoon"):
    gray = to_gray(img)
    gray_blur = median_blur(gray, 5)
    edges = sobel_edges(gray_blur, thresh=80)
    quantized = kmeans_quantize(img, k=12, max_iter=15)

    # Combine edges with quantized
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(quantized, edges_colored)

    if mode == "Anime":
        cartoon = boost_saturation_contrast(cartoon, sat_mult=1.5, val_mult=1.3)
    return cartoon

# ====================================================
# 📌 Streamlit UI
# ====================================================

st.set_page_config(page_title="Cartoonizer (Scratch CV)", layout="wide")
st.title("🎨 Cartoonizer App (Scratch CV Only)")

mode = st.radio("Choose Mode", ["Cartoon", "Anime"])

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
use_camera = st.checkbox("Use Live Camera")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
    cartoon = cartoonize(img, mode=mode)
    st.image(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB), caption=f"{mode} Style")

elif use_camera:
    run = st.checkbox("Start Camera")
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        cartoon = cartoonize(frame, mode=mode)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original")
        with col2:
            st.image(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB), caption=f"{mode} Style")

    camera.release()
else:
    st.info("👆 Upload an image or enable 'Use Live Camera'")
