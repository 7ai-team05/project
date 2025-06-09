

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_drain(image_path):
    # 1) HSV/BGR thresholds
    brown_lower = np.array([0,  20,  10])   # H 0–70, S 20–255, V 10–255
    brown_upper = np.array([70, 255, 255])
    gray_lower  = np.array([0,   0,  10])   # H 0–180, S 0–100, V 10–255
    gray_upper  = np.array([180,100,255])
    pack_lower  = np.array([0,   80,  50])  # packaging: high saturation
    pack_upper  = np.array([180,255,255])
    white_lower = np.array([200,200,200])   # cigarette: white BGR
    white_upper = np.array([255,255,255])

    # 2) Load ROI image
    roi = cv2.imread(image_path)
    
    if roi is None:
        print(f"Cannot load image: {image_path}")
        return

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 3) Build masks
    mask_brown = cv2.inRange(hsv, brown_lower, brown_upper)
    mask_gray  = cv2.inRange(hsv, gray_lower,  gray_upper)
    soil_mask  = cv2.bitwise_or(mask_brown, mask_gray)

    pack_mask  = cv2.inRange(hsv, pack_lower, pack_upper)
    pack_mask  = cv2.bitwise_and(pack_mask, cv2.bitwise_not(soil_mask))

    white_mask  = cv2.inRange(roi, white_lower, white_upper)
    white_clean = cv2.morphologyEx(white_mask,
                                   cv2.MORPH_OPEN,
                                   np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(white_clean,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cig_mask = np.zeros_like(white_clean)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if h > w*3 and w*h > 100:
            cv2.drawContours(cig_mask, [c], -1, 255, cv2.FILLED)

    # 4) Compute combined ratio
    h_, w_ = roi.shape[:2]
    total_pixels = h_ * w_
    soil_pixels = cv2.countNonZero(soil_mask)
    pack_pixels = cv2.countNonZero(pack_mask)
    cig_pixels  = cv2.countNonZero(cig_mask)
    combined_ratio = (soil_pixels + pack_pixels + cig_pixels) / total_pixels * 100

    print(f"Combined contamination ratio: {combined_ratio:.1f}%")

    # 5) Show overlay
    overlay = roi.copy()
    overlay[soil_mask>0] = (  0,165,255)  # soil: orange
    overlay[pack_mask>0] = (255,  0,  0)  # pack: blue
    overlay[cig_mask>0]  = (  0,  0,255)  # cig: red

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Soil/Pack/Cig Overlay")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze drain ROI")
    parser.add_argument("image", help="Path to cropped ROI image")
    args = parser.parse_args()

    analyze_drain(args.image)

if __name__ == "__main__":
    main()
