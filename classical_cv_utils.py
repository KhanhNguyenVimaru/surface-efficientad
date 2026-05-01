"""
classical_cv_utils.py

Module tích hợp các kỹ thuật xử lý ảnh truyền thống (Classical CV)
vào pipeline Deep Learning (EfficientAD) cho kiểm tra ngoại quan.

Các kỹ thuật hỗ trợ:
- Morphological Operations: Opening, Closing, Hit-or-Miss
- Contour Analysis: area, perimeter, aspect ratio, solidity
- Post-processing anomaly map từ EfficientAD
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DefectFeature:
    """Đặc trưng hình học của một vùng lỗi (defect)."""
    contour_id: int
    area: float
    perimeter: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    aspect_ratio: float
    extent: float
    solidity: float
    equivalent_diameter: float
    centroid: Tuple[float, float]
    orientation: Optional[float] = None
    defect_type_hint: str = "unknown"


def apply_opening(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Áp dụng Morphological Opening để xóa nhiễu điểm nhỏ (salt noise).

    Opening = Erosion followed by Dilation
    - Loại bỏ các đốm nhiễu nhỏ hơn kernel
    - Giữ nguyên hình dạng và kích thước các vùng lớn
    - Làm mịn biên vùng lỗi

    Args:
        mask: Ảnh nhị phân (0/255 hoặc 0/1)
        kernel_size: Kích thước kernel (mặc định 3×3 ellipse)

    Returns:
        Ảnh nhị phân đã làm sạch
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened


def apply_closing(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Áp dụng Morphological Closing để lấp lỗ hổng nhỏ.

    Closing = Dilation followed by Erosion
    - Nối liền các vùng lỗi bị đứt quãng
    - Lấp đầy lỗ nhỏ trong vùng defect
    - Giữ nguyên kích thước tổng thể

    Args:
        mask: Ảnh nhị phân (0/255 hoặc 0/1)
        kernel_size: Kích thước kernel (mặc định 3×3 ellipse)

    Returns:
        Ảnh nhị phân đã làm sạch
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed


def apply_opening_closing(mask: np.ndarray, open_k: int = 3, close_k: int = 3) -> np.ndarray:
    """
    Kết hợp Opening → Closing để làm sạch mask hoàn chỉnh.

    Pipeline:
        mask → Opening (xóa nhiễu) → Closing (nối đứt quãng) → clean_mask

    Args:
        mask: Ảnh nhị phân
        open_k: Kernel size cho Opening
        close_k: Kernel size cho Closing

    Returns:
        Mask đã được làm sạch toàn diện
    """
    step1 = apply_opening(mask, kernel_size=open_k)
    step2 = apply_closing(step1, kernel_size=close_k)
    return step2


def apply_hit_or_miss(image: np.ndarray, pattern_type: str = "top_left_corner") -> np.ndarray:
    """
    Áp dụng Hit-or-Miss Transform để phát hiện pattern cụ thể.

    Hit-or-Miss giúp tìm các cấu trúc hình thái có hình dạng xác định
    trong ảnh nhị phân (ví dụ: góc cạnh, điểm uốn).

    Args:
        image: Ảnh nhị phân (0/255)
        pattern_type: Loại pattern cần tìm:
            - "top_left_corner": Góc trên-trái
            - " isolated_point": Điểm cô lập
            - "spur": Điểm nhô ra (spur/branch)

    Returns:
        Ảnh nhị phân với các điểm tìm được đánh dấu 255
    """
    if image.dtype != np.uint8:
        image = (image > 0).astype(np.uint8) * 255

    if pattern_type == "top_left_corner":
        # Tìm góc trên-trái (foreground ở góc, background xung quanh)
        kernel = np.array([
            [-1,  1,  1],
            [-1,  1,  1],
            [-1, -1, -1]
        ], dtype=np.int8)
    elif pattern_type == "isolated_point":
        # Tìm điểm cô lập (1 điểm trắng bao quanh bởi đen)
        kernel = np.array([
            [-1, -1, -1],
            [-1,  1, -1],
            [-1, -1, -1]
        ], dtype=np.int8)
    elif pattern_type == "spur":
        # Tìm điểm nhô ra (spur)
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  1, -1],
            [-1,  1, -1]
        ], dtype=np.int8)
    else:
        # Mặc định: cross pattern
        kernel = np.array([
            [-1,  1, -1],
            [ 1,  1,  1],
            [-1,  1, -1]
        ], dtype=np.int8)

    result = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    return result


def analyze_contours(mask: np.ndarray, min_area: float = 50.0) -> List[DefectFeature]:
    """
    Phân tích tất cả contour trong mask và trích xuất đặc trưng hình học.

    Args:
        mask: Ảnh nhị phân (0/255 hoặc bool)
        min_area: Diện tích tối thiểu để được coi là defect (loại bỏ nhiễu)

    Returns:
        Danh sách các DefectFeature
    """
    if mask.dtype != np.uint8:
        binary = (mask > 0).astype(np.uint8)
    else:
        binary = mask.copy()

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,          # Chỉ lấy contour ngoài cùng
        cv2.CHAIN_APPROX_SIMPLE     # Nén các điểm thẳng hàng
    )

    features: List[DefectFeature] = []

    for idx, cnt in enumerate(contours):
        area = float(cv2.contourArea(cnt))

        if area < min_area:
            continue  # Bỏ qua nhiễu nhỏ

        perimeter = float(cv2.arcLength(cnt, True))
        x, y, w, h = cv2.boundingRect(cnt)

        # Aspect ratio
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0

        # Extent = area / bbox_area
        bbox_area = float(w * h)
        extent = area / bbox_area if bbox_area > 0 else 0.0

        # Solidity = area / convex_hull_area
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0

        # Equivalent diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi)

        # Centroid (moments)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx, cy = float(x + w / 2), float(y + h / 2)

        # Orientation (fit ellipse nếu contour đủ lớn)
        orientation = None
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                orientation = float(ellipse[2])  # Góc nghiêng độ
            except cv2.error:
                pass

        # Heuristic phân loại defect type
        defect_hint = classify_defect_by_shape(
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            area=area,
            equivalent_diameter=equivalent_diameter
        )

        feature = DefectFeature(
            contour_id=idx,
            area=area,
            perimeter=perimeter,
            bbox=(x, y, w, h),
            aspect_ratio=aspect_ratio,
            extent=extent,
            solidity=solidity,
            equivalent_diameter=equivalent_diameter,
            centroid=(cx, cy),
            orientation=orientation,
            defect_type_hint=defect_hint
        )
        features.append(feature)

    # Sắp xếp theo diện tích giảm dần
    features.sort(key=lambda f: f.area, reverse=True)
    return features


def classify_defect_by_shape(
    aspect_ratio: float,
    solidity: float,
    area: float,
    equivalent_diameter: float
) -> str:
    """
    Phân loại loại lỗi dựa trên đặc trưng hình học (heuristic).

    Args:
        aspect_ratio: Tỷ lệ dài/rộng (w/h)
        solidity: Độ đặc (0-1)
        area: Diện tích (pixels²)
        equivalent_diameter: Đường kính tương đương (pixels)

    Returns:
        Chuỗi gợi ý loại defect
    """
    if aspect_ratio > 4.0:
        return "scratch"          # Vết xước dài
    elif solidity < 0.75:
        return "crack_or_poke"    # Lõm, nứt (có phần lõm)
    elif equivalent_diameter > 25:
        return "squeeze"          # Ép biến dạng (vùng lớn)
    elif 0.85 < solidity < 0.98 and aspect_ratio < 1.5:
        return "faulty_imprint"   # Lỗi in mờ (hình dạng gần ellipse)
    else:
        return "unknown"


def draw_contours_on_image(
    image: np.ndarray,
    features: List[DefectFeature],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_info: bool = True
) -> np.ndarray:
    """
    Vẽ bounding box và thông tin contour lên ảnh.

    Args:
        image: Ảnh gốc (RGB)
        features: Danh sách DefectFeature
        color: Màu vẽ (B, G, R)
        thickness: Độ dày đường viền
        show_info: Có hiển thị text thông tin không

    Returns:
        Ảnh đã vẽ
    """
    output = image.copy()

    for feat in features:
        x, y, w, h = feat.bbox

        # Vẽ bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

        if show_info:
            # Vẽ label
            label = f"#{feat.contour_id} {feat.defect_type_hint}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                output,
                (x, y - label_size[1] - 4),
                (x + label_size[0], y),
                color,
                -1
            )
            cv2.putText(
                output, label, (x, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            # Vẽ thông số bên dưới
            info = f"A:{int(feat.area)} AR:{feat.aspect_ratio:.1f} S:{feat.solidity:.2f}"
            cv2.putText(
                output, info, (x, y + h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

    return output


def postprocess_anomaly_map(
    anomaly_map: np.ndarray,
    threshold: float = 0.5,
    open_kernel: int = 3,
    close_kernel: int = 3,
    min_contour_area: float = 50.0
) -> Tuple[np.ndarray, List[DefectFeature]]:
    """
    Pipeline post-processing hoàn chỉnh cho anomaly map từ EfficientAD.

    Các bước:
        1. Threshold anomaly map → binary mask
        2. Morphological Opening (xóa nhiễu)
        3. Morphological Closing (nối đứt quãng)
        4. Contour Analysis (trích đặc trưng hình học)

    Args:
        anomaly_map: Heatmap từ EfficientAD (giá trị float 0-1 hoặc 0-255)
        threshold: Ngưỡng nhị phân hóa
        open_kernel: Kernel size cho Opening
        close_kernel: Kernel size cho Closing
        min_contour_area: Diện tích tối thiểu contour

    Returns:
        (cleaned_mask, list_defect_features)
    """
    # Bước 1: Thresholding
    if anomaly_map.max() <= 1.0:
        binary = (anomaly_map >= threshold).astype(np.uint8) * 255
    else:
        # Nếu map đã chuẩn hóa 0-255
        binary = (anomaly_map >= threshold * 255).astype(np.uint8) * 255

    # Bước 2 & 3: Morphological cleanup
    cleaned = apply_opening_closing(binary, open_k=open_kernel, close_k=close_kernel)

    # Bước 4: Contour analysis
    features = analyze_contours(cleaned, min_area=min_contour_area)

    return cleaned, features


def example_usage():
    """
    Ví dụ minh họa cách sử dụng module với anomaly map từ EfficientAD.
    """
    import matplotlib.pyplot as plt

    # Giả lập anomaly map (trong thực tế lấy từ result.anomaly_map)
    h, w = 256, 256
    anomaly_map = np.zeros((h, w), dtype=np.float32)

    # Tạo vết xước giả lập
    cv2.line(anomaly_map, (80, 100), (180, 120), 0.8, 3)
    # Thêm nhiễu
    noise = np.random.rand(h, w) * 0.3
    anomaly_map = np.clip(anomaly_map + noise, 0, 1)

    # Post-processing
    cleaned_mask, defects = postprocess_anomaly_map(
        anomaly_map,
        threshold=0.5,
        open_kernel=3,
        close_kernel=3,
        min_contour_area=30
    )

    print(f"Tìm thấy {len(defects)} defect(s):")
    for d in defects:
        print(f"  - #{d.contour_id}: {d.defect_type_hint}, "
              f"area={d.area:.0f}px², aspect_ratio={d.aspect_ratio:.2f}, "
              f"solidity={d.solidity:.3f}")

    # Hiển thị
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(anomaly_map, cmap='jet')
    axes[0].set_title('Anomaly Map (DL)')
    axes[0].axis('off')

    axes[1].imshow(cleaned_mask, cmap='gray')
    axes[1].set_title('Cleaned Mask (Classical CV)')
    axes[1].axis('off')

    axes[2].imshow(anomaly_map, cmap='jet')
    for d in defects:
        x, y, w, h = d.bbox
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=2)
        axes[2].add_patch(rect)
        axes[2].text(x, y - 5, f"#{d.contour_id}", color='lime', fontsize=10)
    axes[2].set_title(f'Contours Detected: {len(defects)}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('classical_cv_demo.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    example_usage()
