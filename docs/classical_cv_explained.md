# Hệ thống kiểm tra ngoại quan (Quality Control) - Giải thích chi tiết

> **Áp dụng các kỹ thuật biến đổi hình thái học và phân tích viền trong phát hiện lỗi bề mặt**

---

## 1. Tổng quan về Xử lý ảnh trong QC

Trong kiểm tra ngoại quan công nghiệp, có hai phương pháp chính:

| Phương pháp | Đặc điểm | Độ chính xác | Tốc độ |
|-------------|----------|--------------|--------|
| **Classical CV** (truyền thống) | Dựa trên luật toán học, kernel cố định | Trung bình | Rất nhanh |
| **Deep Learning** (EfficientAD) | Học đặc trưng từ dữ liệu | Cao | Nhanh |

**Kết hợp cả hai** (Hybrid Approach) là giải pháp tối ưu trong thực tế:
- **DL** phát hiện vị trí nghi ngờ (anomaly map)
- **Classical CV** làm sạch và đo lường chính xác vùng lỗi

---

## 2. Biến đổi Hình thái học (Mathematical Morphology)

### 2.1. Cơ sở lý thuyết

Biến đổi hình thái học dựa trên **phép toán tập hợp** trên ảnh nhị phân/đa mức xám. Các phép toán cơ bản:

#### a) Phép giãn (Dilation) — "Làm phình to"
```
A ⊕ B = { z | (B̂)_z ∩ A ≠ ∅ }
```
- Làm các vùng trắng (foreground) **phình to ra**
- Lấp đầy lỗ hổng nhỏ, nối liền các vùng gần nhau

#### b) Phép co (Erosion) — "Làm teo nhỏ"
```
A ⊖ B = { z | B_z ⊆ A }
```
- Làm các vùng trắng **teo lại**
- Xóa nhiễu điểm nhỏ, tách các vùng dính nhau

**Kernel (Structuring Element):** Hình dạng quét qua ảnh (thường dùng ellipse 3×3 hoặc 5×5).

---

### 2.2. Opening — "Mở"

```python
Opening(A) = Dilation(Erosion(A))
#           = (A ⊖ B) ⊕ B
```

**Hiệu ứng:**
- **Xóa nhiễu điểm nhỏ** (salt noise) mà không làm thay đổi kích thước đáng kể các vùng lớn
- Tách các vùng nối liền bằng cầu mỏng
- **Làm mịn biên** vùng lỗi

**Ứng dụng QC:**
- Loại bỏ các điểm nhiễu trên anomaly map do DL dự đoán sai lung tung
- Tách các vết xước gần nhau thành đối tượng riêng biệt

```python
import cv2
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
```

---

### 2.3. Closing — "Đóng"

```python
Closing(A) = Erosion(Dilation(A))
#           = (A ⊕ B) ⊖ B
```

**Hiệu ứng:**
- **Lấp đầy lỗ hổng nhỏ** trong vùng lỗi (vết xước bị đứt quãng)
- Nối liền các thành phần gần nhau
- **Giữ nguyên kích thước** vùng lớn

**Ứng dụng QC:**
- Vết xước dài bị DL phát hiện thành nhiều đoạn rỗng → Closing nối lại thành 1 vết liền mạch
- Lấp lỗ nhỏ trong vùng lỗi lõm (poke)

```python
closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
```

---

### 2.4. Opening → Closing (Làm sạch hoàn chỉnh)

Trong thực tế, ta thường kết hợp **cả hai** để có mask sạch nhất:

```
Ảnh gốc
  ↓
Opening  → Xóa nhiễu điểm nhỏ (salt)
  ↓
Closing  → Lấp lỗ nhỏ trong vùng lỗi (pepper)
  ↓
Mask hoàn chỉnh
```

---

### 2.5. Hit-or-Miss Transform (HMT)

**Định nghĩa:** Phép toán hình thái để **phát hiện pattern cụ thể** (hình dạng, góc cạnh, điểm uốn) trong ảnh nhị phân.

```
A ⊛ B = (A ⊖ B₁) ∩ (Aᶜ ⊖ B₂)
```

Trong đó:
- `B₁`: Kernel tìm foreground (phần muốn có)
- `B₂`: Kernel tìm background (phần muốn không có)

**Ứng dụng QC:**
- Tìm góc vuông bị mẻ trên viên thuốc capsule
- Phát hiện điểm đuôi tôm (spur) trên biên dạng
- Tìm các pattern lỗi có hình dạng cụ thể (ví dụ: lỗ kim nhỏ hình tròn)

```python
# Ví dụ: Tìm góc trên-trái trong ảnh nhị phân
kernel_foreground = np.array([
    [0, 1, 1],
    [0, 1, 1],
    [0, 0, 0]
], dtype=np.int8)
kernel_background = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 1, 1]
], dtype=np.int8)

# Trong OpenCV 4.x
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
result = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
```

> **Lưu ý:** Hit-or-Miss ít dùng trong QC hiện đại vì DL đã thay thế hiệu quả hơn cho việc phát hiện pattern phức tạp. Tuy nhiên, nó vẫn hữu ích cho **các pattern lỗi có hình dạng cố định, đã biết trước**.

---

## 3. Phân tích Viền (Contour Analysis)

### 3.1. Contour là gì?

Contour là **đường bao liên tục** nối các điểm có cùng màu sắc/cường độ (thường là biên của vùng trắng trên ảnh nhị phân).

```python
contours, hierarchy = cv2.findContours(
    mask.astype(np.uint8), 
    cv2.RETR_EXTERNAL,      # Chỉ lấy contour ngoài cùng
    cv2.CHAIN_APPROX_SIMPLE # Nén điểm thẳng hàng
)
```

### 3.2. Các đặc trưng hình học trích xuất từ Contour

| Đặc trưng | Công thức | Ý nghĩa trong QC |
|-----------|-----------|------------------|
| **Area** | `cv2.contourArea(cnt)` | Diện tích vùng lỗi (pixels²) |
| **Perimeter** | `cv2.arcLength(cnt, True)` | Chu vi vùng lỗi |
| **Bounding Box** | `cv2.boundingRect(cnt)` | Hình chữ nhật bao ngoài (x, y, w, h) |
| **Aspect Ratio** | `w / h` | Tỷ lệ dài/rộng (vết xước dài có ratio cao) |
| **Extent** | `Area / (w × h)` | Mức độ lấp đầy bounding box |
| **Solidity** | `Area / ConvexHull_Area` | Độ "đặc" của vùng (lõm có solidity thấp) |
| **Equivalent Diameter** | `√(4 × Area / π)` | Đường kính tương đương hình tròn |
| **Orientation** | `cv2.fitEllipse(cnt)[2]` | Góc nghiêng của vết lỗi |

### 3.3. Ứng dụng phân loại defect trong QC

```
Sau khi có contour của vùng lỗi:

if area < 50 pixels:
    → "Nhiễu nhỏ, bỏ qua"
elif aspect_ratio > 5:
    → "Vết xước dài (scratch)"
elif solidity < 0.8:
    → "Vết lõm (poke/crack)"
elif equivalent_diameter > 30:
    → "Vết nứt lớn hoặc ép biến dạng (squeeze)"
else:
    → "Lỗi in mờ (faulty_imprint)"
```

---

## 4. Pipeline Hybrid: DL + Classical CV

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE KIỂM TRA NGOẠI QUAN              │
└─────────────────────────────────────────────────────────────┘

Bước 1: INPUT
[Ảnh viên thuốc capsule]
         ↓

Bước 2: TIỀN XỬ LÝ (EfficientAD)
[Resize 256×256] → [Normalize]
         ↓

Bước 3: DEEP LEARNING DETECTION
[Teacher Network] ──→ Feature chuẩn
[Student Network] ──→ Feature tái tạo
         ↓
[Sự khác biệt T-S] ──→ Anomaly Map (heatmap)
[Global Max Pooling] ──→ Anomaly Score (0.624)
         ↓

Bước 4: CLASSICAL CV POST-PROCESSING ⭐
[Thresholding]        score ≥ 0.515 → Binary Mask
         ↓
[Morphological Opening]  Xóa nhiễu điểm nhỏ (salt)
         ↓
[Morphological Closing]  Nối vết xước đứt quãng
         ↓
[Contour Analysis]       Đo diện tích, chu vi, aspect ratio
         ↓

Bước 5: DECISION & OUTPUT
[Phân loại defect type]  crack / scratch / poke / squeeze
[Đánh giá mức độ]        Nghiêm trọng / Nhẹ
[Vẽ bounding box]        Overlay lên ảnh gốc
         ↓
[Kết luận]              DEFECT (Score: 0.624, Area: 145 px²)
```

---

## 5. Minh họa bằng hình ảnh

### Trước khi xử lý (Anomaly map thô từ DL):
```
┌────────────────┐
│  ░░░████░░░    │  ← Vết xước chính
│  ░░█░░░░█░     │
│  ░█░░██░░█░    │  ← Nhiễu điểm nhỏ (salt)
│  ░░░░██░░░     │
│     ░█░        │  ← Vết xước bị đứt quãng
│      █         │
└────────────────┘
```

### Sau Opening (xóa nhiễu):
```
┌────────────────┐
│  ░░░████░░░    │
│  ░░█░░░░█░     │
│  ░░░░░░░░░░    │  ← Nhiễu đã bị xóa
│  ░░░░██░░░     │
│     ░█░        │
│      █         │
└────────────────┘
```

### Sau Closing (nối đứt quãng):
```
┌────────────────┐
│  ░░░████░░░    │
│  ░░█░░░░█░     │
│  ░░░░░░░░░░    │
│  ░░░░██░░░     │
│     ░██░       │  ← Đã nối liền
│      ██        │
└────────────────┘
```

---

## 6. So sánh: Khi nào dùng Classical CV vs Deep Learning?

| Tình huống | Classical CV | Deep Learning | Kết hợp |
|-----------|--------------|---------------|---------|
| Lỗi có hình dạng cố định, biết trước | ✅ Tốt | ⚠️ Cần nhiều data | ✅ Tốt nhất |
| Lỗi đa dạng, không lường trước | ❌ Khó | ✅ Tốt | ✅ Tốt nhất |
| Tốc độ xử lý cực nhanh (1000 FPS) | ✅ Rất nhanh | ⚠️ Cần GPU | ⚠️ Tùy config |
| Nền phức tạp, đổi sáng | ❌ Kém | ✅ Tốt | ✅ Tốt nhất |
| Cần giải thích từng bước | ✅ Dễ giải thích | ⚠️ Black box | ✅ Dễ giải thích |
| Dataset nhỏ (< 50 ảnh) | ✅ Không cần train | ❌ Cần nhiều data hơn | ✅ Classical CV đủ |

---

## 7. Code thực tế

Xem file `classical_cv_utils.py` để có implementation đầy đủ với OpenCV.

---

*Document này giải thích lý thuyết đằng sau các kỹ thuật xử lý ảnh truyền thống trong kiểm tra ngoại quan công nghiệp.*
