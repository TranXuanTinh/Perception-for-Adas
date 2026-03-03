<a name="tech-top"></a>

<h1 align="center">CÔNG NGHỆ SỬ DỤNG TRONG DỰ ÁN LANE DETECTION FOR SELF-DRIVING CARS</h1>

<div align="center">
  <i>Tài liệu mô tả chi tiết toàn bộ công nghệ, thuật toán, framework và kiến trúc được sử dụng trong project.</i>
</div>

---

## Mục lục
- [1. Tổng quan tính năng](#1-tổng-quan-tính-năng)
- [2. Thư viện Python sử dụng](#2-thư-viện-python-sử-dụng)
- [3. Mô hình AI / Deep Learning](#3-mô-hình-ai--deep-learning)
  - [3.1. YOLO - Phát hiện vật thể](#31-yolo---phát-hiện-vật-thể)
  - [3.2. Ultra-Fast Lane Detection - Phát hiện làn đường](#32-ultra-fast-lane-detection---phát-hiện-làn-đường)
  - [3.3. ByteTrack - Theo dõi vật thể](#33-bytetrack---theo-dõi-vật-thể)
- [4. Inference Engine & Định dạng Model](#4-inference-engine--định-dạng-model)
- [5. Thuật toán & Kỹ thuật xử lý ảnh](#5-thuật-toán--kỹ-thuật-xử-lý-ảnh)
  - [5.1. Non-Maximum Suppression (NMS)](#51-non-maximum-suppression-nms)
  - [5.2. Kalman Filter](#52-kalman-filter)
  - [5.3. Hungarian Algorithm (Linear Assignment)](#53-hungarian-algorithm-linear-assignment)
  - [5.4. Perspective Transformation](#54-perspective-transformation-biến-đổi-phối-cảnh)
  - [5.5. Polynomial Curve Fitting](#55-polynomial-curve-fitting-khớp-đường-cong)
  - [5.6. Monocular Distance Estimation](#56-monocular-distance-estimation-ước-lượng-khoảng-cách)
  - [5.7. Point-in-Polygon Test](#57-point-in-polygon-test)
  - [5.8. IoU (Intersection over Union)](#58-iou-intersection-over-union)
- [6. Hệ thống ADAS](#6-hệ-thống-adas)
- [7. Pipeline xử lý ảnh](#7-pipeline-xử-lý-ảnh)
- [8. Design Patterns & Kiến trúc](#8-design-patterns--kiến-trúc)
- [9. Tổng hợp Dependencies](#9-tổng-hợp-dependencies)

---

## 1. Tổng quan tính năng

Dự án xây dựng hệ thống **ADAS (Advanced Driver Assistance System)** cho xe tự lái, bao gồm:

| Tính năng | Mô tả |
|-----------|--------|
| **Phát hiện vật thể (Object Detection)** | Nhận diện xe, người, xe máy, xe buýt, xe tải trên đường bằng YOLO |
| **Phát hiện làn đường (Lane Detection)** | Xác định vạch kẻ đường bằng Ultra-Fast Lane Detection |
| **Theo dõi vật thể (Object Tracking)** | Gán ID và theo dõi quỹ đạo vật thể qua các frame bằng ByteTrack |
| **Ước lượng khoảng cách (Distance Estimation)** | Tính khoảng cách từ camera đến vật thể bằng mô hình Pinhole Camera |
| **Cảnh báo va chạm (FCWS)** | Cảnh báo 3 mức: Normal / Prompt / Warning dựa trên khoảng cách |
| **Cảnh báo lệch làn (LDWS)** | Phát hiện xe lệch trái / phải / giữa so với làn đường |
| **Hỗ trợ giữ làn (LKAS)** | Phát hiện độ cong đường: Thẳng / Cua nhẹ / Cua gấp |
| **Bird's-eye View** | Chuyển đổi góc nhìn từ trước sang trên xuống (top-down) |
| **Hiển thị Dashboard** | Giao diện hiển thị thông tin ADAS, FPS, trạng thái cảnh báo |

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

## 2. Thư viện Python sử dụng

| Thư viện | File sử dụng | Mục đích |
|----------|-------------|----------|
| **OpenCV (`cv2`)** | Hầu hết mọi file | Đọc/ghi video, resize ảnh, chuyển đổi màu, vẽ bounding box, `warpPerspective`, `blobFromImage`, `pointPolygonTest`, `fillPoly`, `addWeighted` |
| **NumPy (`numpy`)** | Mọi file | Tính toán ma trận, `polyfit`, `linspace`, `linalg`, `einsum`, broadcasting |
| **ONNX Runtime (`onnxruntime`)** | `coreEngine.py` | Chạy inference model ONNX với CPU/CUDA providers |
| **TensorRT (`tensorrt`)** | `coreEngine.py` | Inference GPU hiệu suất cao — deserialize engine & thực thi |
| **PyCUDA (`pycuda`)** | `coreEngine.py`, `demo.py` | Quản lý CUDA context/stream, cấp phát bộ nhớ GPU, truyền dữ liệu host↔device |
| **SciPy (`scipy`)** | `kalman_filter.py`, `matching.py`, `ultrafastLaneDetector.py` | `scipy.linalg` (Cholesky decomposition), `scipy.spatial.distance.cdist`, `scipy.special.softmax` |
| **Numba (`numba`)** | `ObjectDetector/utils.py` | JIT compilation (`@jit(nopython=True)`) tăng tốc vòng lặp NMS/Soft-NMS |
| **LAP (`lap`)** | `matching.py` | Giải bài toán Linear Assignment — thuật toán Jonker-Volgenant (`lapjv`) |
| **ctypes** | `taskConditions.py` | Gọi Windows kernel32 API để hiển thị màu trong console |
| **logging** | `taskConditions.py`, `demo.py` | Hệ thống logging tùy chỉnh với class `Logger` |

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

## 3. Mô hình AI / Deep Learning

### 3.1. YOLO - Phát hiện vật thể

**YOLO (You Only Look Once)** là mô hình object detection real-time, nhận diện vật thể chỉ trong một lần forward pass qua mạng neural.

#### Các phiên bản được hỗ trợ

| Model | Enum | Định dạng Output |
|-------|------|-------------------|
| **YOLOv5** | `YOLOV5 = 0` | `(-1, obj_conf + 5[bbox, cls_conf])` |
| **YOLOv5-Lite** | `YOLOV5_LITE = 1` | Tương tự + anchor-based post-processing |
| **YOLOv6** | `YOLOV6 = 2` | Tương tự YOLOv5 |
| **YOLOv7** | `YOLOV7 = 3` | Tương tự YOLOv5 |
| **YOLOv8** | `YOLOV8 = 4` | `(obj_conf + 4[bbox], -1)` — transposed |
| **YOLOv9** | `YOLOV9 = 5` | Tương tự YOLOv8 |
| **YOLOv10** | `YOLOV10 = 6` | Tương tự YOLOv8 (anchor-free) |

> **Config hiện tại**: Sử dụng **YOLOv10m** qua TensorRT

#### Quy trình xử lý YOLO

```
Input Frame
    │
    ▼
┌──────────────────────┐
│  Letterbox Resize     │  ← Giữ tỷ lệ ảnh + padding (Scaler)
│  BGR → RGB            │
│  Normalize [0, 1]     │
│  HWC → NCHW           │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  TensorRT / ONNX      │  ← Forward pass qua model
│  Inference Engine      │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  Decode Predictions   │  ← Xử lý output khác nhau theo version
│  v5/6/7: anchor-based │
│  v8/9/10: anchor-free │ 
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  Soft-NMS             │  ← Loại bỏ bounding box trùng lặp
│  (3 mode: linear,     │     bằng score decay thay vì hard suppress
│   gaussian, greedy)    │
└──────────────────────┘
    │
    ▼
  Detected Objects (RectInfo: x, y, w, h, conf, label)
```

#### Đặc điểm kỹ thuật
- **YOLOv5-Lite** sử dụng **anchor grids** với strides `[8, 16, 32]` và 3 anchor scales
- **YOLOv8/9/10** output được **transposed** (`output.T`) trước khi xử lý — kiến trúc anchor-free
- Dùng **Soft-NMS** (không phải NMS thông thường) để giữ lại các detection gần nhau tốt hơn

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

### 3.2. Ultra-Fast Lane Detection - Phát hiện làn đường

**UFLD (Ultra-Fast Lane Detection)** là mô hình phát hiện làn đường cực nhanh, hoạt động dựa trên phương pháp **row/column anchor classification** thay vì segmentation truyền thống.

#### Các phiên bản được hỗ trợ

| Model | Enum | Dataset |
|-------|------|---------|
| **UFLD v1 TuSimple** | `UFLD_TUSIMPLE = 0` | TuSimple |
| **UFLD v1 CULane** | `UFLD_CULANE = 1` | CULane |
| **UFLD v2 TuSimple** | `UFLDV2_TUSIMPLE = 2` | TuSimple |
| **UFLD v2 CULane** | `UFLDV2_CULANE = 3` | CULane |
| **UFLD v2 CurveLanes** | `UFLDV2_CURVELANES = 4` | CurveLanes |

> **Config hiện tại**: **UFLDv2 CULane ResNet-18** qua TensorRT
>
> **Backbone**: ResNet-18 / ResNet-34

#### Post-processing UFLD v2

```
Model Output (4 heads)
    │
    ├── loc_row    ← Vị trí hàng (row anchors) cho làn trong
    ├── loc_col    ← Vị trí cột (col anchors) cho làn ngoài
    ├── exist_row  ← Xác suất tồn tại (row)
    └── exist_col  ← Xác suất tồn tại (col)
    │
    ▼
┌──────────────────────────┐
│  Row Anchors → Làn trong  │  ← left-ego (1), right-ego (2)
│  Col Anchors → Làn ngoài  │  ← left-side (0), right-side (3)
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Local Softmax Refinement │  ← Weighted average over local_width
│  → Sub-pixel vị trí làn   │     neighboring grid cells
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Existence Check          │  ← Row lanes: > num_cls/2 valid points
│                           │     Col lanes: > num_cls/4 valid points
└──────────────────────────┘
    │
    ▼
  4 Lane Points → Ego Lane Area Polygon
```

#### Xây dựng Lane Area
- Chỉ tạo area khi **cả 2 làn ego** (left-ego + right-ego) được phát hiện
- Area polygon = điểm left lane + `np.flipud(right lane)`
- Hỗ trợ polynomial smoothing tùy chọn qua `__adjust_lanes_points()`

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

### 3.3. ByteTrack - Theo dõi vật thể

**ByteTrack** là thuật toán Multi-Object Tracking (MOT), nổi bật với chiến lược **two-stage association** — tận dụng cả detection có score thấp.

#### Vòng đời Track (State Machine)

```
         activate()          mark_lost()         mark_removed()
  New ──────────────► Tracked ──────────────► Lost ──────────────► Removed
  (0)                  (1)                    (2)                    (3)
                        ▲                      │
                        │    re_activate()      │
                        └──────────────────────┘
```

#### Quy trình Update mỗi Frame

| Bước | Mô tả |
|------|--------|
| **Step 1** | Tách detections thành **high-score** (`> track_thresh`) và **low-score** (`0.1 – track_thresh`) |
| **Step 2** | **First association** — ghép tracked + lost tracks với high-score detections bằng IoU + score fusion |
| **Step 3** | **Second association** — ghép remaining tracked tracks với low-score detections |
| **Step 3b** | Xử lý **unconfirmed tracks** (chỉ xuất hiện 1 frame) với remaining high-score detections |
| **Step 4** | **Init new tracks** từ unmatched detections (score > `det_thresh`) |
| **Step 5** | **Age-out** — lost tracks vượt quá `max_time_lost` frames → Removed |

#### Đặc điểm kỹ thuật
- **Class ID voting**: Track lưu `class_id_history`, class cuối cùng = `max(count)`
- **Crop storage**: Lưu ảnh cắt vật thể cho potential Re-ID
- **Trajectory buffer**: `LimitedList(maxlen=30)` lưu 30 vị trí gần nhất
- **Duplicate removal**: IoU-based, giữ track sống lâu hơn
- **Shared Kalman Filter**: Class-level instance dùng chung cho `multi_predict()`

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

## 4. Inference Engine & Định dạng Model

### TensorRT Engine

| Thành phần | Chi tiết |
|-----------|----------|
| **Class** | `TensorRTBase` + `TensorRTEngine(EngineBase, TensorRTBase)` |
| **Mục đích** | Inference GPU hiệu suất cao trên NVIDIA GPU |
| **Quản lý bộ nhớ** | `cuda.pagelocked_empty`, `cuda.mem_alloc`, async transfers |
| **Thực thi** | `context.execute_async_v2()` — asynchoronous inference qua CUDA stream |
| **Định dạng** | File `.trt` — engine đã được tối ưu hóa cho GPU cụ thể |

### ONNX Runtime Engine

| Thành phần | Chi tiết |
|-----------|----------|
| **Class** | `OnnxEngine(EngineBase)` |
| **Mục đích** | Inference đa nền tảng (CPU & GPU) |
| **Providers** | `CUDAExecutionProvider` (GPU) hoặc `CPUExecutionProvider` |
| **Input types** | Tự động detect float16/float32 |
| **Định dạng** | File `.onnx` — chuẩn mở cho model AI |

### Chọn Engine tự động
```python
# Trong yoloDetector.py — tự động chọn engine dựa trên extension file
if model_path.endswith(".trt"):
    engine = TensorRTEngine(model_path)
elif model_path.endswith(".onnx"):
    engine = OnnxEngine(model_path)
```

### Các file model trong project

| File | Format | Vai trò |
|------|--------|---------|
| `culane_res18.trt` | TensorRT | Lane detection (production) |
| `culane_res18.onnx` | ONNX | Lane detection (cross-platform) |
| `culane_res18.pth` | PyTorch | Lane detection (source weights) |
| `yolov10m.trt` | TensorRT | Object detection (production) |
| `yolov10m.onnx` | ONNX | Object detection (cross-platform) |
| `yolov10m.pt` | PyTorch | Object detection (source weights) |
| `coco_label.txt` | Text | 80 class labels COCO dataset |

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

## 5. Thuật toán & Kỹ thuật xử lý ảnh

### 5.1. Non-Maximum Suppression (NMS)

**Mục đích**: Loại bỏ các bounding box trùng lặp, chỉ giữ detection tốt nhất.

#### Standard NMS (Greedy)
```
1. Sắp xếp boxes theo confidence giảm dần
2. Chọn box có conf cao nhất
3. Tính IoU với tất cả box còn lại
4. Loại bỏ box có IoU > threshold
5. Lặp lại cho đến hết
```
- Được tăng tốc bằng **Numba JIT** (`@jit(nopython=True)`)

#### Soft-NMS (Được sử dụng trong production)
Thay vì loại bỏ hoàn toàn, **giảm score** của box trùng lặp:

| Mode | Công thức | Đặc điểm |
|------|-----------|----------|
| **Linear** | $s_i = s_i \times (1 - IoU)$ nếu $IoU > \sigma$ | Giảm tuyến tính |
| **Gaussian** | $s_i = s_i \times e^{-IoU^2/\sigma}$ | Giảm mượt theo Gaussian |
| **Greedy** | $s_i = 0$ nếu $IoU > \sigma$ | NMS truyền thống |

> Project sử dụng **Soft-NMS** để giữ lại các detection gần nhau tốt hơn, đặc biệt trong cảnh giao thông đông đúc.

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

### 5.2. Kalman Filter

**Mục đích**: Dự đoán vị trí vật thể trong frame tiếp theo dựa trên mô hình chuyển động.

#### Không gian trạng thái 8 chiều

$$\mathbf{x} = [x, y, a, h, \dot{x}, \dot{y}, \dot{a}, \dot{h}]^T$$

| Biến | Ý nghĩa |
|------|---------|
| $x, y$ | Tâm bounding box |
| $a$ | Tỷ lệ khung hình (aspect ratio) |
| $h$ | Chiều cao bounding box |
| $\dot{x}, \dot{y}, \dot{a}, \dot{h}$ | Vận tốc tương ứng |

#### Các bước hoạt động

| Bước | Phương thức | Mô tả |
|------|------------|-------|
| **Khởi tạo** | `initiate(measurement)` | Tạo track mới từ detection đầu tiên |
| **Dự đoán** | `predict(mean, covariance)` | Ngoại suy trạng thái theo mô hình vận tốc không đổi |
| **Chiếu** | `project(mean, covariance)` | Chuyển từ không gian trạng thái → không gian quan sát |
| **Cập nhật** | `update(mean, covariance, measurement)` | Hiệu chỉnh bằng Cholesky decomposition cho Kalman gain |
| **Batch predict** | `multi_predict(mean, covariance)` | Dự đoán nhiều track cùng lúc (vectorized) |

#### Khoảng cách Mahalanobis

$$d^2 = (\mathbf{z} - \hat{\mathbf{z}})^T \mathbf{S}^{-1} (\mathbf{z} - \hat{\mathbf{z}})$$

- Được sử dụng để **gating** — loại bỏ cặp track-detection không hợp lý
- Ngưỡng chi-square ở mức 95% cho 1-9 bậc tự do

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

### 5.3. Hungarian Algorithm (Linear Assignment)

**Mục đích**: Ghép tối ưu detections với tracks đã có (bài toán assignment 1-1).

- Sử dụng thuật toán **Jonker-Volgenant** (`lap.lapjv`) — biến thể nhanh hơn Hungarian gốc
- Ma trận chi phí (cost matrix) = `1 - IoU`
- **Score fusion**: Kết hợp IoU similarity với detection confidence:

$$cost = 1 - (IoU_{sim} \times det_{scores})$$

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

### 5.4. Perspective Transformation (Biến đổi phối cảnh)

**Mục đích**: Chuyển đổi góc nhìn phía trước (frontal view) sang góc nhìn từ trên xuống (bird's-eye view).

```
Frontal View                    Bird's-Eye View
┌──────────────────┐           ┌──────────────────┐
│    ╱        ╲    │           │  │            │  │
│   ╱          ╲   │    →→→    │  │            │  │
│  ╱    Road    ╲  │  warp    │  │    Road    │  │
│ ╱              ╲ │           │  │            │  │
│╱                ╲│           │  │            │  │
└──────────────────┘           └──────────────────┘
```

- Sử dụng `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()`
- Batch point transformation: `np.einsum('kl, ...l->...k', M, points)`
- **Adaptive source region**: Tự động cập nhật vùng nguồn dựa trên làn đường phát hiện được (Top/Bottom/Default modes)

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

### 5.5. Polynomial Curve Fitting (Khớp đường cong)

**Mục đích**: Fit đường cong toán học cho làn đường, tính bán kính cong và offset xe.

#### Polynomial bậc 2

$$x = Ay^2 + By + C$$

- Sử dụng `np.polyfit(y, x, 2)` trên bird-view lane points

#### Bán kính cong (Radius of Curvature)

$$R = \frac{(1 + (2Ay_{eval} \cdot y_m + B)^2)^{3/2}}{|2A|}$$

#### Chuyển đổi pixel → mét

| Thông số | Giá trị | Ý nghĩa |
|----------|---------|---------|
| `ym_per_pix` | 30/720 | Mét/pixel theo trục Y |
| `xm_per_pix` | 3.7/700 | Mét/pixel theo trục X |

#### Tính offset xe
- Offset = khoảng cách từ trung tâm ảnh đến trung điểm 2 làn ego
- Dương = lệch phải, Âm = lệch trái

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

### 5.6. Monocular Distance Estimation (Ước lượng khoảng cách)

**Mục đích**: Tính khoảng cách từ camera đến vật thể bằng 1 camera duy nhất.

#### Mô hình Pinhole Camera

$$d = \frac{H_{real} \times f}{H_{pixel}}$$

| Thông số | Giá trị |
|----------|---------|
| $f$ (focal length) | 100 pixels |
| Chuyển đổi | feet → meters: `d = d/12 × 0.3048` |

#### Kích thước tham chiếu vật thể

| Vật thể | Chiều cao (cm) | Chiều rộng (cm) |
|---------|----------------|------------------|
| Người (person) | 160 | 50 |
| Xe đạp (bicycle) | 98 | 65 |
| Xe máy (motorbike) | 100 | 100 |
| Ô tô (car) | 150 | 180 |
| Xe buýt (bus) | 319 | 250 |
| Xe tải (truck) | 346 | 250 |

#### Quy trình phát hiện va chạm
1. `updateDistance()` — Tính khoảng cách cho mỗi vật thể dựa trên chiều cao pixel
2. `calcCollisionPoint()` — Sắp xếp theo khoảng cách (gần nhất trước), kiểm tra bottom-center có nằm trong ego-lane polygon không

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

### 5.7. Point-in-Polygon Test

**Mục đích**: Kiểm tra vật thể có nằm trong vùng làn đường ego hay không (đánh giá nguy cơ va chạm).

- **Ray-casting algorithm** (thuật toán tia): Đếm số lần tia từ điểm cắt qua cạnh polygon
- Kết hợp `cv2.pointPolygonTest()` của OpenCV

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

### 5.8. IoU (Intersection over Union)

**Mục đích**: Đo mức độ trùng lặp giữa 2 bounding box.

$$IoU = \frac{Area_{intersection}}{Area_{union}}$$

- Triển khai vectorized bằng NumPy cho hiệu suất cao
- Dùng làm cost matrix trong tracker: `cost = 1 - IoU`
- Dùng trong NMS để xác định box nào trùng lặp

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

## 6. Hệ thống ADAS

### FCWS — Forward Collision Warning System (Cảnh báo va chạm phía trước)

| Trạng thái | Điều kiện | Mô tả |
|-----------|-----------|--------|
| **WARNING** | Khoảng cách ≤ 1.5m | Nguy hiểm — cảnh báo khẩn cấp |
| **PROMPT** | Khoảng cách ≤ 3.0m | Chú ý — vật thể đang tiến gần |
| **NORMAL** | Khoảng cách > 3.0m | An toàn |

- Sử dụng **median** của 3 lần đo gần nhất để lọc nhiễu

### LDWS — Lane Departure Warning System (Cảnh báo lệch làn)

| Trạng thái | Mô tả |
|-----------|--------|
| **LEFT** | Xe lệch sang trái |
| **RIGHT** | Xe lệch sang phải |
| **CENTER** | Xe ở giữa làn |

- Sử dụng **median** của 5 lần đo, ngưỡng `offset_thres = 0.9m`
- Cross-reference với curvature để tránh false positives khi xe đang cua

### LKAS — Lane Keeping Assist System (Hỗ trợ giữ làn)

| Trạng thái | Mô tả |
|-----------|--------|
| **STRAIGHT** | Đường thẳng |
| **EASY_LEFT** | Cua nhẹ trái |
| **HARD_LEFT** | Cua gấp trái |
| **EASY_RIGHT** | Cua nhẹ phải |
| **HARD_RIGHT** | Cua gấp phải |

- Sử dụng **median** của 10 lần đo, ngưỡng `curvae_thres = 500`
- Bao gồm **adaptive bird-view calibration** — tự động hiệu chỉnh khi phát hiện dao động

### Mức ưu tiên
```
FCWS (va chạm) > LDWS (lệch làn) > LKAS (giữ làn)
```

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

## 7. Pipeline xử lý ảnh

### Tiền xử lý Object Detection

```
Input Frame
    │
    ├── 1. Letterbox resize (giữ tỷ lệ + padding)  ← class Scaler
    ├── 2. BGR → RGB                                 ← cv2.dnn.blobFromImage
    ├── 3. Normalize [0, 1] (÷255)
    └── 4. HWC → NCHW format
```

### Tiền xử lý Lane Detection

```
Input Frame
    │
    ├── 1. BGR → RGB
    ├── 2. Crop bottom portion (crop_ratio)
    ├── 3. Resize to model input size
    ├── 4. ImageNet normalization:
    │      mean = [0.485, 0.456, 0.406]
    │      std  = [0.229, 0.224, 0.225]
    └── 5. HWC → NCHW format
```

### Visualization & Rendering

| Kỹ thuật | Mô tả |
|----------|--------|
| **Alpha-blend overlay** | `cv2.addWeighted` cho lane area overlay mờ |
| **Corner-style bbox** | Vẽ góc thay vì hình chữ nhật đầy đủ (UI hiện đại) |
| **Shadow text/arrows** | `putText_shadow()`, `arrowedLine_shadow()` để tăng khả năng đọc |
| **Trajectory dots** | Vẽ quỹ đạo vật thể với thickness scaling theo vị trí |
| **Bird-view panel** | Hiển thị góc nhìn top-down bên cạnh video |
| **FPS counter** | Rolling 30-frame window calculation |

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

## 8. Design Patterns & Kiến trúc

| Pattern | Implementation | Mô tả |
|---------|---------------|--------|
| **Abstract Base Class (ABC)** | `EngineBase`, `ObjectDetectBase`, `LaneDetectBase`, `ObjectTrackBase` | Định nghĩa interface bắt buộc cho mọi engine/detector/tracker |
| **Multiple Inheritance** | `TensorRTEngine(EngineBase, TensorRTBase)`, `YoloDetector(ObjectDetectBase, YoloLiteParameters)` | Mixin pattern kết hợp interface + implementation |
| **Strategy Pattern** | `OnnxEngine` vs `TensorRTEngine` | Runtime chọn engine dựa trên extension file (`.onnx`/`.trt`) |
| **Factory / Config** | `_defaults` dict + `set_defaults()` | Classmethod injection cấu hình cho mọi detector |
| **Template Method** | `ObjectDetectBase`, `LaneDetectBase` | Base class định nghĩa skeleton, subclass implement logic cụ thể |
| **Dataclass** | `RectInfo`, `LaneInfo`, `Scaler` | Container dữ liệu có cấu trúc với validation |
| **Enum** | `ObjectModelType`, `LaneModelType`, `CollisionType`, `OffsetType`, `CurvatureType`, `TrackState` | Type-safe enums cho state/config |
| **Singleton-like** | `STrack.shared_kalman` | Class-level KalmanFilter dùng chung cho mọi track |
| **Observer / State** | `TaskConditions` | Aggregation thống kê rolling, quản lý chuyển trạng thái ADAS |
| **Property Decorators** | `EngineBase`, `LaneInfo` | Encapsulation với getter/setter validation |

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

## 9. Tổng hợp Dependencies

### Core (Bắt buộc)
```
opencv-python          # Xử lý ảnh & visualization
numpy                  # Tính toán số học
onnxruntime            # ONNX model inference
tensorrt (8.6.0)       # GPU-accelerated inference (NVIDIA)
pycuda                 # Quản lý bộ nhớ CUDA
scipy                  # Linear algebra & distance metrics
numba                  # JIT compilation cho NMS
lap                    # Linear assignment solver
```

### Standard Library (Có sẵn trong Python)
```
ctypes                 # Windows API calls
logging                # Logging framework
dataclasses            # Data containers
typing                 # Type annotations
abc                    # Abstract base classes
enum                   # Enumerations
collections            # OrderedDict, deque
math                   # Hàm toán học
time                   # Đo thời gian, FPS
```

### Cài đặt
```bash
pip install -r requirements.txt
```

> **Lưu ý**: TensorRT yêu cầu **NVIDIA GPU** và phải cài riêng:
> ```bash
> pip install tensorrt-8.6.0-cp39-none-win_amd64.whl
> ```

<p align="right">(<a href="#tech-top">back to top</a>)</p>

---

<div align="center">
  <i>Tài liệu được tạo dựa trên phân tích mã nguồn thực tế của project.</i>
</div>
