<a name="readme-top"></a>

<h1 align="center">LANE DETECTION APPLICATION IN SELF-DRIVING CAR</h1>

<div align="center">
  <img src="image_resources/intro.gif" alt="Result Example" width="800px">
</div>

**Lane detection** is a computer vision task that involves identifying the boundaries of driving lanes in video or image scenes. The goal is to accurately locate and track the lane markings in real-time, even in challenging conditions such as low light, glare, or complex road layouts.

This repository develops a lane detection and tracking solution for self-driving cars. It is a crucial component of an autonomous driving system, ensuring safety and efficiency for the driver and passengers on the road.

# Project Structure

# Pipline
The implementation problem consists of 3 parts:
1. Traditional image processing method
2. Deep Learning Method
------- 
## Traditional image processing Method

> **PIPLINE**

![pipline](image_resources/pipline_1.png)

**1. Compute the camera calibration matrix and distortion coefficients.**

> This step involves calibrating the camera used for capturing images or videos. By taking multiple calibration images of a chessboard pattern from different angles, the camera calibration matrix and distortion coefficients can be computed. These parameters are essential for correcting the distortion in the subsequent image processing steps.
> 
**2. Apply a distortion correction to raw images.**

> Using the camera calibration matrix and distortion coefficients obtained in the previous step, the raw images captured by the camera can be undistorted. This correction ensures that straight lines appear straight in the image, reducing any distortion caused by the camera lens.

**3. Apply a perspective transform to generate a “bird’s-eye view” of the image.**

> Perspective transformation is applied to convert the undistorted image from a regular view to a top-down view, resembling a bird's-eye view. This transformation helps to focus on the lane region of interest and simplifies the subsequent lane detection process.

**4. Use color transforms (HSL) to create a thresholded binary image.**

> Different color spaces, gradient-based edge detection, and other image processing techniques are used to create a binary image where the lane markings are highlighted. Thresholding techniques are applied to identify the pixels that likely belong to the lanes based on their color or gradient properties.

**5. Detect lane pixels and fit to find the lane boundary.**

> Lane pixels are detected by applying a pixel-wise search or sliding window search on the thresholded binary image. The detected pixels are then used to fit a mathematical model (e.g., polynomial) that represents the lane boundary. This step determines the shape and position of the detected lanes.

**6. Determine the curvature of the lane and vehicle position with respect to center.**

> Using the fitted lane boundary, the curvature of the lane can be calculated. This provides valuable information about the curvature radius, which is essential for controlling the vehicle during self-driving. Additionally, the vehicle's position with respect to the lane center can be determined, allowing for appropriate adjustments in driving behavior.

**7. Warp the detected lane boundaries back onto the original image and display information for ADAS System. Information displayed includes:**
* **LKAS: Lane Keeping Assist System with Vietnamese traffic signs**
* **LDWS: Lane Departure Warning System ((Not aiming to develop in this method but develop in Deep Learning method)**
* **Vehicle position from center**
* **And finally Frame ID and FPS are displayed in the upper right corner for testing**


> The detected lane boundaries, along with the calculated curvature and vehicle position, are mapped back onto the original undistorted image. This step visualizes the detected lanes and provides important information for an Advanced Driver Assistance System (ADAS). The displayed information may include LKAS (Lane Keeping Assist System) with Vietnamese traffic signs, LDWS (Lane Departure Warning System), the vehicle's position from the center, and other relevant details. Additionally, for testing purposes, the Frame ID and Frames Per Second (FPS) can be displayed in the upper right corner of the image.

## Deep Learning Method
1. **Lane Detector**: Ultra Fast Lane Detection ([V1](https://github.com/cfzd/Ultra-Fast-Lane-Detection) & [V2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)) on backbone ResNet (18 & 34)
2. **Vehicle Detector**: [YOLOv8 (v8m & v8l)](https://github.com/ultralytics/ultralytics) [ONNX](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection) 
3. **Requirements**
- NIVIDA GPU for TenssoRT
- Others dependencies: [requirements.txt](requirements.txt)

 * ***Video inference*** :

   * Setting Config :
     > Note : can support onnx/tensorRT format model. But it needs to match the same model type.

    ```python
    lane_config = {
     "model_path": "./TrafficLaneDetector/models/culane_res18.trt",
     "model_type" : LaneModelType.UFLDV2_CULANE
    }

    object_config = {
     "model_path": './ObjectDetector/models/yolov8l-coco.trt',
     "model_type" : ObjectModelType.YOLOV8,
     "classes_path" : './ObjectDetector/models/coco_label.txt',
     "box_score" : 0.4,
     "box_nms_iou" : 0.45
    }
   ```
   | Target          | Model Type                       |  Describe                                         | 
   | :-------------: |:-------------------------------- | :------------------------------------------------ | 
   | Lanes           | `LaneModelType.UFLD_TUSIMPLE`    | Support Tusimple data with ResNet18 backbone.     | 
   | Lanes           | `LaneModelType.UFLD_CULANE`      | Support CULane data with ResNet18 backbone.       | 
   | Lanes           | `LaneModelType.UFLDV2_TUSIMPLE`  | Support Tusimple data with ResNet18/34 backbone.  |
   | Lanes           | `LaneModelType.UFLDV2_CULANE`    | Support CULane data with ResNet18/34 backbone.    | 
   | Object          | `ObjectModelType.YOLOV5`         | Support yolov5n/s/m/l/x model.                    | 
   | Object          | `ObjectModelType.YOLOV5_LITE`    | Support yolov5lite-e/s/c/g model.                 | 
   | Object          | `ObjectModelType.YOLOV8`         | Support yolov8n/s/m/l/x model.                    | 
   
   * Run :
   
    ```
    python demo.py
    ```
 4. **Results**
 - Deep Learning 

![DemoDL](image_resources/DL_demo1.gif)
![DemoDL_1](image_resources/DL_demo2.gif)
![DemoDL_2](image_resources/DemoDL2.gif)

> **The best type of model for the problem ULFD2_CULANE_RES18/34 and YOLO8l !**

## Usage
Note: python 3.8.20
CUDA Version: 11.7
cuDNN Version: 8500
TensorRT Version: 8.5.3.1
1. Git clone this repository:
```
git clone 

```

2. Install Pytorch.
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 --use

```
3. Install TensorRT (từ file .whl download from internet)
```
pip install tensorrt-8.5.3.1-cp38-none-win_amd64.whl

```

4. Install the required libraries with the following command:
```
pip install -r requirements.txt
```

1. Running program
 ```
 python demo.py
 ```



