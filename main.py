import cv2
import depthai as dai
import numpy as np

# Define class labels for YOLOv6
class_labels = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


def calculate_depth(disparity_frame, bbox, baseline, focal_length):
    x1, y1, x2, y2 = bbox
    disparity_roi = disparity_frame[y1:y2, x1:x2]

    if disparity_roi.size == 0:
        return None

    valid_disparities = disparity_roi[disparity_roi > 0]
    if valid_disparities.size == 0:
        return None

    disparity_median = np.median(valid_disparities)

    if disparity_median > 0:
        depth = ((baseline * focal_length) / disparity_median) / 10  # Convertirea din mm in cm
        return depth

    return None


def enableIRLaser(device):
    device.setIrLaserDotProjectorBrightness(588)
    device.setIrFloodLightBrightness(200)


# Create the pipeline
pipeline = dai.Pipeline()

# Define nodes
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
colorCam = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)

xout_Left = pipeline.create(dai.node.XLinkOut)
xout_Right = pipeline.create(dai.node.XLinkOut)
xout_Rgb = pipeline.create(dai.node.XLinkOut)
xout_Depth = pipeline.create(dai.node.XLinkOut)
xout_NN = pipeline.create(dai.node.XLinkOut)

xout_Left.setStreamName("left")
xout_Right.setStreamName("right")
xout_Rgb.setStreamName("rgb")
xout_Depth.setStreamName("depth")
xout_NN.setStreamName("detections")

# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setPreviewSize(640, 640)
colorCam.setInterleaved(False)

# Stereo depth configuration
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setRectification(True)

# YOLOv6 network configuration
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(len(class_labels))
detectionNetwork.setCoordinateSize(4)
anchors = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks({"side52": [0, 1, 2], "side26": [3, 4, 5]})
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath("yolov6n_coco_640x640.blob")
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)
detectionNetwork.input.setQueueSize(4)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.disparity.link(xout_Depth.input)
monoLeft.out.link(xout_Left.input)
monoRight.out.link(xout_Right.input)
colorCam.preview.link(detectionNetwork.input)
colorCam.preview.link(xout_Rgb.input)
detectionNetwork.out.link(xout_NN.input)

# Baseline and focal length values
baseline = 75  # in mm, this is a placeholder, replace with actual value
focal_length = 278.68552  # in pixels, this is a placeholder, replace with actual value

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    enableIRLaser(device)

    # Output queues
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    while True:
        inLeft = qLeft.get()
        inRight = qRight.get()
        inRgb = qRgb.get()
        inDepth = qDepth.get()
        inDet = qDet.get()

        # Get BGR frame from NV12 encoded RGB frame
        frameLeft = inLeft.getCvFrame()
        frameRight = inRight.getCvFrame()
        frameRgb = inRgb.getCvFrame()
        frameDepth = inDepth.getFrame()

        # Convert grayscale images to BGR
        frameLeft = cv2.cvtColor(frameLeft, cv2.COLOR_GRAY2BGR)
        frameRight = cv2.cvtColor(frameRight, cv2.COLOR_GRAY2BGR)

        # Colormap for disparity
        frameDepthNorm = cv2.normalize(frameDepth, None, 0, 210, cv2.NORM_MINMAX, cv2.CV_8UC1)
        frameDepthColor = cv2.applyColorMap(frameDepthNorm, cv2.COLORMAP_JET)
        frameDepthColorResized = cv2.resize(frameDepthColor, (600, 400))


        detections = inDet.detections

        for detection in detections:
            bbox = [int(detection.xmin * frameRgb.shape[1]), int(detection.ymin * frameRgb.shape[0]),
                    int(detection.xmax * frameRgb.shape[1]), int(detection.ymax * frameRgb.shape[0])]
            cv2.rectangle(frameRgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            label = f"{class_labels[detection.label]}: {detection.confidence * 100:.2f}%"
            cv2.putText(frameRgb, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)

            # Calcularea adancimii
            depth = calculate_depth(frameDepth, bbox, baseline, focal_length)
            if depth is not None:
                depth_label = f"Depth: {depth:.2f} cm"
                cv2.putText(frameRgb, depth_label, (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                print(f"Detected {class_labels[detection.label]} at distance: {depth:.2f} cm")

        # Afisarea ferestrelor de preview
        combinedStereo = np.hstack((frameLeft, frameRight))
        cv2.imshow("Stereo Camera Streams", combinedStereo)
        cv2.imshow("RGB Camera Stream with YOLOv6", frameRgb)
        cv2.imshow("Disparity Map", frameDepthColorResized)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()