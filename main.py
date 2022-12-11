import argparse
import cv2
import numpy as np
import onnxruntime as ort


class DAMO_YOLO():
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5):
        self.classes = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines()))
        self.num_class = len(self.classes)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, so)
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), 0, 0.7, (0, 255, 0), thickness=2)
        return frame

    def detect(self, frame):
        temp_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded_image = np.ones((self.input_height, self.input_width, 3), dtype=np.uint8)
        ratio = min(self.input_height / temp_image.shape[0], self.input_width / temp_image.shape[1])
        neww, newh = int(temp_image.shape[1] * ratio), int(temp_image.shape[0] * ratio)
        temp_image = cv2.resize(temp_image, (neww, newh), interpolation=cv2.INTER_LINEAR)
        padded_image[:newh, :neww, :] = temp_image

        padded_image = padded_image.transpose(2, 0, 1)
        padded_image = np.expand_dims(padded_image, axis=0).astype(np.float32)

        # Inference
        results = self.session.run(None, {self.input_name: padded_image})

        scores, bboxes = results[0].squeeze(axis=0), results[1].squeeze(axis=0)
        bboxes /= ratio

        boxes, confidences, classIds = [], [], []
        for i in range(bboxes.shape[0]):
            score = np.max(scores[i, :])
            if score < self.confThreshold:
                continue

            class_id = np.argmax(scores[i, :])
            x, y, xmax, ymax = bboxes[i, :].astype(np.int32)
            width, height = xmax - x, ymax - y

            boxes.append([x, y, width, height])
            classIds.append(class_id)
            confidences.append(score)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, default='weights/damoyolo_tinynasL20_T_192x320.onnx',
                        choices=["weights/damoyolo_tinynasL20_T_192x320.onnx",
                                 "weights/damoyolo_tinynasL20_T_256x320.onnx",
                                 "weights/damoyolo_tinynasL20_T_256x416.onnx",
                                 "weights/damoyolo_tinynasL20_T_288x480.onnx",
                                 "weights/damoyolo_tinynasL20_T_384x640.onnx",
                                 "weights/damoyolo_tinynasL20_T_480x640.onnx",
                                 "weights/damoyolo_tinynasL20_T_480x800.onnx",
                                 "weights/damoyolo_tinynasL20_T_640x640.onnx",
                                 "weights/damoyolo_tinynasL20_T_736x1280.onnx",
                                 "weights/damoyolo_tinynasL25_S_192x320.onnx",
                                 "weights/damoyolo_tinynasL25_S_256x320.onnx",
                                 "weights/damoyolo_tinynasL25_S_256x416.onnx",
                                 "weights/damoyolo_tinynasL25_S_288x480.onnx",
                                 "weights/damoyolo_tinynasL25_S_384x640.onnx",
                                 "weights/damoyolo_tinynasL25_S_480x640.onnx",
                                 "weights/damoyolo_tinynasL25_S_480x800.onnx",
                                 "weights/damoyolo_tinynasL25_S_640x640.onnx",
                                 "weights/damoyolo_tinynasL25_S_736x1280.onnx",
                                 "weights/damoyolo_tinynasL35_M_192x320.onnx",
                                 "weights/damoyolo_tinynasL35_M_256x320.onnx",
                                 "weights/damoyolo_tinynasL35_M_256x416.onnx",
                                 "weights/damoyolo_tinynasL35_M_288x480.onnx",
                                 "weights/damoyolo_tinynasL35_M_384x640.onnx",
                                 "weights/damoyolo_tinynasL35_M_480x640.onnx",
                                 "weights/damoyolo_tinynasL35_M_480x800.onnx",
                                 "weights/damoyolo_tinynasL35_M_640x640.onnx",
                                 "weights/damoyolo_tinynasL35_M_736x1280.onnx"], help="model path")
    parser.add_argument("--imgpath", type=str, default='images/dog.jpg', help="image path")
    parser.add_argument("--confThreshold", default=0.5, type=float, help='class confidence')
    parser.add_argument("--nmsThreshold", default=0.85, type=float, help='iou thresh')
    args = parser.parse_args()

    net = DAMO_YOLO(args.modelpath, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)
    srcimg = cv2.imread(args.imgpath)
    srcimg = net.detect(srcimg)

    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
