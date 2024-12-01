import time
import cv2
import numpy as np
import onnxruntime as ort
from .utils import sigmoid, crop_mask

class YOLO_model:
    """
    YOLO model class for book detection
    """
    def __init__(self, model_path):
        self.init_model(model_path)
        self.confidence_threshold = 0.3
        self.iou_threshold = 0.5


    def init_model(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output0_name = self.ort_session.get_outputs()[0].name
        self.output1_name = self.ort_session.get_outputs()[1].name
    
    def check(self):
        """
        Checks if the model is loaded correctly
        """
        return self.ort_session is not None
        
    def preprocess(self, image):
        self.image_height, self.image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize the image to the model input size
        img = cv2.resize(image, (640, 640))
        # Normalize the image
        img = img.astype(np.float32) / 255.0
        # Change to CHW format
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        input_tensor = np.expand_dims(img, axis=0)
        return input_tensor

    def inference(self, input_tensor, verbose=False):
        start = time.perf_counter()
        outputs = self.ort_session.run(None, {self.input_name: input_tensor})
        if verbose:
            print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs
    
    def post_process(self, output0, output1):
        # Remove the batch axis, and transpose the output
        predictions = np.squeeze(output0).T
        prototypes = np.squeeze(output1)
        num_classes = predictions.shape[1] - 4 - prototypes.shape[0]
        assert num_classes == 1, "Only one class (book) is supported"
        # Extract the class predictions
        class_predictions = np.squeeze(predictions[:, 4:4+num_classes])
        # Extract the bounding box predictions
        bounding_boxes = predictions[:, :4]
        # Extract the prototype coefficients
        prototype_coefficients = predictions[:, 4+num_classes:]
        # Apply NMS to the bounding boxes
        indices = cv2.dnn.NMSBoxes(bounding_boxes, class_predictions, self.confidence_threshold, self.iou_threshold)

        # Extract the detections
        detections = []
        masks_in = []
        X_factor = self.image_width / 640
        Y_factor = self.image_height / 640
        for i in indices.flatten():
            box = bounding_boxes[i]
            score = class_predictions[i]
            class_id = 0
            cx, cy, w, h = box
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            x1 = int(x1 * X_factor)
            y1 = int(y1 * Y_factor)
            x2 = int(x2 * X_factor)
            y2 = int(y2 * Y_factor)
            detections.append([x1, y1, x2, y2, score, class_id])
            masks_in.append(prototype_coefficients[i])
        masks = self.process_mask_upsample(prototypes, masks_in, [detection[:4] for detection in detections])
        return detections, masks

    def process_mask(self, protos, masks_in, bboxes):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of lower
        quality but is faster.

        Args:
        protos (np.ndarray): [mask_dim, mask_h, mask_w]
        masks_in (np.ndarray): [n, mask_dim], n is number of masks after nms
        bboxes (np.ndarray): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)
        """
        masks = sigmoid(np.dot(masks_in, protos.reshape(protos.shape[0], -1)))
        masks = masks.reshape(-1, *protos.shape[1:])
        masks = crop_mask(masks, bboxes)
        return masks > 0.5
        
    def process_mask_upsample(self, protos, masks_in, bboxes):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
        quality but is slower.

        Args:
        protos (np.ndarray): [mask_dim, mask_h, mask_w]
        masks_in (np.ndarray): [n, mask_dim], n is number of masks after nms
        bboxes (np.ndarray): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)

        Returns:
        (np.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape  # CHW
        masks = sigmoid(np.dot(masks_in, protos.reshape(c, -1))).reshape(-1, mh, mw)
        masks = cv2.resize(masks.transpose(1, 2, 0), (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        masks = crop_mask(masks, bboxes)  # CHW
        return masks > 0.5
        
    def draw_bboxes(self, image, detections):
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
        image (np.ndarray): The input image.
        detections (list): The list of detections.
        """
        for detection in detections:
            x1, y1, x2, y2, score, class_id = detection
            # Cast all floats to int
            if class_id == 0:
                color = [0, 255, 0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"book: {score:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, font, 0.5, 1)
                label_x = x1
                label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
                cv2.rectangle(image, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
                cv2.putText(image, label, (label_x, label_y), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        return image
    
    def draw_masks(self, image, masks):
        """
        Draw masks on the input image.

        Args:
        image (np.ndarray): The input image.
        masks (list): The list of masks.
        """
        for mask in masks:
            image[mask] = [127, 0, 127]
        return image
    
    def detect_books(self, image, verbose=False):
        """
        Detect books in an image.

        Args:
        image (np.ndarray): The input image.
        verbose (bool): Whether to print additional information.
        """
        input_tensor = self.preprocess(image)
        output0, output1 = self.inference(input_tensor, verbose)
        detections, masks = self.post_process(output0, output1)
        return detections, masks