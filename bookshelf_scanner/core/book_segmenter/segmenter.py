import cv2
import json
import numpy       as np
import onnxruntime as ort

from bookshelf_scanner import ModuleLogger, Utils
from dataclasses       import dataclass
from pathlib           import Path
from typing            import Any

logger = ModuleLogger('segmenter')()

# -------------------- Utility Functions --------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid function element-wise to the input array.
    """
    return 1 / (1 + np.exp(-x))

def crop_mask(masks: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """
    Crops masks using the provided bounding boxes.

    Args:
        masks  : Boolean or binary masks of shape [n, h, w]
        bboxes : Bounding boxes of shape [n, 4]

    Returns:
        np.ndarray: Cropped masks.
    """
    n, h, w = masks.shape
    cropped_masks = np.zeros_like(masks)
    for i in range(n):
        x1, y1, x2, y2 = bboxes[i]
        cropped_masks[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
    return cropped_masks

# -------------------- Data Classes --------------------

@dataclass
class BookSegmentResult:
    """
    Holds information about a single detected book segment.
    """
    file_name   : str  | None
    confidence  : float
    bbox        : list[int]
    image_array : np.ndarray | None

# -------------------- YOLOModel Class --------------------

class YOLOModel:
    """
    YOLO model class for detecting books in an image.
    """

    def __init__(
        self,
        model_path           : Path,
        confidence_threshold : float = 0.3,
        iou_threshold        : float = 0.5
    ):
        """
        Initializes the YOLO model.

        Args:
            model_path           : Path to the ONNX model file
            confidence_threshold : Confidence threshold for detections
            iou_threshold        : IOU threshold for NMS
        """
        self.model_path           = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold        = iou_threshold
        self.ort_session          = None
        self.input_name           = None
        self.output0_name         = None
        self.output1_name         = None

        self.init_model(model_path = self.model_path)

    def init_model(
        self,
        model_path : Path
    ):
        """
        Initializes the ONNX runtime session for the YOLO model.

        Args:
            model_path : Path to the ONNX model file
        """
        self.ort_session  = ort.InferenceSession(str(model_path))
        self.input_name   = self.ort_session.get_inputs()[0].name
        self.output0_name = self.ort_session.get_outputs()[0].name
        self.output1_name = self.ort_session.get_outputs()[1].name
        logger.debug(f"YOLO model loaded from: {model_path}")

    def check(self) -> bool:
        """
        Checks if the model is loaded correctly.

        Returns:
            bool: True if the model is loaded, False otherwise.
        """
        return self.ort_session is not None

    def preprocess(
        self,
        image : np.ndarray
    ) -> np.ndarray:
        """
        Preprocesses the input image for inference.

        Args:
            image : Input BGR image

        Returns:
            np.ndarray: Preprocessed image as input tensor
        """
        self.image_height, self.image_width, _ = image.shape
        image_rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img          = cv2.resize(image_rgb, (640, 640))
        img          = img.astype(np.float32) / 255.0
        img          = np.transpose(img, (2, 0, 1))
        input_tensor = np.expand_dims(img, axis=0)
        return input_tensor

    def inference(
        self,
        input_tensor : np.ndarray,
        verbose      : bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs inference on the preprocessed input tensor.

        Args:
            input_tensor : Preprocessed input tensor
            verbose      : Print inference time if True

        Returns:
            tuple: (output0, output1) from the model
        """
        start   = cv2.getTickCount()
        outputs = self.ort_session.run(None, {self.input_name: input_tensor})
        if verbose:
            time_ms = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
            logger.debug(f"Inference time: {time_ms:.2f} ms")

        return outputs[0], outputs[1]

    def post_process(
        self,
        output0 : np.ndarray,
        output1 : np.ndarray
    ) -> tuple[list[list[float]], np.ndarray]:
        """
        Post-processes the model outputs to get bounding boxes and masks.

        Args:
            output0 : Bounding boxes and scores
            output1 : Prototype masks

        Returns:
            tuple: (detections, masks)
        """
        predictions      = np.squeeze(output0).T
        prototypes       = np.squeeze(output1)
        num_classes      = predictions.shape[1] - 4 - prototypes.shape[0]

        if num_classes != 1:
            logger.error("The model is expected to have only one class (book).")
            return [], np.array([])

        class_predictions       = np.squeeze(predictions[:, 4:4 + num_classes])
        bounding_boxes          = predictions[:, :4]
        prototype_coefficients  = predictions[:, 4 + num_classes:]

        indices = cv2.dnn.NMSBoxes(
            bboxes         = bounding_boxes.tolist(),
            scores         = class_predictions.tolist(),
            score_threshold= self.confidence_threshold,
            nms_threshold  = self.iou_threshold
        )

        detections = []
        masks_in   = []
        X_factor   = self.image_width / 640
        Y_factor   = self.image_height / 640

        for i in indices.flatten():
            box   = bounding_boxes[i]
            score = class_predictions[i]
            cx, cy, w, h = box
            x1 = int((cx - w / 2) * X_factor)
            y1 = int((cy - h / 2) * Y_factor)
            x2 = int((cx + w / 2) * X_factor)
            y2 = int((cy + h / 2) * Y_factor)

            detections.append([x1, y1, x2, y2, float(score), 0])
            masks_in.append(prototype_coefficients[i])

        masks = self.process_mask_upsample(
            protos   = prototypes,
            masks_in = masks_in,
            bboxes   = [det[:4] for det in detections]
        )

        return detections, masks

    def process_mask_upsample(
        self,
        protos   : np.ndarray,
        masks_in : list[np.ndarray],
        bboxes   : list[list[int]]
    ) -> np.ndarray:
        """
        Applies the predicted mask prototypes to the detected bounding boxes for high-quality masks.

        Args:
            protos   : Prototype masks [mask_dim, mask_h, mask_w]
            masks_in : Prototype coefficients for each detection
            bboxes   : List of bounding boxes [x1, y1, x2, y2]

        Returns:
            np.ndarray: Boolean mask array [n, h, w]
        """
        c, mh, mw = protos.shape
        masks = sigmoid(np.dot(masks_in, protos.reshape(c, -1))).reshape(-1, mh, mw)
        masks = cv2.resize(
            masks.transpose(1, 2, 0),
            (self.image_width, self.image_height),
            interpolation = cv2.INTER_LINEAR
        ).transpose(2, 0, 1)

        masks = crop_mask(masks, np.array(bboxes))
        return masks > 0.5

    def detect_books(
        self,
        image   : np.ndarray,
        verbose : bool = False
    ) -> tuple[list[list[float]], np.ndarray]:
        """
        Detects books in the given image.

        Args:
            image   : Input image (BGR)
            verbose : Whether to print inference timing

        Returns:
            tuple: (detections, masks)
        """
        input_tensor      = self.preprocess(image)
        output0, output1  = self.inference(input_tensor, verbose)
        detections, masks = self.post_process(output0, output1)
        logger.debug(f"Detected {len(detections)} books.")
        return detections, masks

# -------------------- BookSegmenter Class --------------------

class BookSegmenter:
    """
    Handles book segmentation from a single full bookshelf image.
    """

    # -------------------- Class Constants --------------------

    PROJECT_ROOT     = Utils.find_root('pyproject.toml')
    MODEL_PATH       = PROJECT_ROOT / 'bookshelf_scanner' / 'core' / 'book_segmenter' / 'models' / 'OpenShelves8.onnx'
    OUTPUT_IMAGE_DIR = PROJECT_ROOT / 'bookshelf_scanner' / 'images' / 'books'

    # -------------------- Initialization --------------------

    def __init__(
        self,
        model_path        : Path  = MODEL_PATH,
        confidence_thresh : float = 0.3,
        iou_thresh        : float = 0.5,
        output_images     : bool  = False,
        output_json       : bool  = False
    ):
        """
        Initializes the BookSegmenter instance.

        Args:
            model_path        : Path to the ONNX model file (defaults to MODEL_PATH)
            confidence_thresh : Confidence threshold for detections
            iou_thresh        : IOU threshold for NMS
            output_images     : Whether to save segmented images to disk
            output_json       : Whether to save segmentation results to a JSON file
        """
        self.model_path        = model_path
        self.confidence_thresh = confidence_thresh
        self.iou_thresh        = iou_thresh
        self.output_images     = output_images
        self.output_json       = output_json

        logger.debug(f"Initializing BookSegmenter with model: {self.model_path}")

        self.yolo = YOLOModel(
            model_path           = self.model_path,
            confidence_threshold = self.confidence_thresh,
            iou_threshold        = self.iou_thresh
        )

        if not self.yolo.check():
            logger.error("Failed to load YOLO model.")
            raise RuntimeError("YOLO model loading failed.")

        if self.output_images and not self.OUTPUT_IMAGE_DIR.exists():
            self.OUTPUT_IMAGE_DIR.mkdir(parents = True, exist_ok = True)

    # -------------------- Image Segmentation Operations --------------------

    def segment_books(self, image_path: Path) -> dict[str, Any]:
        """
        Segments the given image into individual books, optionally saving results to disk.

        Args:
            image_path : Path to the input bookshelf image

        Returns:
            dict: A dictionary containing segmentation results keyed by 'books', 
                  each entry containing information about detected books.
        """
        image = self.load_image(image_path = image_path)
        segments, bboxes, confidences = self.segment_image(image = image)

        results = []
        for i, (seg_img, conf) in enumerate(zip(segments, confidences)):
            seg_name = f"{image_path.stem}_{i+1:03d}{image_path.suffix}" if self.output_images else None

            if self.output_images and seg_name is not None:
                self.save_segmented_image(seg_img, seg_name)

            result = BookSegmentResult(
                file_name   = seg_name,
                confidence  = conf,
                bbox        = bboxes[i],
                image_array = seg_img if not self.output_images else None
            )
            results.append(result)

        results_dict = {
            "books": [
                {
                    "file_name"   : r.file_name,
                    "confidence"  : r.confidence,
                    "bbox"        : r.bbox,
                    "image_array" : None if self.output_images else r.image_array
                }
                for r in results
            ]
        }

        if self.output_json:
            self.save_to_json(results_dict)

        return results_dict

    def segment_image(
        self,
        image     : np.ndarray,
        use_masks : bool = True
    ) -> tuple[list[np.ndarray], list[list[int]], list[float]]:
        """
        Runs the YOLO model on the image to segment it into individual books.

        Args:
            image     : Input image
            use_masks : Whether to apply masks to isolate books

        Returns:
            tuple: (segments, bboxes, confidences)
        """
        detections, masks = self.yolo.detect_books(image)
        segments    = []
        bboxes      = []
        confidences = []

        for i, box in enumerate(detections):
            x1, y1, x2, y2, score, _ = box
            segment = image[y1:y2, x1:x2]

            if use_masks:
                mask = masks[i][y1:y2, x1:x2]
                segment = cv2.bitwise_and(segment, segment, mask = mask.astype(np.uint8))

            segments.append(segment)
            bboxes.append([x1, y1, x2, y2])
            confidences.append(score)

        logger.info(f"Segmented {len(segments)} books from {image.shape[1]}x{image.shape[0]} image.")
        return segments, bboxes, confidences

    # -------------------- Utility Methods --------------------

    @staticmethod
    def load_image(image_path: Path) -> np.ndarray:
        """
        Loads an image from the specified file path.

        Args:
            image_path : Path to the image file

        Returns:
            np.ndarray: Loaded image (BGR format)

        Raises:
            FileNotFoundError: If the image cannot be loaded.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image

    def save_segmented_image(
        self,
        segment      : np.ndarray,
        segment_name : str
    ):
        """
        Saves a segmented book image to the output directory.

        Args:
            segment      : Segmented book image
            segment_name : File name for the output image
        """
        output_path = self.OUTPUT_IMAGE_DIR / segment_name
        cv2.imwrite(str(output_path), segment)
        logger.info(f"Segmented book image saved to {output_path}")

    def save_to_json(self, results: dict[str, Any]
    ):
        """
        Saves segmentation results to a JSON file.

        Args:
            results   : Dictionary of segmentation results
            base_name : Base name for the JSON results file
        """
        output_file = self.PROJECT_ROOT / 'bookshelf_scanner' / 'data' / 'results' / "segmenter.json"
        output_file.parent.mkdir(parents = True, exist_ok = True)

        with output_file.open('w', encoding = 'utf-8') as f:
            json.dump(results, f, ensure_ascii = False, indent = 4)

        logger.info(f"Segmentation results saved to {output_file}")
