from .yolov8 import YOLO_model
import cv2
import numpy as np

class BookSegmenter:
    """
    BookSegmenter class for segmenting an image (of a bookshelf) into individual books
    """
    def __init__(self, model_path = 'BookSegmenter/models/OpenShelves8.onnx'):
        """
        Initialize the BookSegmenter

        Args:
            model_path (str): The path to the model file
        """
        self.yolo = YOLO_model(model_path)

    def check(self):
        """
        Checks if the model is loaded correctly

        Returns:
            bool: Whether the model is loaded correctly
        """
        return self.yolo.check()
    
    def segment(self, image, use_masks = True):
        """
        Segment Image into books

        Args:
            image (np.array): The input image
            use_masks (bool): Whether to use the model's masks to black out the background
        """
        detections, masks = self.yolo.detect_books(image)
        segments = []
        bboxes = []
        confidences = []
        for i, box in enumerate(detections):
            segment = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if use_masks:
                mask = masks[i][int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                segment = cv2.bitwise_and(segment, segment, mask= mask.astype(np.uint8))
            segments.append(segment)
            bboxes.append(box[:4])
            confidences.append(box[4])
        return segments, bboxes, confidences
    
    def display_segmented_books(self, books, confidence):
        """
        Display the segmented books, with their confidence scores as titles

        Args:
            books (List[np.array]): The list of segmented book images
            confidence (List[float]): The list of confidence scores for each book
        """
        import matplotlib.pyplot as plt
        # Display the segmented images
        fig, axes = plt.subplots(1, len(books), figsize=(20, 10))
        for i, book in enumerate(books):
            book = cv2.cvtColor(book, cv2.COLOR_BGR2RGB)
            axes[i].set_title(f"{confidence[i]:.2f}")
            axes[i].imshow(book)
            axes[i].axis("off")
        plt.show()

def main():
    import os

    # Load the image(s)
    image_dir = "images/Shelves"
    image_files = os.listdir(image_dir)
    segmenter = BookSegmenter()
    for image_file in image_files:
        image = cv2.imread(os.path.join(image_dir, image_file))
        # Segment the books
        books, bboxes, confidence = segmenter.segment(image)
        # Display the segmented books
        segmenter.display_segmented_books(books, confidence)

if __name__ == "__main__":
    main()