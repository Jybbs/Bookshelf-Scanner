import numpy as np
import cv2

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def crop_mask(masks, bboxes):
    """
    Crop the masks using the bounding boxes.
    Args:
    masks (np.ndarray): [n, h, w]
    bboxes (np.ndarray): [n, 4]
    Returns:
    (np.ndarray): Cropped masks.
    """
    n, h, w = masks.shape
    cropped_masks = np.zeros_like(masks)
    for i in range(n):
        x1, y1, x2, y2 = bboxes[i]
        cropped_masks[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
    return cropped_masks

def display_image(image):
    import matplotlib.pyplot as plt
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.show()