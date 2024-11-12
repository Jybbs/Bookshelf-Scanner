import cv2
from james_bookshelf_scanner import *

def load_images():
    import os
    image_dir = "images/Shelves"
    for image_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, image_name))
        if image is not None:
            yield image

def display_image(image, screen_res = (1920, 1920)):
    """
    Display the image, scaling it to fit the screen (if necessary)
    """
    scale = min(screen_res[0] / image.shape[1], screen_res[1] / image.shape[0])
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    cv2.imshow('image', image)
    cv2.waitKey(0)

def ensure_odd(n):
    return n | 1

def process_image(image, params, k_size = 11, median_size = 21):
    """
    Preprocess the image. Remove noise, and fine details to segment books
    """
    # Downsample the image
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    # Infer Median Blur Size from size of the image
    median_size = int(max(image.shape[0], image.shape[1]) / 500)
    # Use "Shadow Removal" (Inspired from Jake's code)
    processed_image = image.copy()
    blur_size = ensure_odd(median_size)
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    channel = cv2.split(processed_image)

    # Apply morphological operationseee
    for i, channel in enumerate(channel):
        # Apply median blur
        channel = cv2.medianBlur(channel, blur_size)
        # Apply morphological operations to remove noise
        channel = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernal)
        processed_image[:, :, i] = channel
    
    # Edge detection
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    return edges

def segment_into_books(image):
    """"
    Segment books given image of a bookshelf (preprocessed)
    """
    rho = 1
    theta = np.pi/180
    threshold = 50
    min_line_length = 100
    max_line_gap = 10

    hough_lines = cv2.HoughLinesP(image, rho, theta, threshold, min_line_length, max_line_gap)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    display_image(image)
    return image


def main():
    params = {
        'use_shadow_removal'      : True,
        'shadow_kernel_size'      : 11,

    }
    for image in load_images():
        image = process_image(image, params)
        books = segment_into_books(image)
        # for book in books:
        #     display_image(book) 

if __name__ == "__main__":
    main()

