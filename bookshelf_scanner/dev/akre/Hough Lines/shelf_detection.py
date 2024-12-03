import cv2

def ocr(image):
    import easyocr
    reader = easyocr.Reader(['en'], gpu = True)
    result = reader.readtext(image)
    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        #color text from red to green based on the confidence
        color = (0, 0, int(255 * (1 - prob)))
        cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('image', image)
    cv2.waitKey(0)

#Display the image
def display_image(image):
    screen_res = 1080, 1920
    scale = min(screen_res[0] / image.shape[1], screen_res[1] / image.shape[0])
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    cv2.imshow('image', image)
    cv2.waitKey(0)

#Function to segment books from the bookshelf
def segment_into_shelves(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur
    gray = cv2.GaussianBlur(gray, (51,51),0)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    thresh = cv2.erode(thresh, kernel, iterations=3)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    # Filter contours based on area
    image_area = image.shape[0] * image.shape[1]
    min_area = 0.1 * image_area
    max_area = 0.9 * image_area
    contours = [contour for contour, bbox in zip(contours, bounding_boxes) if min_area < bbox[2] * bbox[3] < max_area]
    # Segment shelves 
    shelves = []
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
         # Draw contours
        cv2.drawContours(thresh, [contour], -1, (0, 255, 0), 3)
        cv2.rectangle(thresh, (x, y), (x+w, y+h), (0, 0, 255), 10)
        shelves.append(image[y:y+h, x:x+w])
    # Display the image
    display_image(thresh)
    return shelves

def main():
    image_names = ['IMG_1538.JPG', 'IMG_1539.JPG', 'IMG_1346.JPG', 'IMG_1347.JPG', 'IMG_1348.JPG']
    actual_shelves = [3, 3, 4, 4, 4]
    for image_name in image_names:
        image = cv2.imread(f'images/bookcases/{image_name}')
        assert image is not None
        print(f"Processing {image_name}")
        # turn image sideways
        # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        shelves = segment_into_shelves(image)
        print(f"Found {len(shelves)} of {actual_shelves.pop(0)} shelves.")

if __name__ == '__main__':
    main()
