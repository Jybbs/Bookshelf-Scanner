# BookSegmenter 

BookSegmenter currently uses a pre-trained YOLOv8 model fine called [OpenShelves (version 8)](https://universe.roboflow.com/capjamesg/open-shelves/model/8). This model is fined tuned on the [OpenShelves dataset](https://universe.roboflow.com/capjamesg/open-shelves) and is capable of detecting books in images. 

I managed to get the ONNX model, and implement a custom inference class so it does not need API keys or an internet connection to run. This also ensures that any future changes to the model will not affect the functionality of the BookSegmenter.

## Demo

Do the following from the project root to quickly check if the BookSegmenter is working on your machine:

```bash
PYTHON_PATH=(pwd) python BookSegmenter/BookSegmenter_base.py
```

## Usage
BookSegmenter is packaged as a Python module. To import the module into your file, you can use the following code:

```python
from booksegmenter.BookSegmenter_base import BookSegmenter 
```

The [BookSegmenter_base.py](BookSegmenter/BookSegmenter_base.py) contains the class definition and methods for the BookSegmenter. A simple example of how to use the BookSegmenter is shown below:

```python
from booksegmenter.BookSegmenter_base import BookSegmenter

# Initialize the BookSegmenter
segmenter = BookSegmenter()

# Load an image
image_path = "path/to/image.jpg"

# Segment the image
books, confidences = segmenter.segment_image(image_path)

# Display the segmented books
segmenter.display_books(image_path, books, confidences)
```





## Future Plans

- [ ] Optimize the methods for faster processing. Currently, the model takes about 1-2 seconds
- [ ] Implement a custom YOLOv11 model using a bigger dataset of bookshelf images