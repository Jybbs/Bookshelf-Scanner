"""
Dan O'Brien
CS5330, Fall '24
R. Bockman
11/15/24

book_text_match.py explores matching title and author information–interpretted from images with varying confidence–
to a book database created by ReMo
"""

"""
Notes:

* Determine types and formats of outputs from tesseract
* Determine algorithm for checking matching
* rapidfuzz for string comparison
* consider types of matching
	parallel: search

* Library:
	Sorted?

	
* Book Object Format (from Phil, working for REMO), "useful" only
	Book
		properties
			author
			publisher
			defaultEdition
			editions
				items
					edition
					title
			series
			subtitle
			title
		title

* Survey of images
	Most common on spine (from looks of things)
		Title
		Author Last Name
		Author Full Name
		Publisher
		Subtitle

* Proposed Matching Algorithm
1. OCR Text fuzzy match with title
2. If confidence below X, assume bad read or book not in library

"""

#LIBRARIES
import rapidfuzz as rfzz


def main():

	return 1

if __name__ == "__main__":
	main()