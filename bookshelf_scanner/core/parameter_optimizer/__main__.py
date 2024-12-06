"""
Entry point for running the parameter optimizer as a module.
"""

from bookshelf_scanner import ParameterOptimizer, TextExtractor

def main():

    extractor   = TextExtractor(headless = True)
    optimizer   = ParameterOptimizer(extractor = extractor)
    image_files = TextExtractor.find_image_files(subdirectory = 'Books')

    # Run the optimization process
    optimizer.optimize(image_files)

if __name__ == "__main__":
    main()
