"""
Entry point for running the text extractor as a module.
"""

from bookshelf_scanner import TextExtractor

def main():

    extractor   = TextExtractor(headless = False)
    image_files = extractor.find_image_files('images/books')
    
    params_override = {
        'color_clahe'    : {'enabled' : True},
        'shadow_removal' : {'enabled' : True}
    }
    
    if extractor.headless:
        extractor.run_headless(
            image_files     = image_files,
            params_override = params_override
        )
    else:
        extractor.interactive_experiment(
            image_files     = image_files,
            params_override = params_override
        )

if __name__ == "__main__":
    main()