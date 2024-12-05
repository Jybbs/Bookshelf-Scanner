"""
Entry point for running the text extractor as a module.
"""

from bookshelf_scanner import TextExtractor

def main():

    extractor       = TextExtractor(headless = False)
    image_files     = extractor.find_image_files('Books')
    params_override = {
        'color_clahe'    : {'enabled': True},
        'shadow_removal' : {'enabled': True},
        'image_rotation' : {'enabled': True}
    }
    
    if extractor.headless:
        extractor.run_headless_mode(
            image_files     = image_files,
            params_override = params_override
        )
    else:
        extractor.interactive_mode(
            image_files     = image_files,
            params_override = params_override
        )

if __name__ == "__main__":
    main()
