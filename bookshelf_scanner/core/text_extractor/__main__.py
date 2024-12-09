"""
Entry point for running the text extractor as a module.
"""

from bookshelf_scanner import TextExtractor

def main():
    extractor = TextExtractor()
    
    params = {
        'image_files': extractor.find_image_files('Books'),

        'config_override': {
            'steps': {
                'color_clahe'    : {'enabled' : True},
                'shadow_removal' : {'enabled' : True},
                'image_rotation' : {'enabled' : True}
            }
        }
    }
    
    extractor.run_interactive_mode(**params)

if __name__ == "__main__":
    main()
