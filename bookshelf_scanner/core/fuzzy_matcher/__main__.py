"""
Entry point for running the text matcher as a module.
"""

from bookshelf_scanner import FuzzyMatcher

def main():

    matcher = FuzzyMatcher()
    matcher.match_books()

if __name__ == "__main__":
    main()