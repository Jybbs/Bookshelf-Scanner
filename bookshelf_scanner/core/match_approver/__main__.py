"""
Entry point for running the match approver as a module.
"""

from bookshelf_scanner import MatchApprover

def main():
    
    approver = MatchApprover()
    approver.run_interactive_mode()

if __name__ == "__main__":
    main()
