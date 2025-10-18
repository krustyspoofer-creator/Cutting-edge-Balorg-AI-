#!/usr/bin/env python3
"""
Example script demonstrating Balorg AI technical mode
"""

from balorg_ai import TechnicalMode


def main():
    """Run technical mode demonstration"""
    print("=" * 60)
    print("Balorg AI - Technical Mode Demonstration")
    print("=" * 60)
    
    # Initialize technical mode
    tech_mode = TechnicalMode()
    
    # Start interactive session
    tech_mode.interactive_session()


if __name__ == "__main__":
    main()
