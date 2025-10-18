#!/usr/bin/env python3
"""
Balorg AI Command-Line Interface

A simple CLI tool for managing Balorg AI projects and models.
"""

import argparse
import sys
from balorg import __version__


def create_project(args):
    """Create a new Balorg AI project."""
    print(f"Creating new Balorg AI project: {args.name}")
    print("Project structure:")
    print(f"  {args.name}/")
    print(f"  ├── data/")
    print(f"  ├── models/")
    print(f"  ├── notebooks/")
    print(f"  ├── scripts/")
    print(f"  │   └── train.py")
    print(f"  ├── config.json")
    print(f"  └── README.md")
    print("\nProject created successfully!")


def list_models(args):
    """List available models."""
    print("Available Balorg AI models:")
    print("  - NeuralNetwork: Sequential neural network")
    print("  - Model: Base model class for custom implementations")
    print("\nUse 'balorg-cli info <model>' for more details")


def show_info(args):
    """Show information about a specific model or component."""
    print(f"Information for: {args.component}")
    if args.component.lower() == "neuralnetwork":
        print("\nNeuralNetwork:")
        print("  A sequential neural network implementation")
        print("  Supports multiple layer types:")
        print("    - DenseLayer")
        print("    - DropoutLayer")
        print("    - BatchNormLayer")
    else:
        print(f"\nNo information available for '{args.component}'")


def show_version(args):
    """Show Balorg AI version."""
    print(f"Balorg AI version {__version__}")


def train_model(args):
    """Train a model (placeholder)."""
    print(f"Training model from config: {args.config}")
    print("Training not yet implemented in CLI")
    print("Please use the Python API directly")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Balorg AI Command-Line Interface",
        prog="balorg-cli"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )
    
    # Create project command
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new Balorg AI project"
    )
    create_parser.add_argument(
        "name",
        help="Project name"
    )
    create_parser.set_defaults(func=create_project)
    
    # List models command
    list_parser = subparsers.add_parser(
        "list",
        help="List available models"
    )
    list_parser.set_defaults(func=list_models)
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a component"
    )
    info_parser.add_argument(
        "component",
        help="Component name"
    )
    info_parser.set_defaults(func=show_info)
    
    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model"
    )
    train_parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file"
    )
    train_parser.set_defaults(func=train_model)
    
    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information"
    )
    version_parser.set_defaults(func=show_version)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    try:
        args.func(args)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
