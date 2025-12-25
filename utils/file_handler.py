"""File handling utilities."""

import os
from pathlib import Path
from typing import Optional


def read_input_file(path: str) -> str:
    """
    Read content from input file.
    
    Args:
        path: Path to the input file
        
    Returns:
        File content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise IOError(f"Failed to read input file {path}: {str(e)}") from e


def write_output_file(path: str, content: str, force: bool = False) -> None:
    """
    Write content to output file.
    
    Args:
        path: Path to the output file
        content: Content to write
        force: Whether to overwrite existing file
        
    Raises:
        FileExistsError: If file exists and force is False
        IOError: If file cannot be written
    """
    file_path = Path(path)
    
    # Check if file exists
    if file_path.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {path}. Use -f/--force to overwrite."
        )
    
    # Create parent directories if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Failed to write output file {path}: {str(e)}") from e


def detect_file_type(path: str) -> str:
    """
    Detect file type based on extension.
    
    Args:
        path: File path
        
    Returns:
        File extension (e.g., '.txt', '.md') or empty string
    """
    return Path(path).suffix.lower()


def generate_output_path(input_path: str, custom_path: Optional[str] = None) -> str:
    """
    Generate output file path.
    
    Args:
        input_path: Input file path
        custom_path: Custom output path (if provided)
        
    Returns:
        Output file path
    """
    if custom_path:
        return custom_path
    
    input_file = Path(input_path)
    suffix = input_file.suffix
    stem = input_file.stem
    
    # Generate output filename: input_name_output.txt/md
    output_filename = f"{stem}_output{suffix}"
    return str(input_file.parent / output_filename)

