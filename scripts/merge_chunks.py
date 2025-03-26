#!/usr/bin/env python
"""
CSV Chunk Merger Utility

This script merges CSV chunks that were previously split using split_large_file.py
back into a single CSV file. It preserves headers and ensures data integrity.

Usage:
    python merge_chunks.py [chunk_pattern] [output_file]

Arguments:
    chunk_pattern - Glob pattern matching input chunk files (e.g., "data/file_part*.csv")
    output_file   - Path to save the merged output file
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path

def merge_csv_chunks(chunk_pattern, output_file):
    """
    Merge CSV chunks into a single file.
    
    Args:
        chunk_pattern (str): Glob pattern to match chunk files
        output_file (str): Path to save the merged output
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Find all matching chunk files
    chunk_files = sorted(glob.glob(chunk_pattern))
    
    if not chunk_files:
        print(f"Error: No files found matching pattern '{chunk_pattern}'")
        return False
    
    print(f"Found {len(chunk_files)} chunk files to merge:")
    for file in chunk_files:
        file_size = Path(file).stat().st_size / (1024*1024)
        print(f"  - {Path(file).name} ({file_size:.1f} MB)")
    
    # Prepare output path
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get header from first file
    first_file = pd.read_csv(chunk_files[0], nrows=0)
    header = first_file.columns.tolist()
    
    # Create output file with header
    print(f"\nCreating output file: {output_path}")
    first_file.to_csv(output_path, index=False)
    
    # Append each chunk file without header
    total_rows = 0
    for i, chunk_file in enumerate(chunk_files):
        print(f"Processing chunk {i+1}/{len(chunk_files)}: {Path(chunk_file).name}")
        
        # Read the chunk
        chunk = pd.read_csv(chunk_file)
        total_rows += len(chunk)
        
        # Append to output file without header
        chunk.to_csv(output_path, mode='a', header=False, index=False)
    
    # Calculate final size
    output_size = output_path.stat().st_size / (1024*1024)
    
    print(f"\nMerge complete!")
    print(f"Total rows: {total_rows:,}")
    print(f"Output file: {output_path} ({output_size:.1f} MB)")
    
    return True

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    if len(sys.argv) < 3:
        print(__doc__)
        
        # If at least one argument provided, use it as pattern and ask for output
        if len(sys.argv) == 2:
            chunk_pattern = sys.argv[1]
            output_file = input("Enter path for output file: ")
        else:
            chunk_pattern = input("Enter glob pattern for chunk files (e.g., data/file_part*.csv): ")
            output_file = input("Enter path for output file: ")
    else:
        chunk_pattern = sys.argv[1]
        output_file = sys.argv[2]
    
    # Merge the chunks
    merge_csv_chunks(chunk_pattern, output_file)

if __name__ == "__main__":
    main() 