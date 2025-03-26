#!/usr/bin/env python
"""
CSV File Splitter Utility

This script helps split large CSV files into smaller chunks that can be uploaded
to the Soccer Prediction System. It preserves headers and ensures data integrity.

Usage:
    python split_large_file.py input_file.csv [chunk_size] [output_dir]

Arguments:
    input_file  - Path to the large CSV file to split
    chunk_size  - Number of rows per chunk (default: 500000)
    output_dir  - Directory to save chunks (default: same as input file)
"""

import os
import sys
import pandas as pd
import math
from pathlib import Path

def split_csv(input_file, chunk_size=500000, output_dir=None):
    """
    Split a large CSV file into smaller chunks.
    
    Args:
        input_file (str): Path to input CSV file
        chunk_size (int): Number of rows per chunk
        output_dir (str): Directory to save output files
    
    Returns:
        list: Paths to created chunk files
    """
    # Resolve paths
    input_path = Path(input_file).resolve()
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return []
    
    # Determine output directory
    if output_dir:
        output_path = Path(output_dir).resolve()
    else:
        output_path = input_path.parent
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get base filename without extension
    base_name = input_path.stem
    
    # Count total rows to estimate number of chunks
    print(f"Counting rows in {input_path.name}...")
    total_rows = sum(1 for _ in open(input_path, 'r', encoding='utf-8')) - 1  # Subtract header
    
    total_chunks = math.ceil(total_rows / chunk_size)
    print(f"File contains {total_rows:,} rows (plus header)")
    print(f"Splitting into {total_chunks} chunks of {chunk_size:,} rows each...")
    
    # Read the file in chunks and save each chunk
    chunk_files = []
    
    # First, read just the header
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    
    # Then process in chunks
    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        chunk_num = i + 1
        output_file = output_path / f"{base_name}_part{chunk_num:03d}.csv"
        
        print(f"Writing chunk {chunk_num}/{total_chunks} to {output_file.name}...")
        chunk.to_csv(output_file, index=False)
        chunk_files.append(str(output_file))
    
    print(f"\nDone! Split {input_path.name} into {len(chunk_files)} files:")
    for file in chunk_files:
        print(f"  - {Path(file).name} ({Path(file).stat().st_size / (1024*1024):.1f} MB)")
    
    return chunk_files

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Parse optional arguments
    chunk_size = 500000  # Default chunk size
    output_dir = None    # Default output directory
    
    if len(sys.argv) >= 3:
        try:
            chunk_size = int(sys.argv[2])
        except ValueError:
            print(f"Error: Chunk size must be an integer. Got '{sys.argv[2]}'.")
            sys.exit(1)
    
    if len(sys.argv) >= 4:
        output_dir = sys.argv[3]
    
    # Split the file
    split_csv(input_file, chunk_size, output_dir)

if __name__ == "__main__":
    main() 