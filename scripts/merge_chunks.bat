@echo off
echo ===================================
echo CSV Chunk Merger Utility
echo ===================================
echo.
echo This utility helps merge CSV chunks back into a single file.
echo.

if "%~1"=="" (
    echo USAGE:
    echo    merge_chunks.bat "chunk_pattern" output_file.csv
    echo.
    echo EXAMPLE:
    echo    merge_chunks.bat "C:\path\to\file_part*.csv" "C:\path\to\merged_output.csv"
    echo.
    set /p chunk_pattern="Enter glob pattern for chunks (e.g., data\file_part*.csv): "
) else (
    set chunk_pattern=%~1
)

if "%~2"=="" (
    set /p output_file="Enter path for output CSV file: "
) else (
    set output_file=%~2
)

echo.
echo Processing chunks matching: %chunk_pattern%
echo Output file: %output_file%
echo.

:: Run the Python script
python "%~dp0\merge_chunks.py" "%chunk_pattern%" "%output_file%"

echo.
echo ===================================
echo.
echo Press any key to exit...
pause > nul 