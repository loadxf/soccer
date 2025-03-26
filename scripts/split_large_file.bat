@echo off
echo ===================================
echo Large CSV File Splitter Utility
echo ===================================
echo.
echo This utility helps split large CSV files into smaller chunks
echo that can be uploaded to the Soccer Prediction System.
echo.

if "%~1"=="" (
    echo USAGE:
    echo    split_large_file.bat input_file.csv [chunk_size] [output_directory]
    echo.
    echo EXAMPLE:
    echo    split_large_file.bat "C:\path\to\large_file.csv" 500000 "C:\path\to\output"
    echo.
    set /p input_file="Enter path to input CSV file: "
) else (
    set input_file=%~1
)

if "%~2"=="" (
    set chunk_size=500000
    echo Using default chunk size: 500,000 rows per file
) else (
    set chunk_size=%~2
)

if "%~3"=="" (
    set output_dir=""
    echo Files will be saved in the same directory as the input file
) else (
    set output_dir=%~3
    echo Output directory: %output_dir%
)

echo.
echo Processing file: %input_file%
echo Chunk size: %chunk_size% rows
echo.

:: Run the Python script
if %output_dir%=="" (
    python "%~dp0\split_large_file.py" "%input_file%" %chunk_size%
) else (
    python "%~dp0\split_large_file.py" "%input_file%" %chunk_size% "%output_dir%"
)

echo.
echo ===================================
echo.
echo Press any key to exit...
pause > nul 