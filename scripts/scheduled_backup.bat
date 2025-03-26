@echo off
REM scheduled_backup.bat - Automated database backup script for Soccer Prediction System (Windows)
REM
REM This script is designed to be run as a scheduled task using Windows Task Scheduler
REM It creates a database backup and can optionally upload it to a remote storage location

setlocal enabledelayedexpansion

REM Change to the project root directory
cd /d "%~dp0\.."

REM Load environment variables from .env file if it exists
if exist .env (
    for /F "tokens=*" %%A in (.env) do set %%A
)

REM Configuration (can be overridden by environment variables)
if not defined KEEP_BACKUPS set KEEP_BACKUPS=10
if not defined UPLOAD_BACKUP set UPLOAD_BACKUP=false
if not defined REMOTE_STORAGE set REMOTE_STORAGE=
if not defined REMOTE_PATH set REMOTE_PATH=
if not defined NOTIFICATION_EMAIL set NOTIFICATION_EMAIL=

REM Ensure logs directory exists
if not exist logs mkdir logs

REM Timestamp for logs
set timestamp=%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=!timestamp: =0!

echo [%date% %time%] Starting scheduled database backup... >> logs\backup.log

REM Create a new backup and clean old ones
echo [%date% %time%] Creating database backup... >> logs\backup.log
python scripts\db_backup.py backup --clean --keep %KEEP_BACKUPS%

REM Get the most recent backup file using a PowerShell command
for /f "tokens=*" %%a in ('powershell -Command "& {$latest = Get-ChildItem -Path 'backups\*.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if($latest) { $metadata = Get-Content $latest.FullName | ConvertFrom-Json; Write-Output ('backups\' + $metadata.backup_file) }}"') do (
    set LATEST_BACKUP=%%a
)

if not defined LATEST_BACKUP (
    echo [%date% %time%] ERROR: Failed to determine latest backup file >> logs\backup.log
    exit /b 1
)

echo [%date% %time%] Latest backup: !LATEST_BACKUP! >> logs\backup.log

REM Upload to remote storage if configured
if "%UPLOAD_BACKUP%"=="true" if not "%REMOTE_STORAGE%"=="" (
    echo [%date% %time%] Uploading backup to remote storage (%REMOTE_STORAGE%)... >> logs\backup.log
    
    for %%F in ("!LATEST_BACKUP!") do set BACKUP_FILENAME=%%~nxF
    
    if "%REMOTE_STORAGE%"=="s3" (
        REM AWS S3
        where aws >nul 2>&1
        if %ERRORLEVEL% EQU 0 (
            aws s3 cp "!LATEST_BACKUP!" "%REMOTE_PATH%/!BACKUP_FILENAME!"
            echo [%date% %time%] Uploaded to S3: %REMOTE_PATH%/!BACKUP_FILENAME! >> logs\backup.log
        ) else (
            echo [%date% %time%] ERROR: aws CLI not found. Install it with 'pip install awscli' >> logs\backup.log
        )
    ) else if "%REMOTE_STORAGE%"=="gcs" (
        REM Google Cloud Storage
        where gsutil >nul 2>&1
        if %ERRORLEVEL% EQU 0 (
            gsutil cp "!LATEST_BACKUP!" "%REMOTE_PATH%/!BACKUP_FILENAME!"
            echo [%date% %time%] Uploaded to GCS: %REMOTE_PATH%/!BACKUP_FILENAME! >> logs\backup.log
        ) else (
            echo [%date% %time%] ERROR: gsutil not found. Install Google Cloud SDK >> logs\backup.log
        )
    ) else if "%REMOTE_STORAGE%"=="azure" (
        REM Azure Blob Storage
        where az >nul 2>&1
        if %ERRORLEVEL% EQU 0 (
            REM Parse container name from REMOTE_PATH
            for /f "tokens=1 delims=/" %%a in ("%REMOTE_PATH%") do set CONTAINER=%%a
            
            REM PowerShell to extract the path after container name
            for /f "tokens=*" %%a in ('powershell -Command "& {'%REMOTE_PATH%' -replace '^[^/]+/',''}}"') do set REMOTE_DIR=%%a
            
            if "!REMOTE_DIR!"=="" (
                set DEST_PATH=!BACKUP_FILENAME!
            ) else (
                set DEST_PATH=!REMOTE_DIR!/!BACKUP_FILENAME!
            )
            
            az storage blob upload --container-name "!CONTAINER!" --file "!LATEST_BACKUP!" --name "!DEST_PATH!"
            echo [%date% %time%] Uploaded to Azure: !CONTAINER!/!DEST_PATH! >> logs\backup.log
        ) else (
            echo [%date% %time%] ERROR: Azure CLI not found. Install it with 'pip install azure-cli' >> logs\backup.log
        )
    ) else (
        echo [%date% %time%] ERROR: Unsupported remote storage type: %REMOTE_STORAGE% >> logs\backup.log
    )
)

REM Send email notification if configured using PowerShell
if not "%NOTIFICATION_EMAIL%"=="" (
    echo [%date% %time%] Sending notification email to %NOTIFICATION_EMAIL%... >> logs\backup.log
    
    REM Get backup size
    for /f "tokens=*" %%a in ('powershell -Command "& {$file = Get-Item '!LATEST_BACKUP!'; $size = if($file.Length -lt 1KB) {$file.Length.ToString() + ' B'} elseif($file.Length -lt 1MB) {'{0:N1} KB' -f ($file.Length / 1KB)} else {'{0:N1} MB' -f ($file.Length / 1MB)}; Write-Output $size}"') do (
        set BACKUP_SIZE=%%a
    )
    
    REM Create email text
    set EMAIL_SUBJECT=Soccer Prediction System - Database Backup Completed
    
    REM Create a temporary file for the email body
    echo Soccer Prediction System - Database Backup Report > email_body.tmp
    echo ======================================== >> email_body.tmp
    echo. >> email_body.tmp
    echo Host: %COMPUTERNAME% >> email_body.tmp
    echo Date: %date% %time% >> email_body.tmp
    echo Backup file: !BACKUP_FILENAME! >> email_body.tmp
    echo Size: !BACKUP_SIZE! >> email_body.tmp
    echo. >> email_body.tmp
    if "%UPLOAD_BACKUP%"=="true" if not "%REMOTE_STORAGE%"=="" (
        echo Remote storage: %REMOTE_STORAGE% >> email_body.tmp
        echo Remote path: %REMOTE_PATH%/!BACKUP_FILENAME! >> email_body.tmp
    )
    echo. >> email_body.tmp
    echo This is an automated message. Please do not reply. >> email_body.tmp
    
    REM Send email using PowerShell
    powershell -Command "& {$EmailFrom = 'noreply@soccerprediction.system'; $EmailTo = '%NOTIFICATION_EMAIL%'; $Subject = '%EMAIL_SUBJECT%'; $Body = Get-Content -Path 'email_body.tmp' | Out-String; $SMTPServer = 'smtp.your-server.com'; $SMTPPort = 587; $Username = 'your-username'; $Password = 'your-password'; $secpasswd = ConvertTo-SecureString $Password -AsPlainText -Force; $mycreds = New-Object System.Management.Automation.PSCredential ($Username, $secpasswd); try { Send-MailMessage -From $EmailFrom -To $EmailTo -Subject $Subject -Body $Body -SmtpServer $SMTPServer -Port $SMTPPort -UseSsl -Credential $mycreds; Write-Output 'Email sent successfully' } catch { Write-Output ('Email send failed: ' + $_.Exception.Message) }}"
    
    REM Clean up temporary file
    del email_body.tmp
)

echo [%date% %time%] Scheduled backup completed successfully >> logs\backup.log
exit /b 0 