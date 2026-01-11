# PowerShell script to create Windows Scheduled Task for Stock Monitor
# Run this as Administrator

# Get Python path (adjust if needed)
$pythonPath = (Get-Command python).Source
if (-not $pythonPath) {
    $pythonPath = "python"  # Will use PATH
}

$scriptPath = "C:\Users\r00t4\trade\monitor.py"
$workingDir = "C:\Users\r00t4\trade"

# Create the task trigger (runs every 1 hour at :02 minutes)
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration ([TimeSpan]::MaxValue)

# Create the action
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument $scriptPath -WorkingDirectory $workingDir

# Create the principal (run with highest privileges)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Register the task
Register-ScheduledTask `
    -TaskName "Stock Monitor" `
    -Trigger $trigger `
    -Action $action `
    -Principal $principal `
    -Description "Run stock analysis every hour at :02 minutes" `
    -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries)

Write-Host "Task 'Stock Monitor' created successfully!" -ForegroundColor Green
Write-Host "The task will run every hour starting from the next :02 minute." -ForegroundColor Yellow

# List the task to confirm
Get-ScheduledTask -TaskName "Stock Monitor" | Format-List
