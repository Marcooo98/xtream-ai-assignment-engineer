@echo on

rem Set the number of times to execute the script
set NUM_EXECUTIONS=5

rem Set the interval between executions (in seconds)
set INTERVAL=60

rem Loop to execute the script multiple times
:LOOP
if %NUM_EXECUTIONS% gtr 0 (
    rem Execute the Python script
    python train_pipeline.py
    
    rem Decrement NUM_EXECUTIONS
    set /a NUM_EXECUTIONS-=1
    
    rem Wait for the specified interval
    timeout /t %INTERVAL% >nul
    
    rem Go back to the loop
    goto LOOP
)

rem End of script
echo Script executions completed.