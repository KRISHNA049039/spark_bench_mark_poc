@echo off
REM ============================================
REM Start Spark Master on Node 1 (192.168.4.100)
REM ============================================

REM Use Spark bundled inside pyspark package
for /f "delims=" %%i in ('C:\Users\pc\AppData\Local\Python\pythoncore-3.12-64\python.exe -c "import pyspark, os; print(os.path.dirname(pyspark.__file__))"') do set SPARK_HOME=%%i
set PATH=%SPARK_HOME%\bin;%PATH%

REM Find Java — check common locations
if exist "%JAVA_HOME%\bin\java.exe" goto java_ok
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.11.9-hotspot
if exist "%JAVA_HOME%\bin\java.exe" goto java_ok
set JAVA_HOME=C:\Program Files\Java\jdk-17
if exist "%JAVA_HOME%\bin\java.exe" goto java_ok
REM Try to find java in PATH
where java >nul 2>&1
if %ERRORLEVEL%==0 goto java_ok
echo ERROR: Java not found. Install Java 17 from https://adoptium.net/
echo Then set JAVA_HOME in this script.
pause
exit /b 1
:java_ok
echo Using Java: %JAVA_HOME%

set SPARK_LOCAL_HOSTNAME=192.168.4.100
set SPARK_MASTER_HOST=192.168.4.100
set SPARK_PUBLIC_DNS=192.168.4.100

echo ==========================================
echo Starting Spark Master on %SPARK_LOCAL_HOSTNAME%:7077
echo Web UI: http://%SPARK_LOCAL_HOSTNAME%:8080
echo ==========================================

call "%SPARK_HOME%\bin\spark-class.cmd" org.apache.spark.deploy.master.Master ^
  --host %SPARK_LOCAL_HOSTNAME% ^
  --port 7077 ^
  --webui-port 8080

pause
