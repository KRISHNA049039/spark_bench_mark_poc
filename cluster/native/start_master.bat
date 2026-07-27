@echo off
REM ============================================
REM Start Spark Master on Node 1 (192.168.4.100)
REM ============================================

set SPARK_HOME=%~dp0spark\spark-3.5.1-bin-hadoop3
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.11.9-hotspot
set PATH=%SPARK_HOME%\bin;%JAVA_HOME%\bin;%PATH%

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
