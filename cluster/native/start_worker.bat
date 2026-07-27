@echo off
REM ============================================
REM Start Spark Worker (run on Node 2 / Node 3)
REM ============================================

set SPARK_HOME=%~dp0spark\spark-3.5.1-bin-hadoop3
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.11.9-hotspot
set PATH=%SPARK_HOME%\bin;%JAVA_HOME%\bin;%PATH%

REM === CHANGE THIS to the worker machine's own LAN IP ===
set SPARK_LOCAL_HOSTNAME=192.168.4.101
set SPARK_PUBLIC_DNS=192.168.4.101

REM === Master IP (Node 1) ===
set MASTER_IP=192.168.4.100

echo ==========================================
echo Starting Spark Worker
echo Worker IP: %SPARK_LOCAL_HOSTNAME%
echo Master: spark://%MASTER_IP%:7077
echo Memory: 28g, Cores: all
echo ==========================================

call "%SPARK_HOME%\bin\spark-class.cmd" org.apache.spark.deploy.worker.Worker ^
  spark://%MASTER_IP%:7077 ^
  --host %SPARK_LOCAL_HOSTNAME% ^
  --memory 28g ^
  --webui-port 8081

pause
