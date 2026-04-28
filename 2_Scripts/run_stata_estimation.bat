@echo off
set STATA_EXE=C:\Stata16\StataMP-64.exe
"%STATA_EXE%" /e do "2_Scripts\3_Estimation\00_run_all.do"
