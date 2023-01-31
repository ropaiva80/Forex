:: https://stackoverflow.com/questions/62984477/running-python-scripts-in-anaconda-environment-through-windows-cmd
:: call "<condapath>\Scripts\activate.bat" <env_name> & cd "<folder_for_your_py_script>" & python <scriptname.py> [<arguments>]

:: Day_Trading_BTC_1h YAHOO

::Script4
call "C:\Users\ropai\anaconda3\Scripts\activate.bat" Python3.7_Env-GluonTS_off_1h_simple & cd "C:\Python\Crypto\Day_Trading\1H" & python Script4_BTC-YAHOO_DAILY-SKlearn_ML-Hourly-700D_1H.py

::Script3
call "C:\Users\ropai\anaconda3\Scripts\activate.bat" Python3.7_Env-GluonTS_off_1h_simple & cd "C:\Python\Crypto\Day_Trading\1H" & python Script3_BTC-v2_Arima1H-Yahoo-30D.py

::Script2
call "C:\Users\ropai\anaconda3\Scripts\activate.bat" Python3.7_Env-GluonTS_off_1h_simple & cd "C:\Python\Crypto\Day_Trading\1H" & python Script2_BTC_YAHOO_1H-Autots_GluonTS-OFF_Simple-30D_v1.py


