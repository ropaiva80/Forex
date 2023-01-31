:: https://stackoverflow.com/questions/62984477/running-python-scripts-in-anaconda-environment-through-windows-cmd
:: call "<condapath>\Scripts\activate.bat" <env_name> & cd "<folder_for_your_py_script>" & python <scriptname.py> [<arguments>]

:: Day_Trading_BTC_1h COINBASE

:: BTC_COINBASE_REPORT_DAY_TRADING_1H
call "C:\Users\ropai\anaconda3\Scripts\activate.bat" Python3.7_Env-GluonTS_off_1h_simple & cd "C:\Python\Crypto\Day_Trading\1H" & python Script_Report_BTC_1H_Datetimes1H_AGO.py