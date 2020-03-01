from glob import glob
import pandas as pd
import time
import os
from datetime import datetime

done_tickers = []
while True:
	original_tickers = map(lambda x: x.replace('./data/', '').replace('.pkl', ''), glob("./data/*.pkl"))
	accuracy_csv = pd.read_csv("./data/accuracy.csv")
	processed_tickers = accuracy_csv["ticker"].values
	tickers = list(filter(lambda x: x not in processed_tickers and x not in done_tickers, original_tickers))

	os.system("ps -a | grep python | wc -l > process.log")
	total_process = 0
	with open("process.log", "r") as f:
		lines = f.readlines()
		total_process = int(lines[0])
	os.system("rm -f process.log")

	total_tickers = len(tickers)
	if total_process < 10:
		ticker = tickers[0]
		done_tickers.append(ticker)
		os.system("python process-one.py " + ticker + " &")
	else:
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print(current_time, " Total processes:", total_process)

	time.sleep(5 * 60)
