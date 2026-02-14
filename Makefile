PYTHON := python3

.PHONY: deps download eda train all clean

deps:
	$(PYTHON) -m pip install --user -r requirements.txt

download:
	$(PYTHON) download_ucibank.py

eda: download
	$(PYTHON) eda_bank.py

train: download
	$(PYTHON) train_baseline.py

all: deps download eda train

clean:
	rm -rf reports datasets/classification/*.zip datasets/classification/bank.zip
