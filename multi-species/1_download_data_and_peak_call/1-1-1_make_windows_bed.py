import sys
import os

RAW_DATA_DIR	= sys.argv[1]
GENOME			= sys.argv[2]

# These parameters decide the length and stride of the windows created
# across all chromosomes. The window size should be consistent with the
# expected input sequence of the model.
WINDOW_SIZE		= int(sys.argv[3])
WINDOW_STRIDE	= int(sys.argv[4])

# Created by genomepy
CHROMOSOME_SIZE_FILE	= f"{RAW_DATA_DIR}/{GENOME}/{GENOME}.fa.sizes"
OUT_FILE				= f"{RAW_DATA_DIR}/{GENOME}/windows.unfiltered.bed"

def make_windows():
	if (os.path.exists(OUT_FILE)):
		print("This file already exists! delete it if you want to make a new one...")
	else:
		with open(CHROMOSOME_SIZE_FILE, "r") as gInfoFile, open(OUT_FILE, "w") as outFile:
			for chromLine in gInfoFile:
				chrom,length = chromLine.strip().split()
				length = int(length)
				window_start = 0
				while window_start + WINDOW_SIZE < length:
					line = "\t".join([chrom, str(window_start), str(window_start + WINDOW_SIZE)])
					outFile.write(line + "\n")
					window_start += WINDOW_STRIDE

if __name__ == "__main__":
	make_windows()