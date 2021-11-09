import requests
import csv
import os
from tqdm import tqdm

# Request data constants
URL_PREFIX = "https://gliders.ioos.us/erddap/tabledap/"
URL_SUFFIX = "?trajectory%2Cprofile_id%2Ctime%2Clatitude%2Clongitude&distinct()"

DATASET_IDS = [
	"cp_583-20200613T2010",
	"cp_583-20200819T1925",
	"cp_564-20210403T1808",
	"cp_583-20210403T1913",
	"cp_564-20210903T1945",
	"cp_583-20200613T2010-delayed",
	"cp_583-20210403T1913-delayed",
	"cp_376-20200819T1711",
	"cp_339-20200613T2009",
	"cp_339-20210131T1640",
	"cp_340-20200926T0614",
	"cp_340-20201121T1930",
	"cp_376-20210131T1800",
	"cp_379-20200819T1718",
	"cp_380-20200613T2124",
	"cp_380-20201121T1744",
	"cp_387-20201121T1740",
	"cp_388-20200613T2149",
	"cp_514-20200613T2130",
	"cp_379-20210131T1757-delayed",
	"cp_336-20190516T2337",
	"cp_559-20210131T1801",
	"cp_379-20210517T2048",
	"cp_388-20210517T2025",
	"cp_559-20210517T2034",
	"cp_559-20210903T2253",
	"cp_339-20210904T0031",
	"cp_340-20210620T1735",
	"cp_376-20210903T2255",
	"cp_379-20210903T2251",
	"cp_376-20210131T1800-delayed",
	"cp_336-20190408T0014-delayed",
	"cp_380-20200613T2124-delayed",
	"cp_339-20210131T1640-delayed",
	"cp_387-20190408T0135-delayed",
	"cp_379-20190927T1222-delayed",
	"cp_340-20191213T0104-delayed",
	"cp_376-20190618T2336-delayed",
	"cp_340-20190927T1226-delayed",
	"cp_376-20200302T0946-delayed",
	"cp_379-20200302T0944-delayed",
	"cp_380-20191212T2231-delayed",
	"cp_339-20210402T1334-delayed",
	"cp_379-20190618T2314-delayed",
	"cp_388-20200613T2149-delayed",
	"cp_388-20190618T2258-delayed",
	"cp_514-20200613T2130-delayed",
	"cp_514-20200302T0958-delayed",
	"cp_340-20210620T1735-delayed",
	"cp_559-20210131T1801-delayed",
	"cp_339-20200302T1109-delayed",
	"cp_339-20200613T2009-delayed",
	"cp_339-20200302T1109",
	"cp_376-20200302T0946",
	"cp_379-20200302T0944",
	"cp_388-20191212T2247-delayed",
	"cp_388-20210517T2025-delayed",
	"cp_559-20210517T2034-delayed",
	"cp_379-20210517T2048-delayed",
	"cp_335-20160527T2033",
	"cp_335-20170116T1459",
	"cp_340-20160121T1708",
	"cp_374-20161011T0106",
	"cp_335-20141006T2016",
	"cp_335-20151013T0112",
	"cp_335-20160404T1857",
	"cp_336-20161011T0027",
	"cp_336-20180724T1433",
	"cp_339-20150112T0601",
	"cp_339-20170606T0355",
	"cp_340-20150506T0237",
	"cp_340-20160809T0230",
	"cp_374-20150509T0311",
	"cp_374-20160529T0035",
	"cp_380-20170607T0254",
	"cp_388-20160809T1409",
	"cp_389-20151013T0150",
	"cp_564-20170817T1020",
	"cp_336-20170817T1159",
	"cp_336-20180126T0000",
	"cp_339-20160121T1629",
	"cp_339-20171029T0452",
	"cp_339-20180126T0000",
	"cp_376-20151012T2326",
	"cp_376-20160121T1515",
	"cp_376-20160527T2050",
	"cp_379-20150509T1102",
	"cp_379-20160121T1500",
	"cp_379-20170116T1246",
	"cp_380-20161011T2046",
	"cp_380-20171101T0150",
	"cp_380-20180126T0000",
	"cp_387-20150111T1716",
	"cp_387-20151014T0119",
	"cp_387-20170419T2053",
	"cp_388-20151022T1034",
	"cp_389-20161011T2040",
	"cp_389-20170419T2114",
	"cp_389-20180724T1620",
	"cp_376-20180724T1552",
	"cp_387-20160404T1858",
	"cp_387-20170419T0000",
	"cp_389-20170419T0000",
	"cp_339-20170116T2353",
	"cp_336-20170116T1254",
	"cp_388-20170116T1324"
]

# Write data to location
OUT_DIR = "../data/raw"

# Example request URL
# "https://gliders.ioos.us/erddap/tabledap/cp_583-20200613T2010.csv?trajectory%2Cprofile_id%2Ctime%2Clatitude%2Clongitude&distinct()"


def dedup_dataset_ids():
	dataset_ids = []
	for dataset_id in DATASET_IDS:
		# Only append delayed-mode trajectory if realtime trajectory is not available
		if "-delayed" in dataset_id:
			if dataset_id[0: -8] not in DATASET_IDS:
				dataset_ids.append(dataset_id)
		else:
			dataset_ids.append(dataset_id)
	return dataset_ids


def main():
	dataset_ids = dedup_dataset_ids()

	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)

	for dataset_id in tqdm(dataset_ids):
		fname = dataset_id + ".csv"
		url = URL_PREFIX + fname + URL_SUFFIX
		response = requests.get(url)
		with open(os.path.join(OUT_DIR, fname), 'w') as f:
			writer = csv.writer(f)
			for line in response.iter_lines():
				writer.writerow(line.decode('utf-8').split(','))


if __name__ == '__main__':
	main()