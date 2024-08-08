import os, pathlib
import pandas as pd
trip_fare_path = "/Users/ryan/Library/Mobile Documents/com~apple~CloudDocs/datasets/nyctaxi/trip_fare"


# get list of file names in the trip_fare_path directory
trip_fare_file_list = os.listdir(trip_fare_path)
print(trip_fare_file_list)

df = pd.read_csv(trip_fare_path + "/" + trip_fare_file_list[0])
print(df.head())