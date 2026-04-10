#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:01:33 2026

@author: prakashgawas
"""

import os
import csv
from datetime import datetime

# Change this to your folder path
folder_path = "NN_iter800_runs5_um1_new1_stoch1_sc30_van1_gat0_tl30_mg0.02_nc5/"

# Output CSV file name
output_csv = "files_with_NN_creation_times.csv"

with open(output_csv, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    # Write header
    writer.writerow(["File Name", "Creation Time"])

    # Loop through files
    for root, dirs, files in os.walk(folder_path):
        for filename in files:

            # Select only files containing "NN"
            if "NN" in filename:
                file_path = os.path.join(root, filename)

                creation_time = os.path.getctime(file_path)
                readable_time = datetime.fromtimestamp(creation_time)

                writer.writerow([file_path, readable_time])

print(f"CSV file '{output_csv}' created successfully ✅")