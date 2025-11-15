import os
import shutil
from time import time
from datetime import datetime
def find_gold_files(root_dir, target_dir):
    # Walk through all directories and subdirectories
    start_time = time()
    print("started")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        largest_file = None
        largest_size = 0
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                # Get the size of the file
                file_size = os.path.getsize(file_path)
                # Check if this is the largest file we've seen
                if file_size > largest_size:
                    largest_size = file_size
                    largest_file = file_path
            except Exception as e:
                print(f"Error accessing file {file_path}: {e}")

        os.makedirs(target_dir, exist_ok=True)
        target_file_path = os.path.join(target_dir, os.path.basename(largest_file))

        try:
            # Copy the largest file to the target directory
            shutil.copy(largest_file, target_file_path)
        except:
            pass
        current_time = time()
        if current_time - start_time > 2:
            print("finished")
            break


def find_fin_reg_files(root_dir):

    cd = os.getcwd()
    folder = "FINANCIAL_REGULATION"

    target_dir = os.path.join(cd, folder)

    start_time = time()
    print("started")
    os.makedirs(target_dir, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            
            file_path = os.path.join(dirpath, filename)
            target_file_path = os.path.join(target_dir, os.path.basename(filename))

            try:
                # Copy the largest file to the target directory
                shutil.copy(file_path, target_file_path)
            except:
                pass
            current_time = time()
            if current_time - start_time > 2:
                print("finished")
                break

# "C:\Users\joelj\Downloads\bofjunction_dataset_001\bofjunction_dataset\gold\FINANCIAL_REGULATION_EU_AND_LOCAL_IN_FORCE_GOLD\output_simple"

find_fin_reg_files(r"C:\Users\joelj\Downloads\bofjunction_dataset_001\bofjunction_dataset\gold\FINANCIAL_REGULATION_EU_AND_LOCAL_IN_FORCE_GOLD\output_simple")