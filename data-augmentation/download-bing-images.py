import pandas as pd
from bing_image_downloader.downloader import download



def convert_csv_column_to_list(csv_file_name, column_name):
    # Read the CSV file
    df = pd.read_csv(csv_file_name) 
    # Extract Column as a list
    item_list = df[column_name].tolist()
    # Return list
    return item_list

def print_list(list):
    for item in list:
        print(item)

file_path = "real-estate-dataset/initial/Authentic-real-estate-dataset-v2.csv"
list = convert_csv_column_to_list(file_path, "Description")
print_list(list)

import os
def download_bing_images_from_list(item_list, start_index,max_items, max_no_img_per_item, output_image_dir):
    for count, item in enumerate(item_list, start=start_index):
        if count >= max_items:
            print("You've reached maximum number of items")
            break
        query_string = item
        folder_name = str(count+1)
        output_dir = f'{output_image_dir}/{folder_name}'
        try:
            os.makedirs(output_dir, exist_ok=True)
            download(query_string, limit=max_no_img_per_item, output_dir=output_dir, adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
        except OSError as e:
            print(f"Error creating directory: {e}")
        

download_bing_images_from_list(list, 400, 1000, 3, "real-estate-dataset/final/images/authentic")




# import os
# def rename_images (folder_path):
#     for root, folder, files in os.walk(folder_path):
#         for count, file, in enumerate(files, start=1):
#             if file.endswith(".jpg"):
#                 old_name = os.path.join(root, file)
#                 folder_name = os.path.basename(root)
#                 new_name = f'{folder_name}a.{count}.jpg'
#                 os.rename(old_name, new_name)

# rename_images("dataset/final/images")

# import os

# def move_files_to_parent_folders(folder_path):
#     for root, _, files in os.walk(folder_path):
#         print(root)
#         old = root
#         for count, file in enumerate(files, start=1):
#             if file.endswith(".jpg"):
#                 old_path = os.path.join(old, os.path.basename(file))
#                 print("Old: "+old_path)
#                 new_path = os.path.join(root, "file")
#                 print("New: "+new_path)
                
                
#                 os.rename(old_path, new_path)

# # Replace 'images' with the actual path to your 'images' folder
# move_files_to_parent_folders('real-estate-dataset/final/test')