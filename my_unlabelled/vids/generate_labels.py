import os
import pandas as pd

# Define the root directory containing subdirectories with images
root_dir = "."

# List to store image data
data = []

# Walk through all subdirectories
for folder, _, files in os.walk(root_dir):
    folder_name = os.path.basename(folder)  # Get only the folder name
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):  # Filter image files
            data.append({"img_folder": folder_name, "img_name": file})

# Create DataFrame
df = pd.DataFrame(data, columns=["img_folder", "img_name"])

# Save DataFrame to CSV (optional)
df.to_csv("image_data.csv", index=False)

# Display DataFrame
print(df)