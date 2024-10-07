import pandas as pd
from transformers import CLIPProcessor 
from PIL import Image
import os
import clip, torch


def create_imageURL_title_pairs(csv_file):
    df = pd.read_csv(csv_file)

    image_urls = df["Image URLs"].str.split(",").tolist()  # Split URLs into a list
    titles = df["Title"].tolist()

    image_title_pairs = []
    for i, url_list in enumerate(image_urls):
        for url in url_list:
            image_title_pairs.append((url.strip(), titles[i]))  # Remove leading/trailing whitespaces
    print(image_title_pairs[:5])  # Print the first 5 tuples
    return image_title_pairs

def create_image_title_pairs(csv_file):
    df = pd.read_csv(csv_file)
    IDs = df["ID"].tolist()
    titles = df["Title"].tolist()
    image_title_pairs = []
    for i, ID in enumerate(IDs):
        if i > 5:
            break
        for i in range(1, 6):
            file_path = f"dataset/authentic_images/A{ID}.{i}.jpg"
            print(file_path)
            image = Image.open(file_path)
            x = int(ID)
            image_title_pairs.append((image, titles[x]))
            print(f'{image}, {titles[x]}')
    print(image_title_pairs[:5])  # Print the first 5 tuples
    
    return image_title_pairs  # Return both image paths and loaded images

# # Define a function to prepare the data for CLIP
# def prepare_inputs(images, titles):
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
#     inputs = processor(images=images, text=titles, return_tensors="pt", truncation=True, padding=True)
#     return inputs

csv_file = "dataset/lamudi_data.csv"
image_title_pairs = create_image_title_pairs(csv_file)
images, titles = zip(*image_title_pairs)

# inputs = prepare_inputs(image_paths, titles)

# OpenAI CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", jit=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

import matplotlib.pyplot as plt
import pandas as pd

# Select indices for three example images
indices = [0, 1, 2]

# Preprocess the text for each title
text_inputs = torch.cat([clip.tokenize(f"a photo of {title}") for title in titles]).to(device)

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop through the indices and process each image
for i, idx in enumerate(indices):
    # Select an example image from the dataset
    image = images[i]
    title = titles[i]

    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Calculate image and text features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Normalize the features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity between image and text features
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)

    # Display the image in the subplot
    axes[i].imshow(image)
    axes[i].set_title(f"Predicted: {titles[indices[0]]}, Actual: {title}")
    axes[i].axis('off')

# Show the plot
plt.tight_layout()
plt.show()