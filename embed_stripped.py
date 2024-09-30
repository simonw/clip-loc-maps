import requests, os, time, glob, json, torch
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Set main directory
emb_dir = "embeddings"
os.chdir(emb_dir)

# Select specific range
INPUT_START = 0
INPUT_SIZE = 100000
IIIF_WIDTH = 2000

# Function to load the fine-tuned model
def load_fine_tuned_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", state_dict=None)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Load fine-tuned model
model_path = "/Users/jamiemahowald/Documents/Python/Fine-tuning/fine_tuned_clip_epoch_2.pt"  # Adjust this to your model's file path
model = load_fine_tuned_model(model_path)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create CSV of all images (comment back in to recreate)
''' 
file_counts = pd.read_csv('p1_map_item_file_counts.csv')
file_list = pd.read_csv('p1_map_file_list.csv')
merged_files = pd.merge(file_counts, file_list, how = "left", on = "p1_item_id")
merged_files.to_csv('merged_files.csv', index = False) 
'''

merged_files = pd.read_csv('merged_files.csv')

# Create several_image_files of all rows with file_count == 1
several_image_files = merged_files[merged_files.file_count != 1]
one_image_files = merged_files[merged_files.file_count == 1]

# Remove all entries without proper IIIF formatting and put them in 
# separate "reports.csv" (these are all some kind of multi-page PDF report)
'''
reports = several_image_files[~several_image_files['file_url'].str.contains("pct:")]
reports.to_csv('reports.csv', index=False)
'''

# Begin pipeline

# Updates JSON with IIIF info
def update_dict_with_iiif(row_idx):

    iiif_url = merged_files.iloc[row_idx]['file_url'].replace("pct:100", "2000,")
    iiif_id = iiif_url.split('/')[-5]

    resource_dict = {}
    iiif_json_url = "https://tile.loc.gov/image-services/iiif/"+iiif_id+"/info.json"
    response = requests.get(iiif_json_url)
    if response.status_code == 200:
        iiif_json = response.json()
        resource_dict.update(iiif_json)
        updated_json = resource_dict
    else:
        print(f"Invalid identifier {iiif_json_url}. Status code: {response.status_code}")
        updated_json = None
        

    # print("update_dict_with_iiif: {:.2f} seconds".format(end-start))

    return iiif_id, updated_json

# Updates JSON with embedding info
def update_dict_with_embedding(iiif_id, updated_json):

    url = "https://tile.loc.gov/image-services/iiif/"+iiif_id+"/full/"+str(IIIF_WIDTH)+",/0/default.jpg"
    try:
        image = Image.open(requests.get(url, stream=True).raw)  
    except: 
        print("Could not download image with ID ", iiif_id)
        return iiif_id, updated_json
    image_preprocess = processor(images = image, return_tensors = "pt", padding = True)
    image_embeds = model.get_image_features(**image_preprocess)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    updated_json['embedding'] = image_embeds.squeeze(0).tolist()

    # print("update_dict_with_embedding: {:.2f} seconds".format(end-start))

    return iiif_id, updated_json

# Writes final JSON to directory
def write_json(iiif_id, updated_json):

    json_dir = "jsons"
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    json_path = os.path.join(json_dir, iiif_id + ".json").replace(":", "_")
    if os.path.exists(json_path):
        print(f"File {json_path} already exists")
    with open(json_path, "w") as file:
        json.dump(updated_json, file, indent = 4)

    # print("write_json: {:.2f} seconds".format(end-start))

# Pipeline that runs each of the previous functions:
def download_and_embed_json(row_idx):
    try: 
        iiif_id, first_updated_json = update_dict_with_iiif(row_idx)
        iiif_id, second_updated_json = update_dict_with_embedding(iiif_id, first_updated_json)
        write_json(iiif_id, second_updated_json)
    except TypeError: 
        print("Could not download json with ID ", iiif_id)

if __name__ == "__main__":
    # Set batch size & width of images 
    range_url_list = several_image_files[INPUT_START : INPUT_START + INPUT_SIZE]['file_url']
    range_list = range(INPUT_START, INPUT_START + INPUT_SIZE)
    unprocessed_index = torch.load("unprocessed_index.pt")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Multiprocessing embedding (multiprocessing improves against traditional as the number of images increases)
    start_time = time.time()
    print("starting")
    results = None

    # Multiprocessing in batches of 500
    for i in tqdm(range(0, INPUT_SIZE, 500), desc = "Embedding images"):
        with mp.Pool(processes = 8) as pool:
            results = pool.map(download_and_embed_json, unprocessed_index[i : i + 500])
        pool.close()
    
    end_time = time.time()
    print("Time taken to embed {} files = {:.3f} seconds \n".format(INPUT_SIZE, end_time-start_time))