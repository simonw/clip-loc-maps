import requests, os, time, glob, json
import multiprocessing as mp
import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel   
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Set main directory
emb_dir = "embeddings"
os.chdir(emb_dir)

# Create CSV of all images
file_counts = pd.read_csv('p1_map_item_file_counts.csv')
file_list = pd.read_csv('p1_map_file_list.csv')
merged_files = pd.merge(file_counts, file_list, how = "left", on = "p1_item_id")
merged_files.to_csv('merged_files.csv', index = False)

# Create one_image_files of all rows with file_count == 1
one_image_files = merged_files[merged_files.file_count == 1]

# Remove all entries without proper IIIF formatting and put them in 
# separate "reports.csv" (these are all some kind of multi-page PDF report)
reports = one_image_files[~one_image_files['file_url'].str.contains("pct:")]
reports.to_csv('reports.csv', index=False)
one_image_file = one_image_files[one_image_files['file_url'].str.contains("pct:")]

# Select specific range
input_start = len(one_image_files["file_url"])-7000
input_size = 10
iiif_width = 2000

# Begin pipeline
# 1. Download image to drive
def download_image(iiif_url):

    # Makes "images/"" folder to place images into
    img_dir = "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    # Writes the image from url into "images" folder
    # The input url should be of the form:
    # https://tile.loc.gov/image-services/iiif/[iiif_id]/full/pct:100/0/default.jpg
    iiif_response = requests.get(iiif_url)
    iiif_id = iiif_url.split('/')[-5]
    if iiif_response.status_code == 200:
        img_path = os.path.join(img_dir, iiif_id+".jpg")
        with open(img_path, "wb") as img_file:
            img_file.write(iiif_response.content)
    else:
        print(f"Could not download image from url {iiif_url}")
    return iiif_id

# Now, we need to download the LOC JSON and the IIIF JSON
# and merge them into one file.

# Modifies iiif_id into a resource_id
# that is readable by the loc.gov API
def iiif2resource(iiif_id):
    # Modifies iiif_id into a resource_id
    # that is readable by the loc.gov API  
    id_list = iiif_id.split(":")
    if len(id_list) == 6:
        resource_id = id_list[-2]+"."+id_list[-1]
    elif len(id_list) == 7:
        resource_id = id_list[-3]+"."+id_list[-2]
    elif len(id_list) == 10 :
        resource_id = id_list[2]+"."+id_list[-2]
    else: resource_id = id_list[-2]+"."+id_list[-1]
    return resource_id

# 2. Creates the resource (i.e., LOC) JSON
def create_resource_json(iiif_id):
    resource_id = iiif2resource(iiif_id)
    if resource_id == None:
        resource_id = ''
    resource_json_url = "https://www.loc.gov/resource/" + resource_id + "?fo=json"
    response = requests.get(resource_json_url)
    if response.status_code == 200:
        resource_json = response.json()
    else:
        print(f"First invalid identifier {iiif_id}. Status code: {response.status_code}")
        resource_json = None
    return iiif_id, resource_json

# 3. Updates JSON with IIIF info
def update_json_with_iiif(iiif_id, resource_json):
    if resource_json is None:
        return
    iiif_json_url = "https://tile.loc.gov/image-services/iiif/"+iiif_id+"/info.json"
    response = requests.get(iiif_json_url)
    if response.status_code == 200:
        iiif_json = response.json()
        resource_json.update(iiif_json)
        updated_json = resource_json
    else:
        print(f"Second invalid identifier {iiif_id}. Status code: {response.status_code}")
        updated_json = None
    return iiif_id, updated_json

# 4. Updates JSON with embedding info
def update_json_with_embedding(iiif_id, updated_json):
    url = "https://tile.loc.gov/image-services/iiif/"+iiif_id+"/full/"+str(iiif_width)+",/0/default.jpg"
    image = Image.open(requests.get(url, stream=True).raw)  
    image_preprocess = processor(images = image, return_tensors = "pt", padding = True)
    image_embeds = model.get_image_features(**image_preprocess)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    updated_json['embedding'] = image_embeds.squeeze(0).tolist()
    return iiif_id, updated_json

# 5. Writes final JSON to directory
def write_json(iiif_id, updated_json):
    json_dir = "jsons"
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    json_path = os.path.join(json_dir, iiif_id + ".json")
    # if os.path.exists(json_path):
    #     print(f"File {iiif_id}.json already exists")
    with open(json_path, "w") as file:
        json.dump(updated_json, file, indent = 4)

# Pipeline that runs each of the previous functions:
def download_and_embed_json(url):
    iiif_id = download_image(url)
    iiif_id, resource_json = create_resource_json(iiif_id)
    try: 
        iiif_id, first_updated_json = update_json_with_iiif(iiif_id, resource_json)
        iiif_id, second_updated_json = update_json_with_embedding(iiif_id, first_updated_json)
        write_json(iiif_id, second_updated_json)
    except TypeError: 
        print("Could not download json with ID ", iiif_id)

if __name__ == "__main__":
    # Set batch size & width of images
    range_url_list = one_image_files[input_start : input_start + input_size]['file_url']
    new_range_url_list = pd.read_csv('new_range_url_list.csv')['url']
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Multiprocessing embedding (multiprocessing improves against traditional as the number of images increases)
    start_time = time.time()
    results = None
    url_list = pd.Series([url.replace("pct:100", f"{iiif_width},") for url in new_range_url_list])
    with mp.Pool(processes = 7) as pool:
        results = pool.map(download_and_embed_json, url_list)
    pool.close()
    
    # Deletes images
    for file in glob.glob("*/*.jpg"):
        os.remove(file)
    end_time = time.time()
    print("Time taken to embed {} files = {:.3f} seconds \n".format(len(new_range_url_list), end_time-start_time))
