import pandas as pd
from tqdm import tqdm
import json
import random, requests, os, time, torch
import multiprocessing as mp
from datetime import datetime

# Set working directory
os.chdir("fine-tuning")

# Load data
merged_files = pd.read_csv("merged_files.csv")
# Subset of merged_files such that p1_item_id does not contain "sanborn" and file_count < 10
merged_files['p1_item_id'] = merged_files['p1_item_id'].fillna('')
merged_files['file_count'] = merged_files['file_count'].fillna(0)
subset = merged_files[~merged_files['p1_item_id'].str.contains("sanborn") & (merged_files['file_count'] < 10)]

def get_resource(row):
    iiif_url = subset.iloc[row]['file_url'].replace("pct:100", "2000,")
    resource_json_url = subset.iloc[row]['resource_url'] + "?fo=json&sp=" + str(int(subset.iloc[row]['segment_num'] + 1))
    response = requests.get(resource_json_url)
    if response.status_code == 200:
        return iiif_url, response.json()
    else:
        print(f"Invalid identifier {resource_json_url}. Status code: {response.status_code}")
        return None, None

def remove_last_period(s):
    return s[:-1] if s.endswith('.') else s

def get_keys(dict):
    return [list(key.keys())[0] for key in dict]

def process_location(item, location_key, title):
    location = get_keys(item.get(location_key)) if item.get(location_key) else None
    if location:
        location = ", and ".join(location)
        if location.lower() in title.lower():
            location = None
    return location

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def format_date(date_string):
    try:
        date_obj = datetime.strptime(date_string, "%Y-%m-%d")
        if date_obj.strftime("%m-%d") == "01-01":
            return date_obj.strftime("%Y")
        return [
            date_obj.strftime(fmt) for fmt in
            ["%B %d, %Y", "%d %b %Y", "%Y-%m-%d", "%m/%d/%Y", "%m.%d.%Y"]
        ]
    except ValueError:
        try:
            date_obj = datetime.strptime(date_string, "%Y-%m")
            return [
                date_obj.strftime(fmt) for fmt in
                ["%B, %Y", "%b %Y", "%Y-%m", "%m/%Y", "%Y.%m"]
            ]
        except ValueError:
            return date_string
    except TypeError:
        date_obj = None
        return date_obj

def get_caption(json_file):
    item = json_file['item']
    title = item['title'].strip("[]").split(":")[0]
    date = item.get('date')
    format_type = list(item.get('format')[0])[0]
    format_type = format_type if format_type.lower() not in title.lower() else None
    if format_type is None: 
        format_type = "map"
    location = item.get('locations')
    city, county, state, country = None, None, None, None
    
    if location:
        city = process_location(item, 'locations_city', title)
        county = process_location(item, 'locations_county', title)
        state = process_location(item, 'locations_state', title)
        country = process_location(item, 'locations_country', title)

    location_parts = [loc for loc in [city, county, state, country] if loc is not None]
    location = ", ".join([loc.title() for loc in location_parts])

    notes = item.get('notes', [])
    notes_set = {note for note in notes[:-2] if "scale" in note.lower() or not has_numbers(note)}
    notes_set = {note for note in notes_set if "division" not in note.lower()}
    notes = ". ".join([remove_last_period(note) for note in notes_set])

    date = format_date(date)
    if isinstance(date, list):
        date = random.choice(date)
    if date and date in title:
        date = None

    caption = ""
    if not any(word in title.lower() for word in ["map", "survey", "plan", "chart", "atlas", format_type.lower()]):
        caption += f"A {format_type} of "
    caption += f"{title}"
    if date:
        caption += f" from {date}"
    caption += f", in {location}. {notes}."
    caption = caption.replace("  ", " ").replace(". .", ".")
    return caption

def download_image(row):
    retry_count = 0
    max_retries = 5

    iiif_url, json_file = get_resource(row)
    if iiif_url is None or json_file is None:
        return None, None

    caption = get_caption(json_file)

    while retry_count < max_retries:
        try: 
            response = requests.get(iiif_url, timeout=5)
            if response.status_code == 200:
                save_path = iiif_url.split("/")[-5]
                save_path = save_path.replace(":", "_")
                image_path = f"images_new/train/{save_path}.jpg"
                with open(image_path, "wb") as file:
                    file.write(response.content)
                return image_path, caption
                    
            else: 
                print(f"Retrying {row} due to status code: {response.status_code}")
                retry_count += 1
                print("Retry: ", retry_count)
                time.sleep(2 ** retry_count)
                return None, None
    
        except Exception as e:
            print(f"Retrying {row} due to error: {e}")
            retry_count += 1
            print("Retry: ", retry_count)
            time.sleep(2 ** retry_count)
            return None, None

def main():
    random.seed(46)
    rand_ints = random.sample(range(0, len(subset)), 10000)
    chunk_size = 50

    list_image_path = []
    list_txt = []

    start = time.time()

    for i in tqdm(range(0, len(rand_ints), chunk_size), desc="Downloading chunk: "):
        chunk_indices = rand_ints[i:i + chunk_size]
        with mp.Pool(processes=5) as pool:
            try:
                results = pool.map(download_image, chunk_indices)
            except Exception as e:
                print(f"Error encountered: {e}")
                continue

        # Uncomment this line if you want to download images sequentially
        # results = [download_image(row) for row in chunk_indices]

        results = [result for result in results if result is not None]
        if results:
            paths, titles = zip(*results)       
            list_image_path.extend(paths)
            list_txt.extend(titles)

            torch.save(list_image_path, "list_image_path.pt")
            torch.save(list_txt, "list_txt.pt")

            print(f"Chunk {i // chunk_size + 1}/{len(rand_ints) // chunk_size}: {len(paths)} files downloaded and saved.")

    end = time.time()
    print(f"Total time taken to download {len(rand_ints)} files: {end - start:.2f} seconds")

if __name__ == '__main__':
    main()