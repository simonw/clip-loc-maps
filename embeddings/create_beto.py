# Creating beto from jsons in chunks
import torch
import glob
import json

betos = {}
sum = 0 
length = len(glob.glob("jsons/*.json"))
BATCH_SIZE = 20000

for index in range((length // BATCH_SIZE) + 1):
    beto = torch.empty(0, 512)
    print("beto made")
    beto_idx = []
    print("beto idx made")

    files = glob.glob("jsons/*.json")[BATCH_SIZE*index:BATCH_SIZE*(index+1)]
    print(f"Processing {len(files)} files...")  # Check how many files are being processed

    for json_file in files:
        sum += 1
        with open(json_file) as file:
            data = json.load(file)
        try: 
            json_embed = data["embedding"]
            json_embed = torch.tensor(json_embed).unsqueeze(0)
            beto = torch.cat((beto, json_embed), 0)
            beto_idx.append(data["@id"]+"/full/2000,/0/default.jpg")
            if sum % 5000 == 0:
                print(sum)
        except Exception as e: 
            continue

    betos[index] = (beto, beto_idx)

beto = torch.empty(0, 512)
beto_idx = []
for i in range((length // BATCH_SIZE) + 1):
    beto = torch.cat((beto, betos[i][0]), 0)
    beto_idx += betos[i][1]
beto_normalized = beto / beto.norm(dim=-1, keepdim=True)

torch.save(beto, "search/beto.pt")
torch.save(beto_idx, "search/beto_idx.pt")
torch.save(beto_normalized, "search/beto_normalized.pt")   
