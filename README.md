# clip-loc-maps
This is the repository for the paper "Integrating Visual and Textual Inputs for Searching Large-Scale Map Collections with CLIP," accepted to the 2024 Computational Humanities Research (CHR) conference.

# Introduction
This paper explores the use of multimodal machine learning to facilitate search and discovery in large-scale map collections. We implement three search strategies using CLIP (Contrastive Language-Image Pre-training) embeddings on 562,842 images of maps from the Library of Congress:

- Text-input search
- Image-input search (reverse image search)
- Combined text and image input search

Our key contributions include:

- CLIP embeddings generated for 562,842 map images
- Search implementation allowing natural language, visual, and multimodal inputs
- Dataset of 10,504 map-caption pairs for fine-tuning CLIP
- Code released as Jupyter notebooks in the public domain

The paper demonstrates the potential for searching maps beyond catalog records and existing metadata. We consulted with Library of Congress staff to explore example searches and evaluate utility and followed the LC Labs AI Planning Framework to ensure responsible and ethical AI practices.
While initial fine-tuning experiments yielded mixed results, we believe further work could reduce noise in searches. 
This research addresses the challenge of improving discoverability in rapidly growing digital collections, with implications for galleries, libraries, archives and museums worldwide.

# Using the repository

Helper files (those too big for GitHub) can be found at the project's public [Zenodo](https://zenodo.org/records/11538437?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjFmOTY0ZTkxLTI4MTMtNDcwZS1iZDlkLTE3MzI0N2UwZjBhOSIsImRhdGEiOnt9LCJyYW5kb20iOiJmY2I1ZDhiMTdiZjdhZGQ4NGExZmYwYTU0ZWQ5NWEwYyJ9.0UTJ1hiE82QAINiushqIYy5YVmT5Af40XCVJxEc63Eppapa5SK1L_kuGkYx4f_OBQoZ5MHdY2Z27QDyCPXYrbQ). We recommend placing `beto`, `beto_idx`, and `beto_normalized` into the `search` folder for compatibility.

The `embeddings` folder contains scripts that accept resource URLs (in the form specified in `p1_map_file_list.csv` and return CLIP-generated embeddings. `embed.stripped` has functionality to load a model checkpoint for fine-tuning experiments. `create_beto` accepts the JSON files generated by `embed_*` and creates `beto`, `beto_idx`, and `beto_normalized.`

The `fine-tuning` folder contains script for dataset creation, a notebook for fine-tuning incrementally, and the fine-tuning script. Fine-tuning accepts a range of image-text pairs with user-specified model hyperparameters.

Lastly, `search` loads in `beto` and `beto_idx` to accept user-specified search inputs. The **first two** cells must be run for the search cells to work. The third cell imports a fine-tuned model which is not included in this repository.

# Using the search notebook

To use the search notebook out of the box, 
1. Clone this repository,
2. Download `beto` and `beto_idx` from the [Zenodo](https://zenodo.org/records/11538437?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjFmOTY0ZTkxLTI4MTMtNDcwZS1iZDlkLTE3MzI0N2UwZjBhOSIsImRhdGEiOnt9LCJyYW5kb20iOiJmY2I1ZDhiMTdiZjdhZGQ4NGExZmYwYTU0ZWQ5NWEwYyJ9.0UTJ1hiE82QAINiushqIYy5YVmT5Af40XCVJxEc63Eppapa5SK1L_kuGkYx4f_OBQoZ5MHdY2Z27QDyCPXYrbQ),
3. Move `beto` and `beto_idx` to the `search` folder,
4. Run the **first two** cells in `search` (ensure all imports are resolved),
5. Run the cell under the corresponding search type.
