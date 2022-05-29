# Automatic semantic relation extraction
<sup>This is the main repository for the NLP project on FRI of the group "Vesoljci". The repository was created for a project that was part of Natural Language Processing course at the University of Ljubljana, Faculty for computer and information science.</sub>

To use this repository as intended you should have a NVIDIA GPU with appropriate NVIDIA driver and CUDA versions that are compatible with PyTorch and Tensorflow.

## Repository folder structure
    .
    ├── assets                      
    ├── config_files                
    ├── data                        
    │   ├── experiments             
    |   |   ├── experiment_1            # folder with dataset setup split into train and test data
    |   │   │   ├── model_1         
    |   |   |   ├── model_2
    |   |   |   ├── test.csv
    |   |   |   └── train.csv
    │   │   ├── experiment_2               
    |   |   |   ├── model_1             # model configuration, saved model, results, generated graphs
    |   |   |   |   ├── annotaion.csv
    |   |   |   |   ├── config_dict.json
    |   |   |   |   ├── model.pt
    |   |   |   |   ├── results.txt              
    |   |   |   |   └── graph.png
    |   |   |   └── ...
    |   |   └── ... 
    │   └── Termframe                   # Termframe dataset with its original folder structure
    └── report                          

- `assets`: files used in README.md
- `config_files`: configuration parameters for each of the experiments
- `data`: folder with the `Termframe` dataset, all the pre-processed data is saved here after running the
          scripts as well as new `experiments` folder with different datasets setups and models that is created automatically by running the scripts.
- `report`: folder with the scientific paper

## Environment creation and activation

```
# create new environment and install all dependencies
conda env create -f environment.yml

# activate environment
conda activate nlp
```

## Preprocessing data
```
# convert data from .tsv to .csv format
python convert_data.py

# prepare and split data for training and testing
# for sequence tagging
python prepare_data.py

#for relation extracion
python prepare_data_regions.py
```

## Train models
```
# train sequence tagger
python train_sequence_tagging.py

# train relation extractor
python train_relation_extraction.py

# run experimental relation extraction (after training relation extractor)
python relation_extraction_growing_window.py
```

## Generate graphs
```
# install nltk packages (used only once, since the packages are saved locally)
python install_nltk_packages.py

# generate graphs
python generate_graphs.py
```

**Example of a complete graph of relations**
![Image of a generated graph](/assets/graph_slo.png)

**Example of a graph with largest connected componens**
![Image of a generated graph](/assets/max_wcc_graph_en.png)