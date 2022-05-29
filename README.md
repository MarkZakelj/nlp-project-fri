# Automatic semantic relation extraction
<sup>This is the main repository for the NLP project on FRI of the group "Vesoljci". The repository was created for a project that was part of Natural Language Processing course at the University of Ljubljana, Faculty for computer and information science.</sub>

To use this repository as intended you should have a NVIDIA GPU with appropriate NVIDIA driver and CUDA versions that are compatible with PyTorch and Tensorflow.

## Repository folder structure
- `assets`: files used in README.md
- `config_files`: configuration parameters for each of the experiments
- `data`: folder with the `Termframe` dataset, all the pre-processed data is saved here after running the
          scripts as well as new `experiments` folder with different datasets and models with . 

    .
    ├── assets
    ├── config_files
    ├── data                        # root folder of experiments and multiple `.csv` files with preproessed 
    |   |                             and split datasets after running preprocessing scripts
    │   ├── experiments             # folder with different dataset setups and models
    |   |   ├── experiment_1        # folder with a particular dataset setup split into train and test data
    |   │   │   ├── model_1         # folders with a particular model configuration, saved model, 
    |   |   |   ├── model_2           and results from that model, as well as generated graphs
    |   |   |   └── ...
    │   │   ├── experiment_2
    |   |   └── ... 
    │   └── Termframe               # Termframe dataset with its original folder structure
    └── ...

## Environment creation and activation

```
# create new environment and install all dependencies
conda env create -f environment.yml

# activate environment
conda activate nlp

# close environment (after you are done using it)
conda deactivate
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
```

## Generate graphs
```
# install nltk packages (used only once, since the packages are saved locally)
pyhton install_nltk_packages.py

# generate graphs
python generate_graphs.py
```

**Example of a complete graph of relations**
![Image of a generated graph](/assets/graph.png)

**Example of a graph with largest connected componens**
![Image of a generated graph](/assets/max_wcc_graph.png)