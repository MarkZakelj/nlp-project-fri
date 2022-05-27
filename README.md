# Automatic semantic relation extraction
<sup>This is the main repository for the NLP project on FRI of the group "Vesoljci". The repository was created for a project that was part of Natural Language Processing course at the University of Ljubljana, Faculty for computer and information science.</sub>

To use this repository as intended you should have a NVIDIA GPU with appropriate NVIDIA driver and CUDA versions that are compatible with PyTorch and Tensorflow.

## Environment creation and activation

```
# create new environment and install all dependencies
conda env create -f environment.yml

# activate environment
conda activate nlp

# close environment (after you are done using it)
conda deactivate
```

## Convert and prepare data
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
# install nltk packages (**used only once**, since the packages are saved locally)
pyhton install_nltk_packages.py

# generate graphs
python generate_graphs.py
```

**Example of a generated graph**
![Image of a generated graph](/assets/graph.png)