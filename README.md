Animal Vocalization Generative Network
==============================

Animal Vocalization Generative Network (AVGN) is a repository of python tools centered around latent models used to generate, visualize, and characterize animal vocalizations. The code for this repository supports the paper Tim Sainburg et al., (reference below). 

Currently, there are two seperate repositories for AVGN: one at [github.com/timsainb/avgn](github.com/timsainb/avgn) that hosts an earlier and less feature rich, but cleaner version of this code. The second is [this repository](github.com/timsainb/avgn_paper). There are more species and examples in this repo, but its not as clean and might be a bit harder to figure out. 

# I want to use these tools on my own dataset. 
Great! The purpose of this repository is for you to be able to apply these tools to your own dataset. To use these tools, you are going to need to be somewhat proficient in python and jupyter notebooks. If you haven't used python or jupyter before, these notebooks can be an opportunity to learn! But first take some time to learn the basics. 

<hr \>

## Getting started
You have a dataset of animal vocalizations and you want to use AVGN to analyse them. Currently, your data is in some format that *probably* isn't the exact same as the one used in AVGN. This is a very breif guide to how to use this repository, so feel free to send me an email if you're running into trouble. The broad steps are: 


1. Clone and install AVGN
2. Convert your data into the correct format. 
3. Run analyses on your correctly formatted data

### 1. Clone and install AVGN

1. Navigate to the folder in your local environment where you want to install the repository. Then type `git clone https://github.com/timsainb/avgn_paper.git`
2. Install the package by typing `python setup.py develop`

### 2. Getting your data into the right format
In building AVGN, we found datasets prepared in several different formats. To use AVGN, you'll need to translate your dataset from whatever format you currently have it in, to our format. Luckily (1) you have [several different examples]() to work off of, in trying to figure out how to translate your dataset into our format, and (2) the format we use is pretty universal and pretty easy. 

There are three files you want to generate for your dataset: 
1. `.WAV` files of each of your vocalization files 
2. `.JSON` files with WAV general information, as well as unit information

An example JSON file with metadata:

```
{
    "length_s": 15,
    "latitude": "32.83",
    "longitude": "-117.27",
    "samplerate_hz": 48000,
    "wav_location": "/location/of/my/dataset/myfile.wav",
    "noise_file": "/location/of/my/dataset/myfile.wav",
    "weight": 30,
    "indvs": {
        "B1534": {
            "species": "European starling",
            "group": "group1",
            "partner": "B1534",
            "age": "15 days",
            "units": {
                "syllables": {
                    "start_times": [1.5, 2.5, 6],
                    "end_times": [2.3, 4.5, 8],
                    "labels": ["a", "b", "c"],
                    "Hz_min": [600, 100, 200],
                    "Hz_max": [5000, 4000, 6000],
                    "contexts": ["singing", "fighting", "fleeing"],
                },
                "notes": {
                    "start_times": [1, 1.5, 6.4],
                    "end_times": [21.1, 1.8, 7.0],
                    "labels": ["1", "4", "2"],
                },
            },
        }
    },
}
```

If the data is not segmented, or does not have much metadata, you'll just want to fill in the information you have. E.g.:

```
{
    "length_s": 15,
    "samplerate_hz": 48000,
    "wav_location": "/location/of/my/dataset/myfile.wav",
    "indvs": {
        "B1534": {
            "species": "European starling",
        }
    },
}
```

To get data into this format, you're generally going to have two write a custom parser to convert your data from your format into AVGN format. There are numberous examples in `avgn/custom_parsing/`. 

### 3. Running analyses on your correctly formatted data
Now that your data is in the right format, this shouldn't be too difficult. Just look at the other notebooks and apply those methods to your dataset!

<hr \>




Project Organization
------------

    ├── LICENSE
    ├── README.md                              <- The top-level README for developers using this project.
    ├── data                                   <- Vocalization data should be stored here 
    │
    ├── docs                                   <- General documentation on AVGN
    │     
    ├── notebooks                              <- Jupyter notebooks. 
    │   ├── 00.0-download-datasets             <- Downloads some of the datasets used in the paper
    │   ├── 00.1-data-exploration              <- Loose scratchpads to explore some datasets.
    │   ├── 01.0-custom-parsing                <- Parse datasets from original format into AVGN jsons
    │   ├── 01.1-syllabic-segmentation         <- Segment unsegmented datasets using dynamic thresholding
    │   ├── 01.1-visualize-dataset-elements    <- In a few datasets this is used to visualize at syllables
    │   ├── 02.0-make-syllable_df              <- Get dataset of syllable spectrograms from WAVs
    │   ├── 02.1-project-UMAP                  <- Project syllable spectrograms into umap (discrete)
    │   ├── 02.5-make-projection-dfs           <- Make dataset of umap projections of discrete syllables
    │   ├── 03.0-compare-clusters-vs-labels    <- Compare clusters vs labels in bengalese finch
    │   ├── 04.0-clusters-vs-labels-metrics    <- Compare clusters vs labels with metrics (finch + cassins)
    │   ├── 05.0-viz-transitions-continuous    <- continuous latent trajectories
    │   ├── 05.1-viz-transitions-discrete      <- discrete latent projections + clustering + markov modelling
    │   ├── 06.0-neural-networks               <- starling and canary projections + sampling
    │   ├── 07.0-segmentation-examples         <- Examples of dynamic thresholding segmentation
    │   ├── 08.0-human-phones-analyses         <- Discrete projections of phones
    │   ├── 09.0-clusterability                <- Clusterability measures
    │   ├── 10.0-dataset-statistics            <- General statistics of the datasets I used in the paper
    │   └── 11.0-umap-projection-fig           <- Fig 1 in the paper. 
    │
    ├── requirements.txt                       <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py                               <- makes project pip installable (pip install -e .) so src can be imported
    ├── avgn                                   <- Source code for use in this project.



## Notes
- There are only notebooks to download datasets for a subset of the datasets that are freely available online. This is because I didn't think to write these as notebooks until later in the analysis pipeline. If you want to write a notebook to download any other datasets, I would be glad to pull it in. 
- Many of the datasets I took a look at didn't make it into the final analysis. For most of these datasets, the reason was that it was too time-consuming to clean up the data before it could be used for my purposes (e.g. segmenting vocalizations out from long noisy recordings). I left most of the exploration notebooks in this repository so that you could take a look at them yourself. 
- I didn't add spectrogram inversion to these notebooks yet. In the past, I've just used griffin and lim inversion. I have an [implementation of griffin + lim in the other avgn repository](https://github.com/timsainb/avgn). [I also have a notebook version in another repository with a google colab example](https://github.com/timsainb/tensorflow2-generative-models). It should be pretty straightforward to get working, but if you're having trouble just ask. It might also be worth looking into [other spectrogram inversions techniques](https://anclab.org/software/phaserecon/). If you have a good example inversion notebook you would like to pull into this repo, please do so!

--------
## Citation

If you use this repository in your research, please cite our paper:

(paper not yet online, contact me)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
