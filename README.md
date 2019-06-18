Animal Vocalization Generative Network
==============================

### Introduction


<hr \>

## Getting started
You have a dataset of animal vocalizations and you want to use AVGN to analyse them. Currently, your data is in some format that *probably* isn't the exact same as the one used in AVGN. To use this package,


1. Clone and install AVGN
2. Create a notebook folder for your dataset 
3. Convert your data into the correct format. 
4. Run analyses on your correctly formatted data

### 1. Clone and install AVGN

### 2. Create a notebook forder for your dataset

### 3. Getting your data into the right format
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

### 4. Running analyses on your correctly formatted data


<hr \>




Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- General documentation on AVGN
    │    
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── avgn               <- Source code for use in this project.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
