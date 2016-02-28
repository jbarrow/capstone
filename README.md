# Capstone Project

Welcome to the senior design project of Joe Barrow and Harang Ju, 4th year CS majors at the University of Virginia. Our research is focused on the similarities between music and pronunciation training. This repository, written for Python 2.7, contains all our code, results, and our technical paper.

## Getting started

### Jupyter Notebook Setup

System Requirements:

- Python, pip
- (Optional) [virtualenv](https://virtualenv.pypa.io/en/latest/)

To start the [Jupyter Notebook](https://jupyter.org/index.html):

```bash
# Clone the repo
git clone https://github.com/jbarrow/capstone.git
cd capstone

# Create a new virtual environment (optional, but recommended)
virtualenv venv
source venv/bin/activate # Always run this before starting the notebook

# Install requirements
pip install -r requirements.txt
# Start the notebook server
jupyter notebook
```

### Downloading the Data

The very first step is to [contact the nice people in charge of the MAPS database](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) in order to get access to the data. Once they have sent you a username and password for their FTP server, you can move on. 

To get started running this project, run:

```
  python download.py -u <username> -p <password> [--all|--new|-h|-f [FILENAME]]
```

Using the `--new` flag will force the download of a new section of the MAPS database not already present on your system, but only one. Use the `--all` flag to download *all* sections of the MAPS database not already on your system.

Additionally, if you already know which section of the database you would like do download, you can specify that with the `-f` flag, as in:

```
  python download.py -f MAPS_SptkBGCl_1.zip
```

To download and organize the MAPS database of Piano music. Note that you should not use the `--all` flag unless you have:
1. A lot of space (60-80GB of free space)
2. A lot of time (to download and unpack 60-80GB)
3. A really good internet connection

Running it without --all or -f, or with --new if you already have a section downloaded, will choose a random subset of the code to download. 

## Preprocessing

The data is preprocessed recursively from a top-level directory. To use the preprocessor, run the python command:

```
  python preprocess.py [TOP_LEVEL_DIRECTORY|-h]
```

Give it the name of a top-level directory (e.g. data/MUS) and let it work (for a while, preprocessing each piece of music takes some time). The resulting TrainingData is saved in a `.pkl` file in its original location, with its original name.

To get an idea of the preprocessing steps we take, check out the `notebooks/Preprocessing_Notebook.iPynb` in the repository. *Note, however,* that the notebook *does not* have a consistent API with the actual code, it simply provides an idea of the steps we take in preprocessing. The steps we take for preprocessing are:

- Window the data into 50ms windows, using a Hanning Window and an overlap of 25ms
- Pad the audio data to increase our spectral resolution (pad size controlled in code)
- Take the STFT of the data
- Run a semitone filterbank over the data, saved in the `X` variable of the class
- Compute a target output matrix, saved in the `Y` variable of the class

## Training

For training, we are using Theano to construct an LSTM network. You can view the steps we take in the `notebooks/Training_Notebook.iPynb` file.

```
  python train.py [-h]
```

## TODO:

- [x] Loading training data
- [ ] Theano RNN
- [ ] Track progress with paper

### MAPS
- [x] Download specific files from MAPS, for consistency
- [x] Standalone preprocessing
- [x] Aligning notes with training data

### TIMIT
- [ ] Get access to TIMIT
- [ ] TIMIT Downloader
- [ ] TIMIT Preprocessing
