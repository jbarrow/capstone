# Capstone Project

## Downloading the Data

To get started running this project, run:

```
  python downloader.py [--all|--new|--help|-f [FILENAME]]
```

Using the `--new` flag will force the download of a new section of the MAPS database not already present on your system, but only one. Use the `--all` flag to download *all* sections of the MAPS database not already on your system.

Additionally, if you already know which section of the database you would like do download, you can specify that with the `-f` flag, as in:

```
  python downloader.py -f MAPS_SptkBGCl_1.zip
```

To download and organize the MAPS database of Piano music. Note that you should not use the `--all` flag unless you have:
1. A lot of space (60-80GB of free space)
2. A lot of time (to download and unpack 60-80GB)
3. A really good internet connection

Running it without --all or -f, or with --new if you already have a section downloaded, will choose a random subset of the code to download. 

## Preprocessing

The data is preprocessed recursively from a top-level directory. To get an idea of the preprocessing steps we take, check out the "Preprocessing_Notebook.iPynb" in the repository. To simply use the preprocessor, use the python command:

```
  python preprocess.py [TOP_LEVEL_DIRECTORY]
  ```

Give it the name of a top-level directory (e.g. data/MUS) and let it work (for a while, preprocessing each piece of music takes some time). The resulting TrainingData is saved in a `.pkl` file in its original location, with its original name.

The steps we take for preprocessing are:

- Window the data into 50ms windows, using a Hanning Window and an overlap of 25ms
- Pad the audio data to increase our spectral resolution (pad size controlled in code)
- Take the STFT of the data
- Run a semitone filterbank over the data, saved in the `X` variable of the class
- Compute a target output matrix, saved in the `Y` variable of the class

## TODO:

- [x] Standalone preprocessing
- [x] Aligning notes with training data
- [ ] Theano RNN
- [ ] TIMIT data and preprocessing
- [x] Download specific files from MAPS, for consistency
