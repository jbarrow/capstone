# Capstone Project

## Downloading the Data

To get started running this project, run:

```
  python downloader.py [--all|--help]
```

To download and organize the MAPS database of Piano music. Note that you should not use the `--all` flag unless you have:
1. A lot of space (60-80GB of free space)
2. A lot of time (to download and unpack 60-80GB)
3. A really good internet connection

Running it without --all will choose a random subset of the code to download. 

## Preprocessing

The preprocessing is not yet working standalone over the whole dataset. To get a good idea of how it works, look in the iPython notebook, called "loader.iPynb". Currently, we can preprocess a single song at a time, but have yet to correlate the outputs with the inputs (and thus have all the elements needed to train our RNN).

## TODO:

- [ ] Standalone preprocessing
- [x] Aligning notes with training data
- [ ] TensorFlow RNN
- [ ] TIMIT data and preprocessing
