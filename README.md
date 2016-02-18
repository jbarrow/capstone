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

The preprocessing is not yet working standalone, but it is present in the iPython notebook, called "loader.iPynb", if you are interested in looking at that.

## TODO:

[] Standalone preprocessing
[] Aligning notes with training data
[] TensorFlow RNN
[] TIMIT data and preprocessing
