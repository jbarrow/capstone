from pandas import DataFrame

def parse_file(filename):
    df = DataFrame.from_csv(filename, sep='\t')
