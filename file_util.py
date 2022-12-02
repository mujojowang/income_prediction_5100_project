import csv

def read_csv(file_path):
    with open(file_path) as csvfile:
        csv_reader = list(csv.reader(csvfile, delimiter=','))
    return csv_reader

def load_file(file_path, split_tag="="):
  data = open(file_path, "r").readlines()
  x, y = zip(*[line.strip().split(split_tag) for line in data])
  return x, y

