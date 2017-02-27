import numpy as np
from ast import literal_eval as make_tuple

def main():
    filename = 'dataset.csv'
    data = np.genfromtxt(filename, dtype='float', delimiter=',', skip_header=1)
    with open(filename, 'r') as file:
        line = file.readline()
    data = data.reshape(make_tuple(line[2:]))
    print(data.shape)

if __name__ == '__main__':
    main()
