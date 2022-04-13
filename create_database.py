import torch
from bs4 import BeautifulSoup
import os
# assign directory

class Data:
    def __init__(self, directory):
        self.data = []
        self.directory = directory
        self.dataset = []
        self.targets = []
        self.number_of_data = 50000

    def get_array_of_data(self):
        print("reading xmls")
        max_time = 1000.0
        i = 0
        for filename in os.listdir(self.directory):
            # print(filename)
            f = os.path.join(self.directory, filename)
            f = self.directory + '/' + filename
            # //print(f)
            # checking if it is a file

            i += 1
            if i > self.number_of_data:
                break

            if os.path.isfile(f):
                with open(f, 'r') as g:
                    data = g.read()

                    Bs_data = BeautifulSoup(data, "xml")

                    # Finding all instances of tag
                    # `unique`
                    b_unique = Bs_data.find_all('head')
                    res = 1
                    if not b_unique:
                        b_time = 100
                        if "PASS" in filename:
                            res = 0
                        elif "FAIL" not in filename:
                            continue
                    else:
                        # Using find() to extract attributes
                        # of the first instance of the tag
                        b_name = Bs_data.find('result').get('value')
                        b_time = Bs_data.find('test-total-time').get('value')
                        if b_name == "PASS":
                            res = 0

                    self.data.append(res + (float(b_time) / max_time))
                    # self.data.append(res)


    def create_dataset(self):
        size_of_sequence = 5
        self.get_array_of_data()
        # print(len(self.data))
        for i in range(len(self.data) - size_of_sequence):
            self.dataset.append(torch.Tensor(self.data[i:i+size_of_sequence]))
            self.targets.append(torch.Tensor(self.data[i+size_of_sequence:i+size_of_sequence+1]))
        self.number_of_data = len(self.data) - size_of_sequence - 1


    def write_data_to_file(self):
        with open('data.txt', 'w') as f:
            for line in self.data:
                # print(line)
                f.write(str(line))
                f.write('\n')


    def read_dataset_from_file(self):
        size_of_sequence = 5
        with open('data.txt', 'r') as f:
            lines = f.readlines()
            lines_float = []
            for i in lines:
                lines_float.append(float(i))
            for i in range(len(lines) - size_of_sequence):
                self.dataset.append(torch.Tensor(lines_float[i:i+size_of_sequence]))
                self.targets.append(torch.Tensor(lines_float[i+size_of_sequence:i+size_of_sequence+1]))
            self.number_of_data = len(self.dataset) - 5 - size_of_sequence

