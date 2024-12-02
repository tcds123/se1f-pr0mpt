import csv
from datasets import load_dataset

def load_csv(path):
    csv_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            csv_list.append(row)
    return csv_list

def load_my_dataset(my_path):
    dataset_path = my_path
    data_files = {
        "train": f"{dataset_path}/train.tsv",
        "validation": f"{dataset_path}/dev.tsv",
        "test": f"{dataset_path}/test.tsv"
    }
    column_names = ["label", "sentence"]

    dataset = load_dataset('csv', data_files=data_files, delimiter='\t', column_names=column_names, header=None)

    print(dataset['train'][:5])



if __name__ == "__main__":
    path = './data/SST-2/test.tsv'
    # load_csv(path)
    my_path = 'D:/Project/self-prompt/eval/data/SST-2'
    load_my_dataset(my_path)
