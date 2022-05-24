import random
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd
from torch.utils.data import DataLoader, TensorDataset
from ont_fast5_api.fast5_interface import get_fast5_file
from Bio import SeqIO
import matplotlib.pyplot as plt

NUCLEOTIDE_TYPE_MASK = ['A', 'T', 'G', 'C']

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, rawdata):
        out, hidden_n = self.rnn(rawdata)
        out = self.fc(out[:, -1, :])
        return out

def tensor_from_string(sequence):
    elems = []
    for c in sequence:
        elem = [0, 0, 0, 0];
        elem[NUCLEOTIDE_TYPE_MASK.index(c)] = 1;
        elems.append(elem)
    return torch.tensor(elems, dtype=torch.float)

def data_from_raw(raw, len_samples, len_seq):
    samples_per_nucleotide = int(len_samples / len_seq)
    elems = []
    pos = 0
    for i in range(len_seq):
        elem = []
        elem.append(raw[min(int(pos), len_samples - 1)])
        sum = 0
        num = int(pos + samples_per_nucleotide) - int(pos)
        for i in range(int(pos), int(pos + samples_per_nucleotide)):
            sum += raw[i]
        elem.append(sum / num)
        pos += samples_per_nucleotide
        elem.append(raw[min(int(pos), len_samples - 1)])
        elems.append(elem)
    return torch.tensor(elems, dtype=torch.float)

def list_fastq(path):
    ans = []
    for file in os.listdir(path):
        if (file.endswith(".fastq")):
            ans.append(SeqIO.index(os.path.join(path, file), "fastq"))
    return ans


TRAIN_SHARE = 80


input_dim = 3
output_dim = 4

best_hidden_dim = 0
best_layer_dim = 0
best_accuracy = 0

path = "/hdd/data/Maxim/Bham/FAB41174-3976885577_Multi"

seqfiles = list_fastq(os.path.join(path, "fastq"))

length_accuracies = []

for hidden_dim in range(50, 151, 5):
    for layer_dim in range(1, 5):
        read_count = 0

        model = RNN(input_dim, hidden_dim, layer_dim, output_dim)

        error = nn.MSELoss()
        learning_rate = 0.05
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

        test_reads = []
        for file in os.listdir(path):
            if file.endswith(".fast5"):
                datareader = get_fast5_file(os.path.join(path, file), mode="r")
                for read in datareader.get_reads():
                    raw = read.get_raw_data()
                    name = read.read_id
                    seq = None
                    for dict in seqfiles:
                        if (dict.get(name) is not None):
                            seq = dict[name].seq
                    if (random.randint(1, 100) <= TRAIN_SHARE):
                        features = data_from_raw(raw, len(raw), len(seq))
                        targets = tensor_from_string(seq)
                        dataset = TensorDataset(features, targets)
                        train = torch.unsqueeze(Variable(features), dim=1)
                        labels = Variable(targets)
                        optimizer.zero_grad()
                        outputs = model(train)
                        loss = error(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    else:
                        test_reads.append(read)
                    read_count += 1
                    if (read_count == 1000):
                        break
            if read_count == 1000:
                break
        combined_length = 0
        combined_correct = 0
        for read in test_reads:
            name = read.read_id
            seq = None
            for dict in seqfiles:
                if (dict.get(name) is not None):
                    seq = dict[name].seq
            if seq is None:
                continue
            elem = [len(seq), 0]
            combined_length += len(seq)
            raw = read.get_raw_data()
            data = torch.unsqueeze(data_from_raw(raw, len(raw), len(seq)), dim=1)
            prediction = model(data)
            for i in range(len(seq)):
                for j in range(4):
                    if prediction[i][j] == torch.max(prediction[i]):
                        sym = NUCLEOTIDE_TYPE_MASK[j]
                if (sym == seq[i]):
                    combined_correct += 1
                    elem[1] += 1
            elem[1] /= elem[0]
            length_accuracies.append(elem)
        accuracy = combined_correct / combined_length
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hidden_dim = hidden_dim
            best_layer_dim = layer_dim
print("Accuracy achieved was:", best_accuracy)
print("Optimal number of layers was:", best_layer_dim)
print("Optimal size of an RNN hidden layer was:", best_hidden_dim)
plot1 = plt.plot(0, 1, data=length_accuracies)
plt.savefig("length_accuracies.png")
