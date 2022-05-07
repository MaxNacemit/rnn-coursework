import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.autograd
from torch.utils.data import DataLoader, TensorDataset
from ont_fast5_api.fast5_interface import get_fast5_file

datareader = get_fast5_file("D:\Developer\pycode\Coursework\data.fast5", mode="r")

NUCLEOTIDE_TYPE_MASK = ['A', 'T', 'G', 'C']

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first = True) # playing with the type of activation function is possible
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, rawdata):
        hidden0 = Variable(torch.zeros(self.layer_dim, rawdata.size()[0], self.hidden_dim))
        out, hidden_n = self.rnn(rawdata, hidden0)
        out = self.fc(out[:, -1, :])
        return out

def tensor_from_string(sequence):
    elems = []
    for c in sequence:
        elem = [0, 0, 0, 0];
        elem[NUCLEOTIDE_TYPE_MASK.index(c)] = 1;
        elems.append(elem)
    return torch.tensor(elems)

def data_from_raw(raw, len_samples, len_seq):
    samples_per_nucleotide = int(len_samples / len_seq)
    elems = []
    pos = 0
    for i in range(len_seq):
        elem = []
        elem.append(raw[int(pos)])
        sum = 0
        num = int(pos + samples_per_nucleotide) - int(pos)
        for i in range(int(pos), int(pos + samples_per_nucleotide)):
            sum += raw[i]
        elem.append(sum / num)
        pos += samples_per_nucleotide
        elem.append(raw[int(pos)])
        elems.append(elem)
    return torch.tensor(elems)


TRAIN_SHARE = 80


input_dim = 3
output_dim = 4

best_hidden_dim = 0
best_layer_dim = 0
best_accuracy = 0

for hidden_dim in range(50, 151, 5):
    for layer_dim in range(1, 15):
        read_count = 0

        model = RNN(input_dim, hidden_dim, layer_dim, output_dim)

        error = nn.CrossEntropyLoss()
        learning_rate = 0.05
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

        test_reads = []
        for read in datareader.get_reads():
            if (read_count == 10000):
                break
            raw = read.get_raw_data()
            fastq = read.get_analysis_dataset('Basecall_1D_000', "BaseCalled_template/Fastq")
            seq = fastq[(fastq.find("\n") + 1):(fastq.find('+') - 1)]
            if (random.randint(1, 100) <= TRAIN_SHARE):
                features = data_from_raw(raw, len(raw), len(seq))
                targets = tensor_from_string(seq)
                dataset = TensorDataset(features, targets)
                loader = DataLoader(dataset, 100, shuffle=False)
                for (raws, probabilities) in loader:
                    train = Variable(raws)
                    labels = Variable(probabilities)
                    optimizer.zero_grad()
                    outputs = model(train)
                    loss = error(outputs, labels)
                    loss.backward()
                    optimizer.step()
            else:
                test_reads.append(read)
            read_count += 1
        combined_length = 0
        combined_correct = 0
        for read in test_reads:
            fastq = read.get_analysis_dataset('Basecall_1D_000', "BaseCalled_template/Fastq")
            seq = fastq[(fastq.find("\n") + 1):fastq.find('+')]
            combined_length += len(seq)
            raw = read.get_raw_data()
            data = data_from_raw(raw, len(raw), len(seq))
            for i in range(len(seq)):
                prediction = model(data[i])
                sym = NUCLEOTIDE_TYPE_MASK[prediction.index(max(prediction))]
                if (sym == seq[i]):
                    combined_correct += 1
        accuracy = combined_correct / combined_length
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_hidden_dim = hidden_dim
            best_layer_dim = layer_dim
print("Accuracy achieved was:", best_accuracy)
print("Optimal number of layers was:", best_layer_dim)
print("Optimal size of an RNN hidden layer was:", best_hidden_dim)

