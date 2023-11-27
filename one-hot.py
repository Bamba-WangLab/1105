import numpy as np
from Bio import SeqIO

def load_fasta_file(file_path, label):
    sequences = []
    labels = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
        labels.append(label)
    return sequences, labels

def one_hot_encode(sequences, amino_acids='ACDEFGHIKLMNPQRSTVWYX'):
    aa_to_int = dict((aa, i) for i, aa in enumerate(amino_acids))
    encoded_data = []
    for seq in sequences:
        encoded_seq = []
        for aa in seq:
            one_hot = [0] * len(amino_acids)
            if aa in aa_to_int:
                one_hot[aa_to_int[aa]] = 1
            encoded_seq.append(one_hot)
        encoded_data.append(encoded_seq)
    return np.array(encoded_data)

# Load sequences and labels
positive_sequences, positive_labels = load_fasta_file('D:/Python002/Deep/data/positive.fasta', 1)
negative_sequences, negative_labels = load_fasta_file('D:/Python002/Deep/data/negative.fasta', 0)

# Combine and shuffle data
all_sequences = positive_sequences + negative_sequences
all_labels = positive_labels + negative_labels
combined = list(zip(all_sequences, all_labels))
np.random.shuffle(combined)
all_sequences, all_labels = zip(*combined)

# One-hot encoding
encoded_sequences = one_hot_encode(all_sequences)

# Save encoded data and labels
np.save('D:/Python002/Deep/data/encoded_sequences.npy', encoded_sequences)
np.save('D:/Python002/Deep/data/labels.npy', all_labels)