import os
import tiktoken
import numpy as np
import pickle

# list of BabyLM directories
directories = ['bnc_spoken', 'childes', 'gutenberg', 'open_subtitles', 'simple_wiki', 'switchboard']

print("Encoding data with tiktoken gpt2 bpe")

print("Directories:")

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = []
val_ids = []
for directory in directories:
    input_file_path = os.path.join(directory, 'input.txt')
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    train_data_ids = enc.encode_ordinary(train_data)
    val_data_ids = enc.encode_ordinary(val_data)

    train_ids.extend(train_data_ids)
    val_ids.extend(val_data_ids)

    print(f"{directory} has {len(train_data_ids)+len(val_data_ids):,} tokens")

print("Total:")

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save enc to pkl file
with open(os.path.join(os.path.dirname(__file__), 'enc.pkl'), 'wb') as f:
    pickle.dump(enc, f)


