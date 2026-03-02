import torch  # deep learning library for tensors

# --- Load text data ---
with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# --- Build character vocabulary ---
chars = sorted(set(text))  # all unique characters, sorted alphabetically
vocab_size = len(chars)    # number of unique characters

# --- Encoder / Decoder ---
string_to_int = { ch: i for i, ch in enumerate(chars) }  # char → index
int_to_string = { i: ch for i, ch in enumerate(chars) }  # index → char

def encode(s):
    # converts a string into a list of integers
    return [string_to_int[char] for char in s]

def decode(indices):
    # converts a list of integers back into a string
    return ''.join([int_to_string[idx] for idx in indices])

# --- Encode entire dataset ---
data = torch.tensor(encode(text), dtype=torch.long)

# --- Train / Validation split ---
n = int(0.8 * len(data))  # 80% train, 20% val
train_data = data[:n]
val_data = data[n:]

# --- Block size and sample ---
block_size = 8  # context window size

x = train_data[:block_size]    # input sequence
y = train_data[1:block_size+1] # target sequence (shifted by 1)
