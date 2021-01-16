# gen n to m items of list l

data = list(range(100))

n_index = 3
prev_start_index = 0
prev_stop_index = n_index
max_seq = len(data) / n_index

for i in range(len(data)):
    if i > max_seq:
        break

    seq_data = data[prev_start_index:prev_stop_index]
    print(seq_data, len(seq_data))

    prev_start_index += n_index
    prev_stop_index += n_index
