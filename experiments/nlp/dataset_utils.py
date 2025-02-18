def load_dataset(base_path, size=None):
    # Load the text list (x)
    text_list = []
    with open(base_path + ".text", "r") as file:
        for line in file:
            text_list.append(line.strip())

    # Load the label list (y)
    label_list = []
    with open(base_path + ".labels", "r") as file:
        for line in file:
            label_list.append(int(line.strip()))
    
    # Crop the dataset to the given size if provided
    if size:
        text_list = text_list[:size]
        label_list = label_list[:size]

    # Check the number of samples
    if len(text_list) == len(label_list):
        print("# of samples in the dataset (\"{:s}\"): {:d}".format(base_path, len(text_list)))
    else:
        raise RuntimeError("# of samples for texts and labels in dataset (\"{:s}\")".format(base_path))

    return { "text_list": text_list, "label_list": label_list }
