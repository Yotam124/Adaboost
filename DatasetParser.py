import numpy as np;


def parse(file_name):
    input_file = open(file_name, "r")
    try:
        dataset = []
        counter = 0
        line = input_file.readline()
        while line:
            counter += 1
            line = line.replace(",", " ")
            if not line.__contains__("Iris-setosa"):
                split_line = line.split()
                if len(split_line) > 3:
                    # Remove the irrelevant columns in Iris dataset
                    split_line.__delitem__(0)
                    split_line.__delitem__(2)
                    # Set labels for Iris dataset
                    if split_line[2] == "Iris-versicolor":
                        split_line[2] = 1
                    else:
                        split_line[2] = -1
                else:
                    # Set labels for HC dataset
                    if len(split_line) > 1:
                        if split_line[1] == "2":
                            split_line[1] = -1
                x = np.array(split_line)
                dataset.append(x.astype(np.float))

            line = input_file.readline()
    finally:
        input_file.close()

    return dataset
