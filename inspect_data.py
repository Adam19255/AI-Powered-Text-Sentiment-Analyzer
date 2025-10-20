import os

def load_imdb_dataset(base_path="data/aclImdb_v1/aclImdb"):
    # This is an inner helper function that loads either "train" or "test"
    # depending on what folder we pass.
    def load_split(split):
        texts = []
        labels = [] # 0 = negative, 1 = positive
        '''
        We loop twice:
            once for the pos subfolder and assign label 1
            once for the neg subfolder and assign label 0
        '''
        for label_name, label_value in [("pos", 1), ("neg", 0)]:
            folder = os.path.join(base_path, split, label_name)
            for filename in os.listdir(folder):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder, filename)
                    # read its contents (the review text)
                    with open(file_path, "r", encoding="utf-8") as f:
                        # store the review
                        texts.append(f.read().strip())
                    # store the numeric label (1 or 0)
                    labels.append(label_value)
        # After reading all files for that split (train or test), we return
        # two lists.
        return texts, labels

    train_texts, train_labels = load_split("train")
    test_texts, test_labels = load_split("test")

    return (train_texts, train_labels), (test_texts, test_labels)

if __name__ == "__main__":
    (train_texts, train_labels), (test_texts, test_labels) = load_imdb_dataset()
    print("Loaded")
    print("Train size:", len(train_texts))
    print("Test size:", len(test_texts))
    # Print a preview of the first review and label
    print(train_texts[0])
    print("Label:", train_labels[0])
