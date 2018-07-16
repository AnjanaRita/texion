from sklearn.model_selection import train_test_split


def train_test_split_data(texts, labels, test_size=0.2, **kwargs):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, **kwargs)
    return train_texts, test_texts, train_labels, test_labels
