import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def calculate_scores(test_data, predicted, classes_dict, cls_num):
    classes_dict_inv = {v: k for k, v in classes_dict.items()}
    Y_test = np.zeros((len(test_data), cls_num))
    y_score = np.zeros((len(test_data), cls_num))

    # fill Y_test
    for i, row in enumerate(test_data):
        ys = row[1]
        for y in ys:
            Y_test[i][classes_dict_inv[y]] = 1

    # fill y_score
    for i, ys in enumerate(predicted):
        for y in ys:
            y_score[i][classes_dict_inv[y]] = 1

    res = precision_recall_fscore_support(Y_test, y_score)

    classes = []
    for i in range(cls_num):
        classes.append(str(i))

    df = pd.DataFrame.from_dict({'class': classes,
                                 'precision': res[0],
                                 'recall': res[1],
                                 'f1': res[2],
                                 'support': res[3],
                                 })

    df['class_text'] = df['class'].apply(lambda class_: classes_dict[int(class_)])
    return df[['class', 'class_text', 'precision', 'recall', 'f1', 'support']]
