import numpy as np
import pickle
from preprocess import extract_data

if __name__ == '__main__':
    datasets = ["ted-zh-en", "ted-ru-en", "robust19-ja-en", "robust19-en-ja"]
    modeldir = "/export/a10/kduh/p/mt/gridsearch/{0}/models/"
    multi_output = "/export/a08/xzhan138/Auto-tuning/multi_output/{0}/random/"
    single_output = "/export/a08/xzhan138/Auto-tuning/single_output/{0}/{1}/"
    random_seeds = [59, 18, 20, 73, 61, 29, 58, 65, 14, 30]

    # Multi-objective
    for dataset in datasets:
        x, y1, y2 = extract_data(modeldir.format(dataset), 5)
        for rs in random_seeds:
            result = np.zeros((len(y1)-3, len(y1)))
            np.random.seed(rs)
            for i in range(len(y1)-3):
                label_ids = np.array([i, i+1, i+2])
                unlabel_ids = np.array([id for id in range(len(y1)) if id not in label_ids])
                np.random.shuffle(unlabel_ids)
                result[i] = np.concatenate((label_ids, unlabel_ids))
            with open(multi_output.format(dataset) + "random" + str(rs) + ".pkl", 'wb') as fobj:
                pickle.dump(result, fobj)

    # Single-objective
    archi = "rnn"
    for dataset in ["ted-zh-en", "ted-ru-en"]:
        for cell in ["lstm", "rnn"]:
            x, y1, y2 = extract_data(modeldir.format(dataset), 5, architecture="rnn", rnn_cell_type=cell)
            result = np.zeros((len(y1)-3, len(y1)))
            np.random.seed(37)
            for i in range(len(y1)-3):
                label_ids = np.array([i, i+1, i+2])
                unlabel_ids = np.array([id for id in range(len(y1)) if id not in label_ids])
                np.random.shuffle(unlabel_ids)
                result[i] = np.concatenate((label_ids, unlabel_ids))
            with open(single_output.format(archi, dataset) + "random_" + cell + ".pkl", 'wb') as fobj:
                pickle.dump(result, fobj)

    archi = "trans"
    for dataset in datasets:
        x, y1, y2 = extract_data(modeldir.format(dataset), 5)
        result = np.zeros((len(y1)-3, len(y1)))
        np.random.seed(37)
        for i in range(len(y1)-3):
            label_ids = np.array([i, i+1, i+2])
            unlabel_ids = np.array([id for id in range(len(y1)) if id not in label_ids])
            np.random.shuffle(unlabel_ids)
            result[i] = np.concatenate((label_ids, unlabel_ids))
        with open(single_output.format(archi, dataset) + "random.pkl", 'wb') as fobj:
            pickle.dump(result, fobj)










