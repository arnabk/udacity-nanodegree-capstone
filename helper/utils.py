import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dir(dir):
  os.makedirs(dir, exist_ok=True)

# Generate clusters for data
def clustering(data, kmeans_predictor):
    clustering_result = kmeans_predictor.predict(pd.DataFrame(data).astype('float32').values)
    clustering_result = list(map(lambda x:x.label["closest_cluster"].float32_tensor.values[0], clustering_result))

    assert len(clustering_result) == len(data), "Length mis-match with clustering and input data"

    cluster_category = pd.DataFrame(clustering_result, columns=["Cluster"])
    x_train_with_cluster = pd.concat([pd.DataFrame(data), cluster_category], axis=1)
    return cluster_category

# save data to local dir
def save_data(cluster_data, folder_name, split_data, local_data_folder):
    Y = cluster_data[["Label"]]
    X = cluster_data.drop(columns=["Label"])
    create_dir(local_data_folder + 's3/' + folder_name)
    if split_data:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.33, random_state=1, shuffle=True)
        pd.concat([pd.DataFrame(y_train), pd.DataFrame(x_train)], axis=1)\
            .to_csv(local_data_folder + 's3/' + folder_name + '/train.csv', header=False, index=False)
        pd.concat([pd.DataFrame(y_test), pd.DataFrame(x_test)], axis=1)\
            .to_csv(local_data_folder + 's3/' + folder_name + '/validation.csv', header=False, index=False)
    else:
        pd.concat([pd.DataFrame(Y), pd.DataFrame(X)], axis=1)\
            .to_csv(local_data_folder + 's3/' + folder_name + '/all-test.csv', header=False, index=False)
