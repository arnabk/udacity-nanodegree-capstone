import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sagemaker.pytorch import PyTorch
import random
import numpy as np
import sagemaker
from sklearn.preprocessing import MinMaxScaler
from sagemaker import KMeans
import pickle

threshold = .4
BUY = 1
SELL = 2
NONE = 3

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
def save_data(cluster_data, folder_name, local_data_folder):
    Y = cluster_data[["Label"]]
    X = cluster_data.drop(columns=["Label"])
    create_dir(local_data_folder + '/s3/' + folder_name)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.33, random_state=1, shuffle=True)
    pd.concat([pd.DataFrame(y_train), pd.DataFrame(x_train)], axis=1)\
        .to_csv(local_data_folder + '/s3/' + folder_name + '/train.csv', header=False, index=False)
    pd.concat([pd.DataFrame(y_test), pd.DataFrame(x_test)], axis=1)\
        .to_csv(local_data_folder + '/s3/' + folder_name + '/validation.csv', header=False, index=False)
        
def generate_NN_predictor(ticker, bucket, prefix, role, sagemaker_session):
    s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/data/{}/train.csv'\
                                        .format(bucket, prefix, ticker), content_type='text/csv')
    s3_input_validation = sagemaker.s3_input(s3_data='s3://{}/{}/data/{}/validation.csv'\
                                             .format(bucket, prefix, ticker), content_type='text/csv')
    estimator = PyTorch(entry_point='train.py',
                        source_dir='pytorch', # this should be just "source" for your code
                        role=role,
                        framework_version='1.0',
                        train_instance_count=1,
                        train_instance_type='ml.c4.xlarge',
                        sagemaker_session=sagemaker_session,
                        hyperparameters={
                            'input_dim': 26,  # num of features
                            'hidden_dim': 260,
                            'output_dim': 1,
                            'epochs': 200 # could change to higher
                        })
    estimator.fit({ 'train': s3_input_train, 'validation': s3_input_validation })
    predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")
    return predictor

def generate_random_direction():
    rand_val = random.random()
    direction = NONE
    if rand_val >= .7:
        direction = BUY
    elif rand_val <= .3:
        direction = SELL
    return direction

def process(ticker, local_data_folder, bucket, role, prefix, sagemaker_session):
    df = pd.read_pickle('{}/{}.{}'.format(local_data_folder, ticker, 'pkl'))
    df.dropna(inplace=True)
    df.drop(columns=["Date"], inplace=True)
    df.loc[df.Label >= threshold, 'direction'] = BUY
    df.loc[df.Label <= -threshold, 'direction'] = SELL
    df.loc[(df.Label < threshold) & (df.Label > -threshold), 'direction'] = NONE

    # Normalize
    scaler = MinMaxScaler()

    Y_df = pd.DataFrame(df["Label"]).astype('float64')
    X_df = df.drop(columns=["Label"]).astype('float64')

    X = scaler.fit_transform(X_df)
    Y = scaler.fit_transform(Y_df)

    X[:, X.shape[1] - 1] = X_df["direction"].to_numpy()

    #### split data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.33, random_state=1, shuffle=True)

    # clustering
    s3_output_folder = "s3://{}/{}/output".format(bucket, prefix)
    kmeans = KMeans(role=role,
                train_instance_count=1,
                train_instance_type="ml.m4.xlarge",
                output_path=s3_output_folder,
                k=3)

    # Remove direction column and train
    kmeans.fit(kmeans.record_set(x_train[:, 0:x_train.shape[1] - 1].astype('float32')))

    # deploy
    print("Deploying model", kmeans.model_data)
    kmeans_predictor = kmeans.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")

    create_dir('{}/s3/{}'.format(local_data_folder, ticker))

    '''
        Label = Change in price(+ve, -ve, none)
        Direction = BUY, SELL, NONE
        Cluster = cluster_0, cluster_1, cluster_2
    '''
    # train data
    y_train_df = pd.DataFrame(y_train, columns=["Label"])
    x_train_df = pd.DataFrame(x_train, columns=['col-{}'.format(i) for i in range(x_train.shape[1] - 1)] + ["direction"])
    dataset_with_cluster = pd.concat([y_train_df.astype("float32"), x_train_df.astype("float32"),\
            clustering(x_train_df.drop(columns=["direction"]).astype('float32').values, kmeans_predictor)
        ], axis=1)
    dataset_with_cluster.to_csv('{}/s3/{}/all-train.csv'.format(local_data_folder, ticker), header=True, index=False)

    # test data
    y_test_df = pd.DataFrame(y_test, columns=["Label"])
    x_test_df = pd.DataFrame(x_test, columns=['col-{}'.format(i) for i in range(x_test.shape[1] - 1)] + ['direction'])
    pd.concat([y_test_df.astype("float32"), x_test_df.astype("float32")], axis=1)\
        .to_csv('{}/s3/{}/all-test.csv'.format(local_data_folder, ticker), header=True, index=False)

    # clean clustering end point
    kmeans_predictor.delete_endpoint(kmeans_predictor.endpoint)

    all_test_pred = pd.read_csv("{}/s3/{}/all-test.csv".format(local_data_folder, ticker)).dropna()
    all_train_pred = pd.read_csv("{}/s3/{}/all-train.csv".format(local_data_folder, ticker)).dropna()

    cluster0_df = dataset_with_cluster[dataset_with_cluster["Cluster"] == 0].drop(columns=["Cluster"])
    save_data(cluster0_df.drop(columns=["direction"]), ticker, local_data_folder)
    sagemaker_session.upload_data(path=local_data_folder + '/s3/' + ticker, bucket=bucket, key_prefix=prefix + '/data/' + ticker)
    estimator = generate_NN_predictor(ticker, bucket, prefix, role, sagemaker_session)
    all_test_pred["cluster0_pred"] = estimator.predict(all_test_pred.drop(columns=["Label", "direction"]).astype('float32').values)
    all_train_pred["cluster0_pred"] = estimator.predict(all_train_pred.drop(columns=["Label", "direction", "Cluster"]).astype('float32').values)
    estimator.delete_endpoint(estimator.endpoint)

    cluster1_df = dataset_with_cluster[dataset_with_cluster["Cluster"] == 1].drop(columns=["Cluster"])
    save_data(cluster1_df.drop(columns=["direction"]), ticker, local_data_folder)
    sagemaker_session.upload_data(path=local_data_folder + '/s3/' + ticker, bucket=bucket, key_prefix=prefix + '/data/' + ticker)
    estimator = generate_NN_predictor(ticker, bucket, prefix, role, sagemaker_session)
    all_test_pred["cluster1_pred"] = estimator.predict(all_test_pred.drop(columns=["Label", "direction", "cluster0_pred"]).astype('float32').values)
    all_train_pred["cluster1_pred"] = estimator.predict(all_train_pred.drop(columns=["Label", "direction", "Cluster", "cluster0_pred"]).astype('float32').values)
    estimator.delete_endpoint(estimator.endpoint)

    cluster2_df = dataset_with_cluster[dataset_with_cluster["Cluster"] == 2].drop(columns=["Cluster"])
    save_data(cluster2_df.drop(columns=["direction"]), ticker, local_data_folder)
    sagemaker_session.upload_data(path=local_data_folder + '/s3/' + ticker, bucket=bucket, key_prefix=prefix + '/data/' + ticker)
    estimator = generate_NN_predictor(ticker, bucket, prefix, role, sagemaker_session)
    all_test_pred["cluster2_pred"] = estimator.predict(all_test_pred.drop(columns=["Label", "direction", "cluster0_pred", "cluster1_pred"]).astype('float32').values)
    all_train_pred["cluster2_pred"] = estimator.predict(all_train_pred.drop(columns=["Label", "direction", "Cluster", "cluster0_pred", "cluster1_pred"]).astype('float32').values)
    estimator.delete_endpoint(estimator.endpoint)

    os.remove(local_data_folder + '/s3/' + ticker + '/train.csv')
    os.remove(local_data_folder + '/s3/' + ticker + '/validation.csv')

    all_buys = pd.DataFrame([cluster0_df[cluster0_df['direction'] == BUY].shape[0],
            cluster1_df[cluster1_df['direction'] == BUY].shape[0],
            cluster2_df[cluster2_df['direction'] == BUY].shape[0]], columns=["BUY"], index=["cluster0_pred", "cluster1_pred", "cluster2_pred"])

    all_sells = pd.DataFrame([cluster0_df[cluster0_df['direction'] == SELL].shape[0],
            cluster1_df[cluster1_df['direction'] == SELL].shape[0],
            cluster2_df[cluster2_df['direction'] == SELL].shape[0]], columns=["SELL"], index=["cluster0_pred", "cluster1_pred", "cluster2_pred"])

    all_nones = pd.DataFrame([cluster0_df[cluster0_df['direction'] == NONE].shape[0],
            cluster1_df[cluster1_df['direction'] == NONE].shape[0],
            cluster2_df[cluster2_df['direction'] == NONE].shape[0]], columns=["NONE"], index=["cluster0_pred", "cluster1_pred", "cluster2_pred"])

    cluster_selection_df = pd.concat([all_buys, all_sells, all_nones], axis=1)


    cluster_selection_index = cluster_selection_df.index
    buy_cluster_name = cluster_selection_index[cluster_selection_df['BUY'].values.argmax()]
    sell_cluster_name = cluster_selection_index[cluster_selection_df.drop(index=[buy_cluster_name])['SELL'].values.argmax()]
    none_cluster_name = cluster_selection_index[cluster_selection_df.drop(index=[buy_cluster_name, sell_cluster_name])['NONE'].values.argmax()]

    # Generate selected-cluster column based on max(cluster0, cluster1, cluster2)
    all_test_pred["selected-cluster"] = all_test_pred[["cluster0_pred", "cluster1_pred", "cluster2_pred"]].idxmax(axis=1)
    all_train_pred["selected-cluster"] = all_train_pred[["cluster0_pred", "cluster1_pred", "cluster2_pred"]].idxmax(axis=1)

    # convert selected-cluster to BUY, SELL, NONE
    all_test_pred.loc[all_test_pred["selected-cluster"] == buy_cluster_name, "prediction"] = BUY
    all_test_pred.loc[all_test_pred["selected-cluster"] == sell_cluster_name, "prediction"] = SELL
    all_test_pred.loc[all_test_pred["selected-cluster"] == none_cluster_name, "prediction"] = NONE

    all_train_pred.loc[all_train_pred["selected-cluster"] == buy_cluster_name, "prediction"] = BUY
    all_train_pred.loc[all_train_pred["selected-cluster"] == sell_cluster_name, "prediction"] = SELL
    all_train_pred.loc[all_train_pred["selected-cluster"] == none_cluster_name, "prediction"] = NONE

    # Bench mark results
    all_test_pred["random-prediction"] = [generate_random_direction() for _ in range(all_test_pred.shape[0])]
    all_train_pred["random-prediction"] = [generate_random_direction() for _ in range(all_train_pred.shape[0])]


    all_test_pred.to_csv('{}/s3/{}/all-test-pred.csv'.format(local_data_folder, ticker), index=None)
    all_train_pred.to_csv('{}/s3/{}/all-train-pred.csv'.format(local_data_folder, ticker), index=None)
    cluster_selection_df.to_csv('{}/s3/{}/cluster-selection.csv'.format(local_data_folder, ticker), index=None)

    # remove NA
    all_test_pred = all_test_pred.dropna()
    all_train_pred = all_train_pred.dropna()

    # test accuracy
    test_accuracy = accuracy_score(all_test_pred["direction"], all_test_pred["prediction"], normalize=True)
    benchmark_test_accuracy = accuracy_score(all_test_pred["direction"], all_test_pred["random-prediction"], normalize=True)
    print('Test accuracy:', test_accuracy, ", Benchmark:", benchmark_test_accuracy)

    # train accuracy
    train_accuracy = accuracy_score(all_train_pred["direction"], all_train_pred["prediction"], normalize=True)
    benchmark_train_accuracy = accuracy_score(all_train_pred["direction"], all_train_pred["random-prediction"], normalize=True)
    print('Train accuracy:', train_accuracy, ", Benchmark:", benchmark_train_accuracy)

    accuracy_df = pd.DataFrame([ticker, test_accuracy, benchmark_test_accuracy, train_accuracy, benchmark_train_accuracy]).T
    accuracy_df.columns = ["ticker", "test_accuracy", "benchmark_test_accuracy", "train_accuracy", "benchmark_train_accuracy"]

    accuracy_file = "{}/accuracy.csv".format(local_data_folder)
    header = not os.path.exists(accuracy_file)
    accuracy_df.to_csv(accuracy_file, mode="a", header=header, index=False)
