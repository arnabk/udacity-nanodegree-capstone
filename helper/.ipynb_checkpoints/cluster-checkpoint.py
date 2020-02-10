import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sagemaker import KMeans
import os
from helper.utils import create_dir

def cluster_helper(role, sagemaker_session, bucket, local_data_folder, prefix, ticker):
  A_df = pd.read_pickle(local_data_folder + ticker + '.pkl')
  A_df.dropna(inplace=True)
  A_df.drop(columns=["Date"], inplace=True)

  # Normalize
  scaler = MinMaxScaler()

  Y_df = pd.DataFrame(A_df["Label"]).astype('float64')
  X_df = A_df.drop(columns=["Label"]).astype('float64')

  X = scaler.fit_transform(X_df)
  Y = scaler.fit_transform(Y_df)

  # split data
  print("Splitting data")
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.33, random_state=1, shuffle=True)

  # clustering
  s3_output_folder = "s3://{}/{}/output".format(bucket, prefix)
  print("Clustering")
  kmeans = KMeans(role=role,
                train_instance_count=1,
                train_instance_type="ml.m4.xlarge",
                output_path=s3_output_folder,
                k=3)

  kmeans.fit(kmeans.record_set(pd.DataFrame(x_train).astype('float32').values))

  # deploy
  print("Deploying model", kmeans.model_data)
  kmeans_predictor = kmeans.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")


  create_dir('{}s3/{}'.format(local_data_folder, ticker))

  # upload train and test data to S3
  dataset_with_cluster = pd.concat([pd.DataFrame(y_train, columns=["label"]).astype("float32"), \
            pd.DataFrame(x_train).astype("float32"),\
            clustering(x_train)
            ], axis=1)
  dataset_with_cluster.to_csv('{}s3/{}/all-train.csv'.format(local_data_folder, ticker), header=False, index=False)
  # prepare cluster data sets    
  create_dir('{}s3/{}/train'.format(local_data_folder, ticker))
  save_data(dataset_with_cluster[dataset_with_cluster["cat"] == 0], "{}/train/cluster-0".format(ticker))
  save_data(dataset_with_cluster[dataset_with_cluster["cat"] == 1], "{}/train/cluster-1".format(ticker))
  save_data(dataset_with_cluster[dataset_with_cluster["cat"] == 2], "{}/train/cluster-2".format(ticker))

  # We have to predict the clusters for each of the test data sets so that we could use it for testing out next model
  dataset_with_cluster = pd.concat([pd.DataFrame(y_test, columns=["label"]).astype("float32"), \
            pd.DataFrame(x_test).astype("float32"),\
            clustering(x_test)
            ], axis=1)
  dataset_with_cluster.to_csv(local_data_folder + 's3/{}/all-test.csv'.format(ticker), header=False, index=False)
  # # prepare cluster data sets    
  create_dir('{}s3/{}/test'.format(local_data_folder, ticker))
  save_data(dataset_with_cluster[dataset_with_cluster["cat"] == 0], "{}/test/cluster-0".format(ticker))
  save_data(dataset_with_cluster[dataset_with_cluster["cat"] == 1], "{}/test/cluster-1".format(ticker))
  save_data(dataset_with_cluster[dataset_with_cluster["cat"] == 2], "{}/test/cluster-2".format(ticker))

  # delete endpoint
  kmeans_predictor.delete_endpoint(kmeans_predictor.endpoint)

  print('Completed clustering for', ticker)

  # delete endpoint
  kmeans_predictor.delete_endpoint(kmeans_predictor.endpoint)

  print('Completed clustering for', ticker)