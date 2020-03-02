import pandas as pd
import os
from helper.utils import create_dir, clustering, save_data, process
import sagemaker
from glob import glob
import boto3
import sys

os.environ['AWS_PROFILE'] = "aws-personal"
os.environ['AWS_DEFAULT_REGION'] = "us-west-2"

iam = boto3.client('iam')
role = iam.get_role(RoleName="AmazonSageMaker-ExecutionRole-20191130T020687")['Role']['Arn']
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
local_data_folder = './data'
prefix = "udacity-capstone-project"

ticker = sys.argv[1]
skip = pd.read_csv(local_data_folder + '/accuracy.csv')["ticker"].values
if ticker not in skip:
  try :
    print('Processing:', ticker)
    process(ticker, local_data_folder, bucket, role, prefix, sagemaker_session)
  except:
    e = sys.exc_info()
    print(e)
    print("Failed to process", ticker)
    pass
