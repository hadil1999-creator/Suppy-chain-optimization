import os
import boto3
import s3fs
import sagemaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

# Load SageMaker session and role
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define S3 paths
bucket = 'your-bucket-name'
prefix = 'prysmian-forecasting'
s3_data_path = f's3://{bucket}/{prefix}/data/train'
s3_output_path = f's3://{bucket}/{prefix}/output'

# Added - Define checkpoint path for resuming training
s3_checkpoint_path = f's3://{bucket}/{prefix}/checkpoints/'

# Select AWS DeepAR image
image_name = get_image_uri(boto3.Session().region_name, 'forecasting-deepar')

# Model parameters
freq = 'M'
prediction_length = 5
context_length = 12

# Read dataset
data = pd.read_excel(s3_data_path + '/RawMaterialItems.xlsx', parse_dates=True, index_col=0)
num_timeseries = data.shape[1]
data_length = data.index.size
t0 = data.index[0]

# Convert data to time series
time_series = []
for i in range(num_timeseries):
    index = pd.date_range(start=t0, freq=freq, periods=data_length)
    time_series.append(pd.Series(data=data.iloc[:, i], index=index))

# Split data into training and test sets
time_series_training = [ts[:-prediction_length] for ts in time_series]

# Convert to JSON format for DeepAR
def series_to_obj(ts):
    return {"start": str(ts.index[0]), "target": list(ts)}

def series_to_jsonline(ts):
    return json.dumps(series_to_obj(ts))

# Upload training and test data to S3
s3filesystem = s3fs.S3FileSystem()

with s3filesystem.open(s3_data_path + "/train/train.json", 'wb') as fp:
    for ts in time_series_training:
        fp.write(series_to_jsonline(ts).encode("utf-8"))
        fp.write('\n'.encode("utf-8"))

with s3filesystem.open(s3_data_path + "/test/test.json", 'wb') as fp:
    for ts in time_series:
        fp.write(series_to_jsonline(ts).encode("utf-8"))
        fp.write('\n'.encode("utf-8"))

# Added -  Use training checkpoint in the estimator
estimator = sagemaker.estimator.Estimator(
    sagemaker_session=sagemaker_session,
    image_uri=image_name,
    role=role,
    instance_count=1,
    instance_type='ml.c4.xlarge',
    base_job_name='DEMO-deepar',
    output_path=s3_output_path,
    checkpoint_s3_uri=s3_checkpoint_path  # Path added
)

# Set hyperparameters
estimator.set_hyperparameters(
    time_freq=freq,
    context_length=str(context_length),
    prediction_length=str(prediction_length),
    num_cells="40",
    num_layers="3",
    likelihood="gaussian",
    epochs="80",
    mini_batch_size="32",
    learning_rate="0.001",
    dropout_rate="0.05",
    early_stopping_patience="10"
)

# Train model with checkpointing
data_channels = {
    "train": f"{s3_data_path}/train/",
    "test": f"{s3_data_path}/test/"
}

estimator.fit(inputs=data_channels, wait=True)

# Deploy Model
job_name = estimator.latest_training_job.name

endpoint_name = sagemaker_session.endpoint_from_job(
    job_name=job_name,
    initial_instance_count=1,
    instance_type='ml.m4.xlarge',
    deployment_image=image_name,
    role=role
)

# Predictor Class
class DeepARPredictor(sagemaker.predictor.RealTimePredictor):
    def set_prediction_parameters(self, freq, prediction_length):
        self.freq = freq
        self.prediction_length = prediction_length

    def predict(self, ts, encoding="utf-8", num_samples=100, quantiles=["0.1", "0.75", "0.9"]):
        instances = [{"start": str(ts.index[0]), "target": list(ts)}]
        config = {"num_samples": num_samples, "output_types": ["quantiles"], "quantiles": quantiles}
        request_data = {"instances": instances, "configuration": config}
        response = super(DeepARPredictor, self).predict(json.dumps(request_data).encode(encoding))
        return json.loads(response.decode(encoding))

# Instantiate and test predictor
predictor = DeepARPredictor(endpoint=endpoint_name, sagemaker_session=sagemaker_session, content_type="application/json")
predictor.set_prediction_parameters(freq, prediction_length)

# Make predictions
list_of_df = predictor.predict(time_series_training[:5])

# Plot results
for k, df in enumerate(list_of_df):
    plt.figure(figsize=(12,6))
    time_series[k][-prediction_length-context_length:].plot(label='target')
    p10, p90 = df['0.1'], df['0.9']
    plt.fill_between(p10.index, p10, p90, color='y', alpha=0.5, label='80% confidence interval')
    df['0.75'].plot(label='prediction median')
    plt.legend()
    plt.show()

# Cleanup
sagemaker_session.delete_endpoint(endpoint_name)
