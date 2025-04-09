import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def upload_file_to_s3(file_path, bucket, object_name=None):
    """Uploads a single file to an S3 bucket."""
    # Use the file’s basename as the object name if none is provided
    if object_name is None:
        object_name = os.path.basename(file_path)

    s3 = boto3.client('s3')
    try:
        s3.upload_file(file_path, bucket, object_name)
        print(f"Upload successful for: {file_path}")
    except (NoCredentialsError, ClientError) as e:
        print(f"Upload failed for {file_path}: {e}")

# Directory containing the files to upload
folder = "context_refining_query"
bucket_name = "sbldatabase"

# Iterate over each item in the folder
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    # Ensure it is a file (skip directories)
    if os.path.isfile(file_path):
        upload_file_to_s3(file_path, bucket_name)
