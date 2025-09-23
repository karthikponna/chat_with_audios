import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def fetch_file_data_from_s3(bucket, object_name):
    """
    Fetches the content of a single file from the specified S3 bucket.
    
    Parameters:
        bucket (str): The name of the S3 bucket.
        object_name (str): The key of the S3 object.
    
    Returns:
        bytes: The content of the file in bytes, or None if an error occurred.
    """
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=bucket, Key=object_name)
        file_data = response['Body'].read()
        print(f"Fetched data for: {object_name}")
        return file_data
    except (NoCredentialsError, ClientError) as e:
        print(f"Failed to fetch {object_name}: {e}")
        return None

def fetch_all_files_data_from_bucket(bucket):
    """
    Fetches the contents of all files from the specified S3 bucket.
    
    Parameters:
        bucket (str): The name of the S3 bucket.
        
    Returns:
        dict: A dictionary with S3 object keys as keys and file data (bytes) as values.
    """
    s3 = boto3.client('s3')
    file_data_dict = {}
    try:
        # List objects in the S3 bucket
        response = s3.list_objects_v2(Bucket=bucket)
        if 'Contents' not in response:
            print("No files found in the bucket.")
            return file_data_dict
        
        # Retrieve content for each object
        for obj in response['Contents']:
            key = obj['Key']
            data = fetch_file_data_from_s3(bucket, key)
            if data is not None:
                file_data_dict[key] = data
        return file_data_dict
    except (NoCredentialsError, ClientError) as e:
        print(f"Error listing objects in bucket {bucket}: {e}")
        return file_data_dict

# Example usage:
bucket_name = "sbldatabase"

# Fetch data from all files in the bucket directly (in-memory)
data_dict = fetch_all_files_data_from_bucket(bucket_name)

# Process or inspect the fetched data (data is in bytes)
for key, data in data_dict.items():
    print(f"Object: {key} fetched with data length: {len(data)} bytes")
