import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

class S3Uploader:
    def __init__(self):
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=self.region
        )

    def upload_file(self, file_path: str, object_name: str) -> str:
        """
        Uploads a file to S3 and returns a public S3 URL.
        Note: The bucket/object must have public read access for this URL to work.
        """
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is not set")

        try:
            # Upload the file
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            
            # Generate a public S3 URL (no credentials in URL)
            # Format: https://bucket-name.s3.region.amazonaws.com/key
            public_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{object_name}"
            return public_url

        except (NoCredentialsError, ClientError) as e:
            print(f"Failed to upload to S3: {e}")
            raise e

