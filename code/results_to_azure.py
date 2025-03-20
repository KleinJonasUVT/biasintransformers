import os
import glob
from azure.storage.blob import BlobServiceClient

class AzureBlobUploader:
    def __init__(self, connection_string: str, container_name: str, target_folder: str):
        """
        Initializes the AzureBlobUploader class.
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.target_folder = target_folder
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def upload_file(self, local_path: str, blob_path: str):
        """
        Uploads a single file to Azure Blob Storage.
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            with open(local_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"Uploaded: {local_path} -> {blob_path}")
        except Exception as e:
            print(f"Failed to upload {local_path}: {e}")

    def upload_directory(self, directory: str):
        """
        Uploads selected files from a directory based on the filtering logic.
        """
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            return

        count = 0  # Track the number of uploaded files
        for root, _, files in os.walk(directory):
            print(f"Uploading files from: {root}")
            for file in sorted(files):  # Ensure a consistent order
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, directory)
                blob_path = f"{self.target_folder}/{directory}/{relative_path}".replace("\\", "/")
                print(f"Local: {local_path} -> Blob: {blob_path}")

                if count < 40:  # First 40 multiples of 296
                    if (count + 1) % 296 == 0:
                        self.upload_file(local_path, blob_path)
                        count += 1
                else:  # After the first 40, every 200th multiple of 296
                    if (count + 1) >= 59200 and (count + 1) % (296 * 200) == 0:
                        self.upload_file(local_path, blob_path)
                        count += 1

    def upload_multiple_directories(self, directories: list):
        """
        Upload multiple directories.
        """
        for directory in directories:
            self.upload_directory(directory)

# Fetch the connection string from environment variables
connection_string = os.getenv("SONAR_STORAGE_KEY")
container_name = "results"  # Change this if needed
target_folder = "bert_dynamic_full"  # Change this as needed

directories_to_upload = [
    "bert_checkpoints",
    "custom_tokenizer",
    "train_dataset",
    "valid_dataset",
    "test_dataset"
]

print("This script is running")

for directory in directories_to_upload:
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")

uploader = AzureBlobUploader(connection_string, container_name, target_folder)
print("Uploader initialized. Starting upload...")
uploader.upload_multiple_directories(directories_to_upload)
