import os
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
        Uploads all files in a directory unless it's a checkpoint directory, in which case applies filtering logic.
        """
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            return
        
        if directory.endswith("_checkpoints"):
            self.upload_checkpoints(directory)
        else:
            for root, _, files in os.walk(directory):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, directory)
                    blob_path = f"{self.target_folder}/{directory}/{relative_path}".replace("\\", "/")
                    self.upload_file(local_path, blob_path)

    def upload_checkpoints(self, directory: str):
        """
        Uploads entire checkpoint folders if they meet filtering criteria.
        """
        checkpoint_numbers = []
        checkpoint_folders = []
        
        for checkpoint_folder in os.listdir(directory):
            checkpoint_path = os.path.join(directory, checkpoint_folder)
            if os.path.isdir(checkpoint_path) and checkpoint_folder.startswith("checkpoint-"):
                try:
                    checkpoint_num = int(checkpoint_folder.split("-")[1])
                    checkpoint_numbers.append((checkpoint_num, checkpoint_path))
                except ValueError:
                    continue
        
        checkpoint_numbers.sort()
        count = 0
        
        for checkpoint_num, checkpoint_path in checkpoint_numbers:
            relative_path = os.path.relpath(checkpoint_path, directory)
            blob_folder_path = f"{self.target_folder}/{directory}/{relative_path}".replace("\\", "/")
            
            # First 40 checkpoints: Multiples of 296
            if count < 40 and checkpoint_num % 296 == 0:
                self.upload_full_folder(checkpoint_path, blob_folder_path)
                count += 1
            # After the first 40, upload every 200th multiple of 296
            elif count >= 40 and checkpoint_num % (296 * 200) == 0:
                self.upload_full_folder(checkpoint_path, blob_folder_path)
                count += 1

    def upload_full_folder(self, local_folder: str, blob_folder_path: str):
        """
        Uploads all files inside a folder to Azure Blob Storage while preserving structure.
        """
        for root, _, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_folder)
                full_blob_path = f"{blob_folder_path}/{relative_path}".replace("\\", "/")
                self.upload_file(local_path, full_blob_path)

    def upload_multiple_directories(self, directories: list):
        """
        Upload multiple directories.
        """
        for directory in directories:
            self.upload_directory(directory)

# Fetch the connection string from environment variables
connection_string = os.getenv("SONAR_STORAGE_KEY")
container_name = "results"  
target_folder = "bert_dynamic_full"  

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