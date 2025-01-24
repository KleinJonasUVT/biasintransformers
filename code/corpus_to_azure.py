import os
from azure.storage.blob import BlobServiceClient

class BlobUploader:
    def __init__(self, connection_string, container_name):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(self.container_name)

    def upload_files(self, local_directory):
        try:
            for root, _, files in os.walk(local_directory):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    blob_name = os.path.relpath(file_path, local_directory).replace("\\", "/")

                    with open(file_path, "rb") as data:
                        print(f"Uploading {file_name} as {blob_name}...")
                        self.container_client.upload_blob(name=blob_name, data=data, overwrite=True)

            print("All files uploaded successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Define the connection string and the container name
connection_string = os.getenv("SONAR_STORAGE_KEY")
container_name = "press-releases"

# Local directory containing the files
local_directory = "/Users/jonasklein/Library/CloudStorage/OneDrive-Personal/DSS/Thesis/code/SoNaRCorpus_NC_1.2/SONAR500/DCOI/WR-P-E-F_press_releases"

# Instantiate the class and upload files
uploader = BlobUploader(connection_string, container_name)
uploader.upload_files(local_directory)
