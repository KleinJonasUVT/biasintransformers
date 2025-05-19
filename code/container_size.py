from azure.storage.blob import BlobServiceClient
import os
from collections import defaultdict

connection_string = os.getenv("SONAR_STORAGE_KEY")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Dictionary: {container_name: {subfolder_path: total_size_in_bytes}}
container_sizes = defaultdict(lambda: defaultdict(int))

# List all containers
all_containers = blob_service_client.list_containers()

for container_item in all_containers:
    container_name = container_item['name']
    container_client = blob_service_client.get_container_client(container_name)
    
    print(f"Processing container: {container_name}")
    
    # List all blobs in the container
    blob_list = container_client.list_blobs()

    for blob in blob_list:
        # Virtual folders are inferred from blob name
        blob_path = blob.name
        if "/" in blob_path:
            subfolder = blob_path.rsplit("/", 1)[0]
        else:
            subfolder = "<root>"
        
        container_sizes[container_name][subfolder] += blob.size

# Print results
for container, subfolders in container_sizes.items():
    print(f"\nContainer: {container}")
    for subfolder, size in subfolders.items():
        print(f"  Subfolder: {subfolder} - Size: {size / (1024 ** 3):.2f} GB")

