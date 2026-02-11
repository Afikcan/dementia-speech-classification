import os
import gdown

def download_folder(folder_url, output_dir):
    """
    Downloads a folder from Google Drive using gdown library.
    Note: If the folder has more than 50 files, gdown cannot list all of them.
    Workaround: 
    1. Use `remaining_ok=True` to download what is possible.
    2. For full download, manually download the folder as a .zip from Google Drive,
       or use `gdown` with cookies: https://github.com/wkentaro/gdown#if-you-get-an-error
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Downloading from {folder_url} to {output_dir}...")
    print("Note: If the folder has >50 files, only the first 50 will be downloaded.")
    try:
        # remaining_ok=True allows partial download if not all files can be listed
        gdown.download_folder(url=folder_url, output=output_dir, quiet=False, remaining_ok=True)
        print(f"Download complete for {output_dir}")
    except Exception as e:
        print(f"Error downloading folder: {e}")
        print("TIP: For folders with >50 files, manually download the folder as a .zip from Google Drive.")

if __name__ == "__main__":
    dementia_url = "https://drive.google.com/drive/folders/1GKlvbU57g80-ofCOXGwatDD4U15tpJ4S"
    nodementia_url = "https://drive.google.com/drive/folders/1jm7w7J8SfuwKHpEALIK6uxR9aQZR1q8I"
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    
    download_folder(dementia_url, os.path.join(data_dir, "dementia"))
    download_folder(nodementia_url, os.path.join(data_dir, "nodementia"))
