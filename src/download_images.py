
import pandas as pd
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'images')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'images_dataset.csv')

os.makedirs(DATA_DIR, exist_ok=True)

def download_image(args):
    url, filename = args
    path = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(path):
        return # Skip existing
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    print(f"Reading CSV from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Extract filename from URL
    # Assuming standard Amazon format or just use last part
    def get_filename(url):
        return url.split('/')[-1]

    print("Preparing download list...")
    tasks = []
    for url in df['image']:
        filename = get_filename(url)
        tasks.append((url, filename))

    print(f"Downloading {len(tasks)} images to {DATA_DIR}...")
    
    # Use ThreadPool for speed
    with ThreadPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(download_image, tasks), total=len(tasks)))

    print("Download complete!")

if __name__ == "__main__":
    main()
