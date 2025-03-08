import requests
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

def search_google_images(query, api_key, num_images=10):
    """
    Search Google Images using SerpAPI
    
    Args:
        query (str): Search query
        api_key (str): Your SerpAPI key
        num_images (int): Number of images to retrieve
    
    Returns:
        list: List of image URLs
    """
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google_images",
        "ijn": 0,  # Page number (0 = first page)
        "num": min(100, num_images)  # SerpAPI allows max 100 per page
    }
    
    all_images = []
    page = 0
    
    # Continue fetching pages until we have enough images or no more results
    while len(all_images) < num_images:
        params["ijn"] = page
        
        response = requests.get("https://serpapi.com/search", params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
        
        data = response.json()
        
        # Check if we have image results
        if "images_results" not in data or not data["images_results"]:
            break
        
        # Extract image URLs
        for image in data["images_results"]:
            if "original" in image:
                all_images.append(image["original"])
            
            # Stop if we have enough images
            if len(all_images) >= num_images:
                break
        
        page += 1
        
        # Safety check to avoid too many requests
        if page > 10:
            break
            
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
    
    return all_images[:num_images]

def download_image(url, folder_path, filename):
    """Download an image from URL and save it to the specified path"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            # Determine file extension from content-type or URL
            if 'content-type' in response.headers and 'image' in response.headers['content-type']:
                ext = response.headers['content-type'].split('/')[-1]
            else:
                ext = url.split('.')[-1].lower()
                if len(ext) > 4 or ext not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
                    ext = 'jpg'
            
            # Create full filename with proper extension
            full_filename = f"{filename}.{ext}"
            file_path = os.path.join(folder_path, full_filename)
            
            # Save the image
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            print(f"Downloaded: {full_filename}")
            return True
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Get user inputs
    search_query = input("Enter search query: ")
    api_key = input("Enter your SerpAPI key: ")
    num_images = int(input("Enter number of images to download: "))
    folder_name = input("Enter folder name to save images: ")
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    
    print(f"Searching for '{search_query}' images...")
    image_urls = search_google_images(search_query, api_key, num_images)
    
    if not image_urls:
        print("No images found!")
        return
    
    print(f"Found {len(image_urls)} images. Starting download...")
    
    # Download images using multiple threads for faster processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, url in enumerate(image_urls):
            futures.append(
                executor.submit(download_image, url, folder_name, f"{search_query.replace(' ', '_')}_{i+1}")
            )
        
        # Wait for all downloads to complete
        completed = 0
        for future in futures:
            if future.result():
                completed += 1
    
    print(f"Download complete! {completed}/{len(image_urls)} images downloaded to '{folder_name}'")

if __name__ == "__main__":
    main()