import requests

# Search for objects that have images - set hasImages=true
response = requests.get("https://collectionapi.metmuseum.org/public/collection/v1/search",
                         params={"hasImages": "true", "q": "*"})  # q="*" for all objects
data = response.json()

image_ids = data["objectIDs"]
print(f"Total objects with images: {len(image_ids)}")

import os
import time

save_dir = "downloaded_images"
os.makedirs(save_dir, exist_ok=True)

for idx, obj_id in enumerate(image_ids):
    try:
        print(f"[{idx+1}/{len(image_ids)}] Processing object ID: {obj_id}")

        details_resp = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1//{obj_id}", timeout=10)
        details = details_resp.json()
        img_url = details.get("primaryImageSmall", "")
        if img_url:
            img_resp = requests.get(img_url, timeout=2)
            if img_resp.status_code == 200:
                img_data = img_resp.content
                img_path = os.path.join(save_dir, f"{obj_id}.jpg")
                with open(img_path, "wb") as handler:
                    handler.write(img_data)
                print(f"Downloaded image for object ID {obj_id}")
            else:
                print(f"Failed to download image for {obj_id}, status code {img_resp.status_code}")
        else:
            print(f"No image available for object ID {obj_id}")
    except Exception as e:
        print(f"Error processing object ID {obj_id}: {e}")
    time.sleep(0.5)
