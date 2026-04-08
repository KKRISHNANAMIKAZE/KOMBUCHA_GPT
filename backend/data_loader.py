import os
import gdown

def download_files():
    os.makedirs("data", exist_ok=True)

    files = {
        "kombucha_index.faiss": "1l_A4-PpN_qnZCNco38ggpLEyWNvbpCRJ",
        "chunks.npy": "1hdX2RAOyX8xhDZ2hGIEI6eb3V9ooLsl8",
        "metadata.npy": "1avuMYw8nicHkvX3IfvVRuBcGbIxFGGTF"
    }

    for filename, file_id in files.items():
        path = os.path.join("data", filename)

        if os.path.exists(path):
            print(f"✅ {filename} already exists")
            continue

        print(f"⬇️ Downloading {filename}...")

        url = f"https://drive.google.com/uc?id={file_id}"

        try:
            gdown.download(url, path, quiet=True, fuzzy=True)
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")

    print("✅ All files ready")