import argparse
import numpy as np

def check_npz_file(feature_path):
    try:
        data = np.load(feature_path + ".npz", allow_pickle=True)
        print("Keys in the npz file:", data.files)
        for key in data.files:
            print(f"{key}: {data[key]}")
    except FileNotFoundError:
        print(f"File {feature_path}.npz not found.")

def read_features(feature_path):
    try:
        data = np.load(feature_path + ".npz", allow_pickle=True)
        images_name = data["images_name"]
        images_emb = data["images_emb"]
        return images_name, images_emb
    except FileNotFoundError:
        print(f"File {feature_path}.npz not found.")
        return None

def delete_entries(feature_path, entries_to_delete):
    # Step 1: Load existing data
    try:
        data = np.load(feature_path + ".npz", allow_pickle=True)
        images_name = data["images_name"]
        images_emb = data["images_emb"]
    except FileNotFoundError:
        print(f"File {feature_path}.npz not found.")
        return

    # Step 2: Remove entries to be deleted
    # Assume entries_to_delete is a list of names to be deleted
    delete_mask = np.isin(images_name, entries_to_delete)
    keep_mask = ~delete_mask

    images_name = images_name[keep_mask]
    images_emb = images_emb[keep_mask]

    # Step 3: Save the updated data
    np.savez_compressed(feature_path, images_name=images_name, images_emb=images_emb)

def main():
    parser = argparse.ArgumentParser(description="Delete entries from a face recognition database.")
    parser.add_argument("--file_path", 
                        type=str, 
                        default= "datasets/face_features/feature",
                        help="Path to the face recognition features file (.npz)")
    parser.add_argument("--user", 
                        nargs="+", 
                        type=str, 
                        default = "huy",
                        help="User names to delete (space-separated)")

    args = parser.parse_args()

    feature_path = args.file_path
    images_name = args.user

    #view existing features
    check_npz_file(feature_path)
    
    # Read existing features
    features = read_features(feature_path)

    if features is not None:
        images_name, _ = features

        # Check if user names to delete are valid
        invalid_user_names = set(images_name) - set(images_name)
        if invalid_user_names:
            print(f"Error: The following user names do not exist in the database: {', '.join(invalid_user_names)}")
        else:
            # Delete entries
            delete_entries(feature_path, images_name)
            print("Entries deleted successfully.")

if __name__ == "__main__":
    # main()
    check_npz_file("/home/khuy/Documents/test/face-recognition/datasets/face_features/feature")

