import glob
import os


def clear_directories():
    """
    Deletes all files inside the directories:
    results/, results/evolution/, results/plots/.
    Does not delete the directories themselves, only their contents.
    """
    
    # Ensure directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/evolution", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    directories = [
        "results/",
        "results/evolution/",
        "results/plots/",
        "results/logs/batteryagent",
        "results/logs/gridagent",
        "results/logs/loadagent",
        "results/logs/solaragent",
        "results/logs/windagent",
    ]

    for dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"Directory does not exist: {dir_path}")
            continue

        files = glob.glob(os.path.join(dir_path, "*"))
        if not files:
            print(f"No files in {dir_path}")
            continue

        for file_path in files:
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Could not delete {file_path}: {e}")
            else:
                print(f"Ignored (not a file): {file_path}")

    print("Cleanup completed.")