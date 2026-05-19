import os
import shutil

def cleanup_workspace():
    print("🧹 Cleaning up old Porto Project files...")
    
    # Files to delete
    files_to_delete = [
        "Architecture_diagram.png", "Gagan.txt", "PIPELINE_DETAILS.md", 
        "PROJECT_UNDERSTANDING.md", "app.py", "build_1km_speed_model.py", 
        "df_model_data_full.csv", "df_model_data_full_01.csv", 
        "df_speed_data_full.csv", "eda_plots.png", "flask_server.err.log", 
        "flask_server.out.log", "lat_bins.pkl", "lat_bins_1km.pkl", 
        "lon_bins.pkl", "lon_bins_1km.pkl", "main.ipynb", "preprocessing.ipynb", 
        "train.csv", "train_only_fast.py", "xgb_grid_model_prod.pkl", 
        "xgb_speed_model_prod.pkl", "xgboost_learning_curve.png"
    ]
    
    # Folders to delete
    folders_to_delete = ["templates", "__pycache__"]
    
    base_dir = r"d:\Semester_06_\ITS\DS_exteded_project"
    
    deleted_files = 0
    for file in files_to_delete:
        path = os.path.join(base_dir, file)
        if os.path.exists(path):
            os.remove(path)
            deleted_files += 1
            print(f"  [-] Deleted: {file}")
            
    deleted_folders = 0
    for folder in folders_to_delete:
        path = os.path.join(base_dir, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
            deleted_folders += 1
            print(f"  [-] Deleted Folder: {folder}/")
            
    print(f"\n✅ Cleanup Complete! Deleted {deleted_files} files and {deleted_folders} folders.")
    print("The workspace is now clean for the Deep Learning project.")

if __name__ == "__main__":
    cleanup_workspace()
