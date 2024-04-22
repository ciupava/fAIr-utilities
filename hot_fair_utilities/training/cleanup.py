# Standard library imports
import os
import shutil


def extract_highest_accuracy_model(output_path):
    model_checkpoints = os.path.join(output_path, "model-checkpts")
    print("model chkpoint parent folder:", model_checkpoints)
    assert os.path.exists(model_checkpoints), "Model Checkpoints Doesn't Exist"
    entries_folders = os.listdir(model_checkpoints)
    print('\n---\nModel checkpoints folders (one per batch size):', entries_folders)
    assert len(entries_folders) > 0, "Couldn't find any models"
    
    for entry_folder in entries_folders:
        print('\nEntry folder:', entry_folder)
        chkpoint_folder = os.path.join(model_checkpoints, entry_folder)
        assert os.path.exists(chkpoint_folder), "Model Checkpoints Doesn't Exist"
        
        entries = os.listdir(chkpoint_folder)
        print('\n Entries:', entries)
        # Create a variable to store the latest entry
        latest_entry = None

        # Create a variable to store the latest entry's modification time
        latest_time = 0

        # Iterate over the list of entries
        for entry in entries:
            # Use the os.stat() method to get the entry's modification time
            entry_time = os.stat(os.path.join(model_checkpoints, entry_folder, entry)).st_mtime
            # Check if the entry's modification time is greater than the latest time
            if entry_time > latest_time:
                # If the entry's modification time is greater, update the latest time and entry variables
                latest_time = entry_time
                latest_entry = entry

        # get the highest accuracy model one
        latest_entry_path = os.path.join(model_checkpoints, entry_folder, latest_entry)
        print('- - latest entry path', latest_entry_path)
        # ---
        # To remove entries that are not the latest, create new list of not latest entries
        not_latest_entries = [entry for entry in entries if entry != latest_entry]
        print("\nLatest entry: ", latest_entry)
        print("\nList of not latest_entries: ", not_latest_entries)
        # Loop over list of not_latest_entries and remove the checkpoint file
        for entry in not_latest_entries:
            # os.remove(os.path.join(model_checkpoints, entry_folder, entry))
            shutil.rmtree(os.path.join(model_checkpoints, entry_folder, entry))
        # ---
        
        highest_accuracy = 0
        highest_accuracy_entry = None
        # for entry in os.listdir(latest_entry_path):
        for entry in os.listdir(chkpoint_folder):
            print('----> entry is ', entry)
            parts = entry.split("_")

            accuracy = parts[-1][:-3]  # remove .tf
            print('- - accuracy:', accuracy)
            if float(accuracy) * 100 > highest_accuracy:
                highest_accuracy_entry = entry
                highest_accuracy = float(accuracy) * 100
        print('\nHighest accuracy entry:', highest_accuracy_entry)
        # for entry in os.listdir(latest_entry_path):
        for entry in os.listdir(chkpoint_folder):
            # Check if the entry is not the file or directory you want to keep
            print('\nAgain entry:', entry)
            if entry != highest_accuracy_entry:
                # If the entry is not the file or directory you want to keep, use the os.remove() method to remove it
                # shutil.rmtree(os.path.join(latest_entry_path, entry))
                shutil.rmtree(os.path.join(latest_entry_path, entry))
    return highest_accuracy, os.path.join(latest_entry_path, highest_accuracy_entry)
