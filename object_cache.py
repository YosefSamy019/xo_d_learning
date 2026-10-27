import pickle
import os

_object_cache_dir = "cache"
_file_extension = '.pickle'

def saveObject(obj, file_name: str):
    """Save an object to the cache directory."""
    
    os.makedirs(_object_cache_dir, exist_ok=True)
    
    file_name = file_name + _file_extension
    file_path = os.path.join(_object_cache_dir, file_name)
    
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Object successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving object: {e}")



def loadObject(file_name: str, callback=None):
    """Load an object from the cache directory or trigger a callback if file not found."""
    
    os.makedirs(_object_cache_dir, exist_ok=True)
    
    file_name = file_name + _file_extension
    file_path = os.path.join(_object_cache_dir, file_name)
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.", end='')
        if callback:
            print(", Return object from callback")
            return callback()
        else:
            print(", Return None, No callback passed")
            return None
    
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        print(f"Object successfully loaded from {file_path}")
        return obj
    except Exception as e:
        print(f"Error loading object: {e}")
        return None