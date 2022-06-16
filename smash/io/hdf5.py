from __future__ import annotations

import h5py

def save_mesh(mesh: dict, path: str):
    
    with h5py.File(path, "w") as f:
    
        for key, value in mesh.items():
            
            if isinstance(value, bytes):
                value = value.strip()
                
                f.attrs[key] = value
                
            elif isintance(value, np.ndarray):
                ...
