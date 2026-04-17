from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import numpy as np



from PIL import Image
import imageio

def visualize_molecule_with_coords(smiles, coords):
    """
    Visualize a molecule with 3D coordinates (excluding hydrogens) using py3Dmol,
    and label each atom with its atomic index.

    Args:
        smiles (str): SMILES string of the molecule
        coords (np.ndarray): (N_atoms_noH, 3) array of 3D coordinates
    """
    from rdkit import Chem
    import py3Dmol

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)  # remove hydrogens to match coordinate shape

    if mol.GetNumAtoms() != coords.shape[0]:
        raise ValueError(f"Number of non-H atoms in molecule ({mol.GetNumAtoms()}) does not match coordinate shape {coords.shape}")

    # Write coordinates to conformer
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = coords[i]
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))

    mol.RemoveAllConformers()
    mol.AddConformer(conf)

    # Generate mol block with 3D coordinates
    block = Chem.MolToMolBlock(mol)

    # Visualization with atom index labels
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(block, 'mol')
    viewer.setStyle({'stick': {}})
    
    # Add index labels instead of atomic symbols
    for i in range(mol.GetNumAtoms()):
        pos = coords[i]
        viewer.addLabel(str(i), {
            'position': {'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2])},
            'backgroundColor': 'white',
            'fontColor': 'black',
            'fontSize': 14,
            'showBackground': False
        })

    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    return viewer.show()


import os
import numpy as np
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

def visualize_and_save_frame(smiles, coords, filename,):
    """
    Visualize a molecule with given coordinates and save as PNG.
    If py3Dmol can't instantiate a viewer (e.g., headless), optionally fall back to RDKit 2D.

    Args:
        smiles (str): SMILES string
        coords (np.ndarray): (N_atoms, 3) coordinates
        filename (str): path to save image
        show (bool): if True, display viewer in Jupyter
        fallback_2d (bool): use RDKit 2D if py3Dmol PNG fails
        size (int): image width/height in pixels
    """
    # Build molecule and set coordinates
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES.")
    #mol = Chem.AddHs(mol)  # ensure atom count matches coordinates if coords include H
    if coords.shape != (mol.GetNumAtoms(), 3):
        # If your coords are heavy-atom only, remove Hs instead:
        # mol = Chem.RemoveHs(mol)
        # (then re-check shape)
        raise ValueError(f"Atom/coord mismatch: mol has {mol.GetNumAtoms()} atoms, coords are {coords.shape}")

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = float(coords[i,0]), float(coords[i,1]), float(coords[i,2])
        conf.SetAtomPosition(i, (x, y, z))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    # Prepare 3D display via py3Dmol
    block = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=800, height=800)
    viewer.addModel(block, 'sdf')  # MolBlock is SDF format
    viewer.setStyle({'stick': {}})
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    viewer.write_html(f=f'{filename}.html', fullpage = True)
    



def make_video_from_frames(folder, output_path='video.mp4', fps=10):
    """
    Combine PNG frames in folder into a single video using imageio.

    Args:
        folder (str): folder containing PNGs
        output_path (str): output .mp4 file path
        fps (int): frames per second
    """
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')
    ])
    images = [Image.open(f) for f in files]
    imageio.mimsave(output_path, images, fps=fps)
