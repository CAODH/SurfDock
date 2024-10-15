
from rdkit import Chem
import os
from tqdm import tqdm
tqdm.pandas()
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from dimorphite_dl import DimorphiteDL
from typing import List, Union
RDLogger.DisableLog('rdApp.*')

def standardize_smi(smiles,basicClean=True,clearCharge=True, clearFrag=True, canonTautomer=True, isomeric=False):
    try:
        clean_mol = Chem.MolFromSmiles(smiles)
        # del H , metal
        if basicClean:
            clean_mol = rdMolStandardize.Cleanup(clean_mol) 
        if clearFrag:

            clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        if clearCharge:
            uncharger = rdMolStandardize.Uncharger() 
            clean_mol = uncharger.uncharge(clean_mol)
        
        if canonTautomer:
            te = rdMolStandardize.TautomerEnumerator() # idem
            clean_mol = te.Canonicalize(clean_mol)
        stan_smiles=Chem.MolToSmiles(clean_mol, isomericSmiles=isomeric)
    except Exception as e:
        print (e, smiles)
        return None
    return stan_smiles
def GetNumTautomers(mol,Canonicalize=True,MaxTautomers=10):
    enumerator = rdMolStandardize.TautomerEnumerator()
    if Canonicalize:
        mol = enumerator.Canonicalize(mol)
        return [mol]
    else:
        enumerator.SetMaxTautomers(MaxTautomers)
        tautomers = enumerator.Enumerate(mol)
    return tautomers
def protonate(mol,min_ph=7.4,max_ph=7.4,max_variants=128,label_states=False,pka_precision=1.0)->List[str]:
    dimorphite_dl = DimorphiteDL(
        min_ph=min_ph,
        max_ph=max_ph,
        max_variants=max_variants,
        label_states=label_states,
        pka_precision=pka_precision
    )
    return dimorphite_dl.protonate(mol)
# def a function to generate all protonation states & tautomers
def generate_protonation_tautomers(mol_or_smi:Union[Chem.Mol, str]):
    if mol_or_smi.__class__ == str:
        mol = Chem.MolFromSmiles(mol_or_smi)
    else:
        mol = mol_or_smi
    all_tautomers = []
    standardize_smiles = standardize_smi(Chem.MolToSmiles(mol),isomeric=True)
    # standardize_smiles = Chem.AddHs(Chem.MolFromSmiles(standardize_smiles))
    protonation_states = protonate(standardize_smiles)
    
    for protonation_state in protonation_states:
        # print(protonation_state)
        protonation_state = Chem.AddHs(Chem.MolFromSmiles(protonation_state))
        tautomers = GetNumTautomers(protonation_state)
        all_tautomers += tautomers
    return all_tautomers
if __name__ == '__main__':

    ligand = '/home/username/SurfDock/data/Screen_sample_dirs/test_samples/1a0q/1a0q_ligand_for_Screen.sdf'
    out_file = '/home/username/SurfDock/data/Screen_sample_dirs/test_samples/1a0q/1a0q_ligand_for_Screen_protonate_tautomer.sdf'
    mols = Chem.SDMolSupplier(ligand,removeHs = False)
    writer = Chem.SDWriter(out_file)
    for mol_idx,mol in enumerate(mols):
        tautomers = generate_protonation_tautomers(mol)
        print('mol_idx:',mol_idx,'tautomers:',len(tautomers))
        for idx, tau in enumerate(tautomers):
            tau.SetProp('_Name', f'{mol.GetProp("_Name")}_mol_idx_{mol_idx}_tautomers_{idx}')
            writer.write(tau)
    writer.close()