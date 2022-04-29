from functools import lru_cache

import pandas as pd
from rdkit import Chem, DataStructs


def main():
    # scores_df = pd.read_csv("corr/small_6s_filterPrec_sqrt.csv")
    # scores_df = pd.read_csv("corr/BILE_6s_filterPrec_sqrt.csv")
    # nodes = pd.read_csv("corr/BILE_6s_filterPrec_sqrt_nodes.csv")

    scores_df = pd.read_parquet("corr/")
    nodes = pd.read_csv("corr/BILE_6s_filterPrec_sqrt_nodes.csv")
    print(scores_df.head(5))

    # map fingerprint to ID
    nodes["fp"] = [calc_fingerprint(smi) for smi in nodes["smiles"]]
    smi_to_fp_dict = pd.Series(nodes["fp"].values, index=nodes["id"]).to_dict()

    scores_df["tanimoto"] = [
        calc_tanimoto(smi_to_fp_dict, a, b)
        for a, b in zip(scores_df["id1"], scores_df["id2"])
    ]
    # scores_df["tanimoto"] = [compute_tanimoto(fp_map, a,b) for a,b in zip(scores_df['SMILES_a'], scores_df[
    #     'SMILES_b'])]

    print("scores done now exporting")

    scores_df.to_csv("corr/bile_full.csv")

    exit(0)
    # for index, row in scores_df.iterrows():
    #     a = row["SMILES_a"]
    #     b = row["SMILES_b"]


@lru_cache(maxsize=None)
def get_mol_struc(smiles: str, inchi: str):
    """

    Parameters
    ----------
    smiles smiles is tried first
    inchi inchi is tried second

    Returns rdkit mol object
    -------

    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
    except:
        pass
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            return mol
    except:
        pass
    return None


@lru_cache(maxsize=None)
def calc_fingerprint(mol):
    # alternative fingerprint
    # return AllChem.GetMorganFingerprintAsBitVect(to_mol(a), 3, nBits=1024)
    try:
        return Chem.RDKFingerprint(mol)
    except:
        return None


def get_fingerprint(fp_map, a):
    fp = fp_map.get(a)
    if fp is None:
        fp = calc_fingerprint(a)
        fp_map[a] = fp
    return fp


@lru_cache(maxsize=None)
def calc_tanimoto(mola, molb) -> float:
    try:
        fpa = calc_fingerprint(mola)
        fpb = calc_fingerprint(molb)
        return round(DataStructs.TanimotoSimilarity(fpa, fpb), 4)
    # DataStructs.FingerprintSimilarity(fps[0],fps[1], metric=DataStructs.DiceSimilarity)
    except:
        return None


def to_mol(smi: str) -> Chem.rdchem.Mol:
    return Chem.MolFromSmiles(smi)


def to_canon_smiles(smiles: str) -> str:
    try:
        return Chem.CanonSmiles(smiles)
    except:
        print("Invalid SMILES:", smiles)
        return None


if __name__ == "__main__":
    main()
