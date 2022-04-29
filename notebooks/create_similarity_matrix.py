import os
import sys
from pathlib import Path

# Make sure all code is in the PATH.
sys.path.append("../src/")


# from multiprocessing import Pool, freeze_support
# from multiprocessing import freeze_support
from pathos.multiprocessing import ProcessingPool as Pool
import pathos.multiprocessing
import numba as nb
import numpy as np
import pandas as pd
import pyteomics.mgf
from tqdm import tqdm

import spectrum_utils.spectrum as sus
import similarity
import bile_mods
import structure_similarity as struc_sim

# activate pandas progress_apply
tqdm.pandas()

# public parameters
library_file = "../data/BILELIB19.mgf"
# library_file = "../data/20220418_ALL_GNPS_NO_PROPOGATED.mgf"

replace_old_files = True

# analysis name
# square root transformation of intensities is often performed to limit the impact of high abundant signals
# size of subset of spectral pairs
# minimum number of signals only removes the spectra with less
# signal alignment tolerance
analysis_name = "all"
apply_sqrt = False
n_spectral_pairs = 500000
min_n_signals = 6
abs_mz_tolerance = 0.02

# structure can be parsed by RDKit
require_structure = True

# only allow precursor mz difference of:
max_mz_delta = 200
min_mz_delta = 4.0

# if defined, we will only search for specific delta m/z between two spectra
# 16 oxygen 15.994914
# otherwise define as -1
# specific_mod_mz = 15.9949
# specific_mod_mz = 180.063388-18.010564  # hexose
# specific_mod_mz = 14.01566
# specific_mod_mz = 18.010564
# specific_mod_mz = 42.01056 # acetic acid
specific_mod_mz = -1

# free bile acids compared to conjugated with amino acids
# or different delta mz between different conjugated bile acids
mod_list = np.empty(0, float)
# mod_list = bile_mods.get_as_mods()
# mod_list = bile_mods.get_as_exchange()

library_file_name_without_ext = Path(library_file).stem
# analysis ID is used for file export
if specific_mod_mz <= 0:
    analysis_id = "{}_{}_sqrt_{}_{}pairs_{}min_signals_{}_requirestruc_{}-{}deltamz_{}mods".format(
        library_file_name_without_ext,
        analysis_name,
        apply_sqrt,
        n_spectral_pairs,
        min_n_signals,
        require_structure,
        min_mz_delta,
        max_mz_delta,
        len(mod_list),
    ).replace(".", "i")
else:
    analysis_id = "{}_{}_sqrt_{}_{}pairs_{}min_signals_{}_requirestruc_{}specific_delta_{}mods".format(
        library_file_name_without_ext,
        analysis_name,
        apply_sqrt,
        n_spectral_pairs,
        min_n_signals,
        require_structure,
        specific_mod_mz,
        len(mod_list),
    ).replace(".", "i")

results_file = "results/{}.parquet".format(analysis_id)

# output filename of pairs only with pair selection relevant parameters:
pairs_filename = (
    "temp/pairs_{}.parquet".format(analysis_id)
    .replace("sqrt_True_", "")
    .replace("sqrt_False_", "")
)
spectra_filename = "tempspectra/spectra_{}_{}min_signals_sqrt_{}_{}_requirestruc.parquet".format(
    library_file_name_without_ext, min_n_signals, apply_sqrt, require_structure
)

spectra_id_tsv_filename = "results/spectra_ids_{}_{}min_signals_{}_requirestruc.tsv".format(
    library_file_name_without_ext, min_n_signals, require_structure
)


def main():
    if replace_old_files:
        Path(results_file).unlink(missing_ok=True)
        Path(spectra_id_tsv_filename).unlink(missing_ok=True)
        Path(spectra_filename).unlink(missing_ok=True)
        Path(pairs_filename).unlink(missing_ok=True)

    spectra = import_from_mgf()

    # compute subset of pairs
    pairs_df = load_or_compute_pairs_df(spectra)
    print("Comparing {} pairs".format(len(pairs_df)))

    # can run on single thread or parallel
    # similarities = compute_similarity_parallel(pairs_df)
    print('computing similarities')
    similarities = compute_similarity(spectra, pairs_df)
    save_results(similarities)


# check for profile spectra in the library file - contains zero intensity values
def is_centroid(spectrum_dict):
    return all(i > 0 for i in spectrum_dict["intensity array"])


def import_from_mgf():
    try:
        # read filtered spectra from file
        spec_df = pd.read_parquet(spectra_filename)
        spectra = []
        for index, spec in spec_df.iterrows():
            spectra.append(
                sus.MsmsSpectrum(
                    spec["id"],
                    spec["mz"],
                    spec["charge"],
                    spec["mzs"],
                    spec["intensities"]
                )
            )
        print("spectra read from parquet file")
        return spectra
    except:
        print("importing data from{}".format(library_file))
        # Read all spectra from the ALL_GNPS.MGF from April 13. 2022
        # library contains propagated spectra, that will be filtered out
        # GNPS library from https://gnps-external.ucsd.edu/gnpslibrary
        c_error = 0
        c_propagated = 0
        c_multi_charged = 0
        c_removed = 0
        c_profile_spec = 0
        c_below_n_signals = 0
        c_polarity = 0
        c_not_protonated = 0
        c_no_structure = 0
        spectra = []
        smiles = []
        inchi = []
        inchikey = []
        mol_structures = []
        with pyteomics.mgf.MGF(library_file) as f_in:
            for spectrum_dict in tqdm(f_in):
                # ignore:
                #   - propagated spectra with LIBRARYQUALITY==4
                #   - multiple charged molecules
                #   - < minimum signals
                #   - not positive mode
                #   - not M+H
                #   - not centroid (contains zero intensity values)
                try:
                    if int(spectrum_dict["params"]["libraryquality"]) > 3:
                        c_propagated += 1
                    elif int(spectrum_dict["params"]["charge"][0]) != 1:
                        c_multi_charged += 1
                    elif len(spectrum_dict["m/z array"]) < min_n_signals:
                        c_below_n_signals += 1
                    elif str(spectrum_dict["params"]["ionmode"]) != "Positive":
                        c_polarity += 1
                    elif (
                        not str(spectrum_dict["params"]["name"])
                        .rstrip()
                        .endswith(" M+H")
                    ):
                        c_not_protonated += 1
                    elif not is_centroid(spectrum_dict):
                        c_profile_spec += 1
                    else:
                        inchiaux_ = spectrum_dict["params"]["inchiaux"]
                        smiles_ = spectrum_dict["params"]["smiles"]
                        inchi_ = spectrum_dict["params"]["inchi"]
                        mol = struc_sim.get_mol_struc(smiles_, inchi_)
                        if require_structure and not mol:
                            c_no_structure += 1
                        else:
                            intensities = spectrum_dict["intensity array"]
                            if apply_sqrt:
                                intensities = np.sqrt(intensities)

                            inchikey.append(inchiaux_)
                            smiles.append(smiles_)
                            inchi.append(inchi_)
                            mol_structures.append(mol)
                            spec = sus.MsmsSpectrum(
                                spectrum_dict["params"]["spectrumid"],
                                float(spectrum_dict["params"]["pepmass"][0]),
                                int(spectrum_dict["params"]["charge"][0]),
                                spectrum_dict["m/z array"],
                                intensities
                            )
                            # remove residual precursor signals with potential isotope pattern
                            spec.remove_precursor_peak(4, "Da")
                            spectra.append(spec)
                except:
                    c_error += 1

        c_removed = (
            c_propagated
            + c_error
            + c_no_structure
            + c_multi_charged
            + c_profile_spec
            + c_below_n_signals
            + c_polarity
            + c_not_protonated
        )
        print(
            "total spectra={};  total removed={};  few signals={};  error={}; no structure (only if active)={};  polarity mismatch={};  multi charge={};  propagated spec={};  not M+H={};  profile spec={}".format(
                len(spectra),
                c_removed,
                c_below_n_signals,
                c_error,
                c_no_structure,
                c_polarity,
                c_multi_charged,
                c_propagated,
                c_not_protonated,
                c_profile_spec,
            )
        )

        # save to csv ID and structures before sorting
        print("saving tsv file with spectrum ids")
        id_df = pd.DataFrame()
        id_df["id"] = [s.identifier for s in spectra]
        id_df["mz"] = [s.precursor_mz for s in spectra]
        id_df["smiles"] = smiles
        id_df["inchi"] = inchi
        id_df["inchi_key"] = inchikey
        Path("results/").mkdir(parents=True, exist_ok=True)
        id_df.to_csv(spectra_id_tsv_filename, sep="\t")

        # sort spectra by precursor mz
        spectra.sort(key=lambda spec: spec.precursor_mz)

        # save for faster load
        Path("tempspectra/").mkdir(parents=True, exist_ok=True)
        spec_df = pd.DataFrame({"spectra": spectra})
        spec_df["id"] = spec_df["spectra"].apply(lambda s: s.identifier)
        spec_df["mz"] = spec_df["spectra"].apply(lambda s: s.precursor_mz)
        spec_df["charge"] = spec_df["spectra"].apply(lambda s: s.precursor_charge)
        spec_df["mzs"] = spec_df["spectra"].apply(lambda s: s.mz)
        spec_df["intensities"] = spec_df["spectra"].apply(lambda s: s.intensity)
        spec_df = spec_df.drop("spectra", axis=1)
        spec_df.to_parquet(spectra_filename)

        return spectra


@nb.njit
def generate_pairs(precursor_mz):
    """
    Create pairs of spectra that are compared.
    Maximum precursor mz difference is 200 to limit the search space to a range from smaller modifications (e.g., +O) and
    larger modifications (e.g., hexose)

    Parameters
    ----------
    spectra a list of all spectra

    Returns yields two indices
    -------

    """
    check_mods = len(mod_list) > 0
    for i in range(len(precursor_mz)):
        j = i + 1
        while (j < len(precursor_mz)) and (
            precursor_mz[j] <= precursor_mz[i] + max_mz_delta
        ):
            delta = precursor_mz[j] - precursor_mz[i]
            if delta >= min_mz_delta:
                # list is sorted by precursor mz so j always > i
                # first check list of modifications, then specific
                # select only one specific precursor mz differance or include all
                if check_mods:
                    for mod_mz in mod_list:
                        if abs(delta - mod_mz) <= abs_mz_tolerance:
                            yield i
                            yield j
                            break

                elif (specific_mod_mz < 0) or (
                    abs(delta - specific_mod_mz) <= abs_mz_tolerance
                ):
                    yield i
                    yield j
            j += 1

def calc_tanimoto_row(pair, spectra, id_mol_dict):
    # spectrum.identifier is the library ID
    a = spectra[pair["index1"]].identifier
    b = spectra[pair["index2"]].identifier
    mola = id_mol_dict.get(a)
    molb = id_mol_dict.get(b)
    if mola and molb:
        return struc_sim.calc_tanimoto(mola, molb)
    return None

def calc_tanimoto(spectra, pairs_df):
    # import ID to smiles / inchi df
    print("calculating structure tanimoto score")
    id_struc_df = pd.read_csv(spectra_id_tsv_filename, sep="\t")
    id_struc_df["mol"] = id_struc_df.apply(lambda row: struc_sim.get_mol_struc(row["smiles"], row["inchi"]), axis=1)

    id_mol_dict = pd.Series(id_struc_df.mol.values,index=id_struc_df.id).to_dict()

    pairs_df["tanimoto"] = pairs_df.progress_apply(lambda row: calc_tanimoto_row(row, spectra, id_mol_dict), axis=1)
    return pairs_df


def load_or_compute_pairs_df(spectra):
    # try to load precomputed pairs
    try:
        pairs_df = pd.read_parquet(pairs_filename)
        print('pairs loaded from parquet')
        return pairs_df
    except:
        print('computing pairs')
        # Extract precursor mz as filter argument
        precursor_mz_list = nb.typed.List()
        for spectrum in spectra:
            precursor_mz_list.append(spectrum.precursor_mz)

        # create pairs and randomly subset
        rng = np.random.default_rng(2022)
        pairs = np.fromiter(generate_pairs(precursor_mz_list), np.uint32).reshape(
            (-1, 2)
        )
        pairs = rng.choice(pairs, min(len(pairs), n_spectral_pairs), replace=False)

        print('saving pairs to file for fast reload')
        # save pairs to speed up reanalysis
        pairs_df = pd.DataFrame(pairs, columns=["index1", "index2"])
        pairs_df = calc_tanimoto(spectra, pairs_df)
        Path("temp/").mkdir(parents=True, exist_ok=True)
        pairs_df.to_parquet(pairs_filename)
        return pairs_df


def load_spectra_compute_similarity(pairs_df):
    spectra = import_from_mgf()
    return compute_similarity(spectra, pairs_df)


def compute_similarity(spectra, pairs_df):
    # common columns
    ids_a, ids_b, delta_mz = [], [], []
    # lists of SimilarityTuples
    cosines, modified_cosines, neutral_losses = [], [], []
    for i, j in tqdm(zip(pairs_df["index1"], pairs_df["index2"])):
        # calculate scores and add to lists
        cos = similarity.cosine(spectra[i], spectra[j], abs_mz_tolerance)
        mod_cos = similarity.modified_cosine(spectra[i], spectra[j], abs_mz_tolerance)
        nl = similarity.neutral_loss(spectra[i], spectra[j], abs_mz_tolerance)
        if (nl.score > 0) and (cos.score > 0) and (mod_cos.score > 0):
            cosines.append(cos)
            modified_cosines.append(mod_cos)
            neutral_losses.append(nl)
            ids_a.append(spectra[i].identifier)
            ids_b.append(spectra[j].identifier)
            delta_mz.append(abs(spectra[i].precursor_mz - spectra[j].precursor_mz))

    similarities = pd.DataFrame({"id1": ids_a, "id2": ids_b, "delta_mz": delta_mz})
    similarities["tanimoto"] = pairs_df["tanimoto"]

    tmp = pd.DataFrame(cosines)
    tmp.drop(["matched_indices", "matched_indices_other"], axis=1, inplace=True)
    tmp = tmp.add_prefix("cos_")
    similarities = similarities.join(tmp)

    tmp = pd.DataFrame(modified_cosines)
    tmp.drop(["matched_indices", "matched_indices_other"], axis=1, inplace=True)
    tmp = tmp.add_prefix("mod_")
    similarities = similarities.join(tmp)

    tmp = pd.DataFrame(neutral_losses)
    tmp.drop(["matched_indices", "matched_indices_other"], axis=1, inplace=True)
    tmp = tmp.add_prefix("nl_")
    similarities = similarities.join(tmp)
    print(len(similarities))
    return similarities


def compute_similarity_parallel(pairs_df, num_of_processes=-1):
    if num_of_processes < 0:
        num_of_processes = pathos.multiprocessing.cpu_count()
    with Pool(num_of_processes) as pool:
        data_split = np.array_split(pairs_df, num_of_processes)
        similarities = pd.concat(
            pool.uimap(load_spectra_compute_similarity, data_split)
        )
        return similarities


def save_results(similarities):
    Path("results/").mkdir(parents=True, exist_ok=True)
    similarities.to_parquet(results_file)
    similarities.head(5)
    print("Results saved to {}".format(results_file))


if __name__ == "__main__":
    # freeze_support()
    main()
