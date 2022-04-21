import os
import pathos.multiprocessing
import sys
from pathlib import Path

# Make sure all code is in the PATH.
sys.path.append("../src/")

import numba as nb
import numpy as np
import pandas as pd
import pyteomics.mgf
import tqdm
import spectrum_utils.spectrum as sus
import similarity
# from multiprocessing import Pool, freeze_support
# from multiprocessing import freeze_support
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool

# public parameters
library_file = "../data/BILELIB19.mgf"
# library_file = "../data/20220418_ALL_GNPS_NO_PROPOGATED.mgf"

# analysis name
analysis_name = "test"

# square root transformation of intensities is often performed to limit the impact of high abundant signals
apply_sqrt = False

# size of subset of spectral pairs
n_spectral_pairs = 1000

# minimum number of signals only removes the spectra with less
min_n_signals = 6

# signal alignment tolerance
abs_mz_tolerance = 0.02
# only allow precursor mz difference of:
max_mz_delta = 200
# if defined, we will only search for specific delta m/z between two spectra
# 16 oxygen 15.994914
# otherwise define as -1
specific_mod_mz = 15.9949
# specific_mod_mz = -1

# analysis ID is used for file export
if specific_mod_mz <= 0:
    analysis_id = "{}_sqrt_{}_{}pairs_{}min_signals_{}maxdelta" \
        .format(analysis_name, apply_sqrt, n_spectral_pairs, min_n_signals, max_mz_delta).replace(".", "i")
else:
    analysis_id = "{}_sqrt_{}_{}pairs_{}min_signals_{}specific_delta" \
        .format(analysis_name, apply_sqrt, n_spectral_pairs, min_n_signals, specific_mod_mz).replace(".", "i")

library_file_name_without_ext = Path(library_file).stem

# output filename of pairs only with pair selection relevant parameters:
pairs_filename = "temp/{}_pairs.parquet".format(analysis_id).replace("sqrt_True_", "").replace("sqrt_False_", "")
spectra_filename = "tempspectra/spectra_{}_{}min_signals_sqrt_{}.parquet".format(library_file_name_without_ext, min_n_signals, apply_sqrt)

def main():
    precursor_mz_list = None
    if (os.path.isfile(pairs_filename) == False) or (os.path.isfile(spectra_filename) == False):
        # missing either the spectra or pairs file
        spectra = import_from_mgf()

        # Extract precursor mz as filter argument
        precursor_mz_list = nb.typed.List()
        for spectrum in spectra:
            precursor_mz_list.append(spectrum.precursor_mz)

    # compute subset of pairs
    pairs_df = load_or_compute_pairs_df(precursor_mz_list)
    print ("Comparing {} pairs".format(len(pairs_df)))

    similarities = compute_similarity_parallel(pairs_df)
    save_results(similarities)


# check for profile spectra in the library file - contains zero intensity values
def is_centroid(spectrum_dict):
    return all(i > 0 for i in spectrum_dict["intensity array"])


def import_from_mgf():
    try:
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
                    # IONMODE=Positive
                    # LIBRARYQUALITY=4
                    # SPECTRUMID
                    # NAME
                    # SMILES
                ))
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
        spectra = []
        with pyteomics.mgf.MGF(library_file) as f_in:
            for spectrum_dict in tqdm.tqdm(f_in):
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
                    elif not str(spectrum_dict["params"]["name"]).rstrip().endswith(" M+H"):
                        c_not_protonated += 1
                    elif not is_centroid(spectrum_dict):
                        c_profile_spec += 1
                    else:
                        intensities = spectrum_dict["intensity array"]
                        if apply_sqrt:
                            intensities = np.sqrt(intensities)
                        spectra.append(
                            sus.MsmsSpectrum(
                                spectrum_dict["params"]["spectrumid"],
                                float(spectrum_dict["params"]["pepmass"][0]),
                                int(spectrum_dict["params"]["charge"][0]),
                                spectrum_dict["m/z array"],
                                intensities,
                                # IONMODE=Positive
                                # LIBRARYQUALITY=4
                                # SPECTRUMID
                                # NAME
                                # SMILES
                            )
                        )
                except:
                    c_error += 1

        c_removed = c_propagated + c_error + c_multi_charged + c_profile_spec + c_below_n_signals + c_polarity + c_not_protonated
        print(
            "total spectra={};  total removed={};  few signals={};  error={};  polarity mismatch={};  multi charge={};  propagated spec={};  not M+H={};  profile spec={}".format(
                len(spectra), c_removed, c_below_n_signals, c_error, c_polarity, c_multi_charged, c_propagated,
                c_not_protonated, c_profile_spec))

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
        spec_df = spec_df.drop("spectra", 1)
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
    for i in range(len(precursor_mz)):
        j = i + 1
        while (j < len(precursor_mz)) and (precursor_mz[j] <= precursor_mz[i] + max_mz_delta):
            delta = precursor_mz[j] - precursor_mz[i]
            if delta > 1:
                # list is sorted by precursor mz so j always > i
                # select only one specific precursor mz differance or include all
                if (specific_mod_mz < 0) or (abs(delta - specific_mod_mz) <= abs_mz_tolerance):
                    yield i
                    yield j
            j += 1


def load_or_compute_pairs_df(precursor_mz_list=None):
    # try to load precomputed pairs
    try:
        pairs_df = pd.read_parquet(pairs_filename)
        return pairs_df
    except:
        if (precursor_mz_list is None) or (len(precursor_mz_list) <= 0):
            print("precursor list was None but file read was not successful")
            exit(1)
        # create pairs and randomly subset
        rng = np.random.default_rng(2022)
        pairs = np.fromiter(
            generate_pairs(precursor_mz_list),
            np.uint32).reshape((-1, 2))
        pairs = rng.choice(pairs, min(len(pairs), n_spectral_pairs), replace=False)

        pairs_df = pd.DataFrame(pairs, columns=['index1', 'index2'])
        # save pairs to speed up reanalysis
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
    for i, j in tqdm.tqdm(zip(pairs_df["index1"], pairs_df["index2"])):
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

    similarities = pd.DataFrame(
        {
            "id1": ids_a,
            "id2": ids_b,
            "delta_mz": delta_mz
        }
    )

    tmp = pd.DataFrame(cosines)
    tmp = tmp.add_prefix('cos_')
    similarities = similarities.join(tmp)

    tmp = pd.DataFrame(modified_cosines)
    tmp = tmp.add_prefix('mod_')
    similarities = similarities.join(tmp)

    tmp = pd.DataFrame(neutral_losses)
    tmp = tmp.add_prefix('nl_')
    similarities = similarities.join(tmp)
    print(len(similarities))
    return similarities


def compute_similarity_parallel(pairs_df, num_of_processes=-1):
    if num_of_processes < 0:
        num_of_processes = pathos.multiprocessing.cpu_count()
    with Pool(num_of_processes) as pool:
        data_split = np.array_split(pairs_df, num_of_processes)
        similarities = pd.concat(pool.uimap(load_spectra_compute_similarity, data_split))
        return similarities


def save_results(similarities):
    fname = "results/{}.parquet".format(analysis_id)
    Path("results/").mkdir(parents=True, exist_ok=True)
    similarities.to_parquet(fname)
    similarities.head(5)
    print("Results saved to {}".format(fname))


if __name__ == "__main__":
    # freeze_support()
    main()
