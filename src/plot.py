import collections
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus

import similarity
import utils


FragmentAnnotation = collections.namedtuple("FragmentAnnotation", ["ion_type"])
FragmentAnnotation.__str__ = lambda _: ""


def plot_mirror(
    spectrum1: sus.MsmsSpectrum,
    spectrum2: sus.MsmsSpectrum,
    score: str,
    filename: str,
) -> None:
    """
    Plot mirror spectra showing peak matches.

    Parameters
    ----------
    spectrum1 : sus.MsmsSpectrum
        The first spectrum.
    spectrum2 : sus.MsmsSpectrum
        The second spectrum.
    score : str
        The similarity score used. Valid values are "cosine",
        "modified_cosine", and "neutral_loss".
    filename : str
        Filename to save the figure.
    """
    fragment_mz_tol = 0.05

    fig, ax = plt.subplots(figsize=(8, 4))

    if score == "cosine":
        sim = similarity.cosine(spectrum1, spectrum2, fragment_mz_tol)
        title = f"Cosine similarity = {sim[0]:.4f}"
    elif score == "modified_cosine":
        sim = similarity.modified_cosine(spectrum1, spectrum2, fragment_mz_tol)
        title = f"Modified cosine similarity = {sim[0]:.4f}"
    elif score == "neutral_loss":
        sim = similarity.neutral_loss(spectrum1, spectrum2, fragment_mz_tol)
        title = f"Neutral loss similarity = {sim[0]:.4f}"
        spectrum1 = utils.spec_to_neutral_loss(spectrum1)
        spectrum2 = utils.spec_to_neutral_loss(spectrum2)
    else:
        raise ValueError("Unknown score specified")

    _annotate_matching_peaks(
        spectrum1,
        spectrum2,
        sim.matched_indices,
        sim.matched_indices_other,
    )

    sup.mirror(spectrum1, spectrum2, ax=ax)
    ax.set_title(title)

    # Neutral loss can lead to negative m/z values.
    min_mz = min(spectrum1.mz[0], spectrum2.mz[0])
    max_mz = max(spectrum1.mz[-1], spectrum2.mz[-1])
    ax.set_xlim(min_mz - max(min_mz / 10, 50), max_mz + max(max_mz / 10, 50))

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def _annotate_matching_peaks(
    spectrum1: sus.MsmsSpectrum,
    spectrum2: sus.MsmsSpectrum,
    peak_matches1: np.ndarray,
    peak_matches2: np.ndarray,
) -> None:
    """
    Somewhat hacky way to get spectrum_utils to annotate matching peaks.

    Parameters
    ----------
    spectrum1 : sus.MsmsSpectrum
        The first spectrum.
    spectrum2 : sus.MsmsSpectrum
        The second spectrum.
    peak_matches1 : np.ndarray
        Matching peak indexes in the first spectrum.
    peak_matches2 : np.ndarray
        Matching peak indexes in the second spectrum.
    """
    spectrum1._annotation = np.full_like(spectrum1.mz, None, object)
    spectrum2._annotation = np.full_like(spectrum2.mz, None, object)
    for match1, match2 in zip(peak_matches1, peak_matches2):
        spectrum1._annotation[match1] = FragmentAnnotation(ion_type="b")
        spectrum2._annotation[match2] = FragmentAnnotation(ion_type="y")
