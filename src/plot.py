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


def plot_cosine(
    spectrum1: sus.MsmsSpectrum,
    spectrum2: sus.MsmsSpectrum,
    filename: str,
    modified: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 4))

    if not modified:
        sim = similarity.cosine(spectrum1, spectrum2, 0.05)
        ax.set_title(f"Cosine similarity = {sim[0]:.4f}")
    else:
        sim = similarity.modified_cosine(spectrum1, spectrum2, 0.05)
        ax.set_title(f"Modified cosine similarity = {sim[0]:.4f}")

    _annotate_matching_peaks(spectrum1, spectrum2, sim[1])

    sup.mirror(spectrum1, spectrum2, ax=ax)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_neutral_loss(
    spectrum1: sus.MsmsSpectrum,
    spectrum2: sus.MsmsSpectrum,
    filename: str,
):
    fig, ax = plt.subplots(figsize=(8, 4))

    sim = similarity.neutral_loss(spectrum1, spectrum2, 0.05)
    ax.set_title(f"Neutral loss similarity = {sim[0]:.4f}")

    spectrum1 = utils.spec_to_neutral_loss(spectrum1)
    spectrum2 = utils.spec_to_neutral_loss(spectrum2)

    _annotate_matching_peaks(spectrum1, spectrum2, sim[1])

    sup.mirror(spectrum1, spectrum2, ax=ax)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def _annotate_matching_peaks(
    spectrum1: sus.MsmsSpectrum,
    spectrum2: sus.MsmsSpectrum,
    peak_matches: List[Tuple[int, int]],
) -> None:
    """
    Somewhat hacky way to get spectrum_utils to annotate matching peaks.

    Parameters
    ----------
    spectrum1 : sus.MsmsSpectrum
        The first spectrum.
    spectrum2 : sus.MsmsSpectrum
        The second spectrum.
    peak_matches : List[Tuple[int, int]
        Tuples with matching peak indexes between both spectra.
    """
    spectrum1._annotation = np.full_like(spectrum1.mz, None, object)
    spectrum2._annotation = np.full_like(spectrum2.mz, None, object)
    for match1, match2 in peak_matches:
        spectrum1._annotation[match1] = FragmentAnnotation(ion_type="b")
        spectrum2._annotation[match2] = FragmentAnnotation(ion_type="y")
