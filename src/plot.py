import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus

import similarity
import utils


def plot_cosine(
        spectrum1: sus.MsmsSpectrum,
        spectrum2: sus.MsmsSpectrum,
        filename: str,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    sup.mirror(spectrum1, spectrum2, ax=ax)

    sim = similarity.modified_cosine(spectrum1, spectrum2, 0.02)
    ax.set_title(f"Modified cosine similarity = {sim[0]:.4f}")

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_neutral_loss(
        spectrum1: sus.MsmsSpectrum,
        spectrum2: sus.MsmsSpectrum,
        filename: str,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    sup.mirror(
        utils.spec_to_neutral_loss(spectrum1),
        utils.spec_to_neutral_loss(spectrum2),
        ax=ax
    )

    sim = similarity.neutral_loss(spectrum1, spectrum2, 0.02)
    ax.set_title(f"Neutral loss similarity = {sim[0]:.4f}")

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
