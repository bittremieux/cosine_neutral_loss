import copy

import numpy as np
import spectrum_utils.spectrum as sus


def spec_to_neutral_loss(spectrum: sus.MsmsSpectrum) -> sus.MsmsSpectrum:
    """
    Convert a spectrum to a neutral loss spectrum by subtracting the peak m/z
    values from the precursor m/z.

    Parameters
    ----------
    spectrum : sus.MsmsSpectrum
        The spectrum to be converted to its neutral loss spectrum.

    Returns
    -------
    sus.MsmsSpectrum
        The converted neutral loss spectrum.
    """
    # Add ghost peak at 0 m/z to anchor the m/z range after transformation.
    spectrum = copy.copy(spectrum)
    spectrum._inner._mz = np.insert(spectrum.mz, 0, [0])
    spectrum._inner._intensity = np.insert(spectrum.intensity, 0, [0])
    # Restrict to the precursor m/z to avoid getting negative peaks in the
    # neutral loss spectrum.
    spectrum = spectrum.set_mz_range(0, spectrum.precursor_mz)
    # Create neutral loss peaks and make sure the peaks are in ascending m/z
    # order.
    spectrum._inner._mz = (spectrum.precursor_mz - spectrum.mz)[::-1]
    spectrum._inner._intensity = spectrum.intensity[::-1]
    return spectrum
