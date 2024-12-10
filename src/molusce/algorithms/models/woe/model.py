import math
from collections import namedtuple

import numpy as np
from numpy import ma as ma
from qgis.PyQt.QtCore import QCoreApplication

from molusce.algorithms.utils import binaryzation, get_gradations

EPSILON = 4 * np.finfo(float).eps  # Small number > 0

Weights = namedtuple("Weights", ["wPlus", "wMinus"])


class WoeError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, msg):
        self.msg = msg


def _binary_woe(factor, sites, unitcell=1):
    """Weight of evidence method (binary form).

    @param factor     Binary pattern raster used for prediction of point objects (sites).
    @param sites      Raster layer consisting of the locations at which the point objects are known to occur.

    @return (W+, W-)  Tuple of the factor's weights (w+, w-).
    """
    # Check rasters type
    if factor.dtype != bool:
        raise WoeError(
            QCoreApplication.translate(
                "WeightOfEvidenceWidget",
                "Factor raster must be binary in this mode of the method!",
            )
        )
    if sites.dtype != bool:
        raise WoeError(
            QCoreApplication.translate(
                "WeightOfEvidenceWidget",
                "Site raster must be binary in this mode of the method!",
            )
        )
    # Check rasters dimentions
    if factor.shape != sites.shape:
        raise WoeError(
            QCoreApplication.translate(
                "WeightOfEvidenceWidget",
                "Factor and sites rasters have different shapes!",
            )
        )
    # Check masked areas of sites and factors are the same
    if (
        factor.mask.shape != () and sites.mask.shape != ()
    ) and not np.array_equal(
        factor.mask, sites.mask
    ):  # if mask = False ,then mask.shape==()
        raise WoeError(
            QCoreApplication.translate(
                "WeightOfEvidenceWidget",
                "Masked areas of factor and sites rasters are different!",
            )
        )

    fm = factor.compressed()  # masked factor
    sm = sites.compressed()  # masked sites

    A = 1.0 * len(fm) / unitcell  # Total map area in unit cells
    B = (
        1.0 * len(fm[fm == True]) / unitcell  # noqa: E712
    )  # Total factor area in unit cells
    N = 1.0 * len(sm[sm == True])  # Count of sites  # noqa: E712

    # Count of sites inside area where the factor occurs:
    siteAndPatten = fm & sm  # Sites inside area where the factor occurs
    Nb = 1.0 * len(
        siteAndPatten[siteAndPatten == True]  # noqa: E712
    )  # Count of sites inside factor area

    # Check areas size
    if A == 0:
        raise WoeError(
            QCoreApplication.translate(
                "WeightOfEvidenceWidget", "Unmasked area is zero-size!"
            )
        )
    if B == 0:
        raise WoeError(
            QCoreApplication.translate(
                "WeightOfEvidenceWidget",
                "Unmasked area of factor (pattern) is zero-size!",
            )
        )
    if N == 0:
        raise WoeError(
            QCoreApplication.translate(
                "WeightOfEvidenceWidget",
                "Unmasked area of sites is zero-size!",
            )
        )
    if (Nb > N) or (N >= A):
        raise WoeError(
            QCoreApplication.translate(
                "WeightOfEvidenceWidget",
                "Unit cell size is too big for your data!",
            )
        )

    pSiteFactor = Nb / N
    pNonSiteFactor = (B - Nb) / (A - N)
    pSiteNonFactor = (N - Nb) / N
    pNonSiteNonFactor = (A - B - N + Nb) / (A - N)

    # Add a small number to prevent devision by zero or log(0):
    pSiteFactor = pSiteFactor + EPSILON
    pNonSiteFactor = pNonSiteFactor + EPSILON
    pSiteNonFactor = pSiteNonFactor + EPSILON
    pNonSiteNonFactor = pNonSiteNonFactor + EPSILON

    # Weights
    wPlus = math.log(pSiteFactor / pNonSiteFactor)
    wMinus = math.log(pSiteNonFactor / pNonSiteNonFactor)

    return Weights(wPlus, wMinus)


def woe(factor, sites, unit_cell=1):
    """Weight of evidence method (multiclass form).

    @param factor     Multiclass pattern array used for prediction of point objects (sites).
    @param sites      Array layer consisting of the locations at which the point objects are known to occur.
    @param unit_cell  Method parameter, pixelsize of resampled rasters.

    @return masked array  Array of total weights of each factor.
    """
    # Get list of categories from the factor raster
    categories = get_gradations(factor.compressed())

    # Try to binarize sites:
    sCategories = get_gradations(sites.compressed())
    if len(sCategories) != 2:
        raise WoeError(
            QCoreApplication.translate(
                "WeightOfEvidenceWidget", "Site raster must be binary!"
            )
        )
    sites = binaryzation(sites, [sCategories[1]])

    # List of the weights of evidence:
    # weights[0] is (wPlus, wMinus) for the first category, weights[1] is (wPlus, wMinus) for the second category, ...
    weights = []
    if len(categories) >= 2:
        for cat in categories:
            fct = binaryzation(factor, [cat])
            weights.append(_binary_woe(fct, sites, unit_cell))
    else:
        raise WoeError(
            QCoreApplication.translate(
                "WeightOfEvidenceWidget",
                "Wrong count of categories in the factor raster!",
            )
        )

    wTotalMin = sum([w[1] for w in weights])
    # List of total weights of evidence of the categories:
    # wMap[0] is the total weight of the first category, wMap[1] is the total weight of the second category, ...
    wMap = [w[0] + wTotalMin - w[1] for w in weights]

    # If len(categories) = 2, then [w[0] + wTotalMin - w[1] for w in weights] increases the answer.
    # In this case:
    if len(categories) == 2:
        wMap = [w / 2 for w in wMap]

    resultMap = np.zeros(ma.shape(factor))
    for i, cat in enumerate(categories):
        resultMap[factor == cat] = wMap[i]

    resultMap = ma.array(data=resultMap, mask=factor.mask)
    result = {"map": resultMap, "categories": categories, "weights": wMap}
    return result


def contrast(wPlus, wMinus):
    """Weight contrast"""
    return wPlus - wMinus
