import numpy as np
def plot_chromaticity_diagram_colours(
        samples=256,
        diagram_opacity=1.0,
        diagram_clipping_path=None,
        cmfs='CIE 1931 2 Degree Standard Observer',
        method='CIE 1931',
        **kwargs):

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    method = method.upper()

    cmfs = first_item(filter_cmfs(cmfs).values())

    illuminant = COLOUR_STYLE_CONSTANTS.colour.colourspace.whitepoint

    ii, jj = np.meshgrid(
        np.linspace(0, 1, samples), np.linspace(1, 0, samples))
    ij = tstack([ii, jj])

    # NOTE: Various values in the grid have potential to generate
    # zero-divisions, they could be avoided by perturbing the grid, e.g. adding
    # a small epsilon. It was decided instead to disable warnings.
    with suppress_warnings(python_warnings=True):
        if method == 'CIE 1931':
            XYZ = xy_to_XYZ(ij)
            spectral_locus = XYZ_to_xy(cmfs.values, illuminant)
        elif method == 'CIE 1960 UCS':
            XYZ = xy_to_XYZ(UCS_uv_to_xy(ij))
            spectral_locus = UCS_to_uv(XYZ_to_UCS(cmfs.values))
        elif method == 'CIE 1976 UCS':
            XYZ = xy_to_XYZ(Luv_uv_to_xy(ij))
            spectral_locus = Luv_to_uv(
                XYZ_to_Luv(cmfs.values, illuminant), illuminant)
        else:
            raise ValueError(
                'Invalid method: "{0}", must be one of '
                '{{\'CIE 1931\', \'CIE 1960 UCS\', \'CIE 1976 UCS\'}}'.format(
                    method))

    RGB = normalise_maximum(
        XYZ_to_plotting_colourspace(XYZ, illuminant), axis=-1)

    polygon = Polygon(
        spectral_locus
        if diagram_clipping_path is None else diagram_clipping_path,
        facecolor='none',
        edgecolor='none')
    axes.add_patch(polygon)
    # Preventing bounding box related issues as per
    # https://github.com/matplotlib/matplotlib/issues/10529
    image = axes.imshow(
        RGB,
        interpolation='bilinear',
        extent=(0, 1, 0, 1),
        clip_path=None,
        alpha=diagram_opacity)
    image.set_clip_path(polygon)

    settings = {'axes': axes}
    settings.update(kwargs)

    return render(**kwargs)