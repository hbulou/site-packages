"""Module for plotting atoms, images, line scans, and diffraction patterns."""
from __future__ import annotations

import string
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from typing import TYPE_CHECKING, Sequence, Iterable, Any

import traitlets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection, EllipseCollection
from matplotlib.colors import ListedColormap
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import Size, Divider
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.axes_grid import _cbaraxes_class_factory, CbarAxesBase
from traitlets.traitlets import link

from abtem.atoms import pad_atoms, plane_to_axes
from abtem.core import config
from abtem.core.axes import ReciprocalSpaceAxis, format_label, LinearAxis
from abtem.core.colors import hsluv_cmap
from abtem.core.units import _get_conversion_factor
from abtem.core.utils import label_to_index, interleave, flatten_list_of_lists

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

ipywidgets_not_installed = RuntimeError(
    "This functionality of abTEM requires ipywidgets, see "
    "https://ipywidgets.readthedocs.io/en/stable/user_install.html."
)


if TYPE_CHECKING:
    from abtem.measurements import (
        BaseMeasurements,
        _BaseMeasurement1D,
        _BaseMeasurement2D,
        IndexedDiffractionPatterns,
    )


def _cbar_orientation(cbar_loc):
    if cbar_loc == "right":
        orientation = "vertical"
    elif cbar_loc == "below":
        orientation = "horizontal"
    else:
        raise ValueError()
    return orientation


def _make_default_sizes(cbar_loc="right"):
    sizes = {
        "cbar_spacing": Size.Fixed(0.4),
        "padding": Size.Fixed(0.1),
        "cbar_shift": Size.Fixed(0.1),
        "cax": Size.Fixed(0.1),
        # "padding": Size.Fixed(0.1),
    }

    if _cbar_orientation(cbar_loc) == "vertical":
        sizes["cbar_padding_left"] = Size.Fixed(0.1)
        sizes["cbar_padding_right"] = Size.Fixed(0.9)

    elif _cbar_orientation(cbar_loc) == "horizontal":
        sizes["cbar_padding_left"] = Size.Fixed(0.4)
        sizes["cbar_padding_right"] = Size.Fixed(0.4)

    return sizes


def _cbar_layout(n: int, sizes: dict) -> list:
    if n == 0:
        return []

    layout = [sizes["cbar_padding_left"]]
    for i in range(n):
        layout.extend([sizes["cbar"]])

        if i < n - 1:
            layout.extend([sizes["cbar_spacing"]])

    layout.extend([sizes["cbar_padding_right"]])
    return layout


def _insert_multiple(l, l2, i):
    return l[:i] + l2 + l[i:]


def _make_grid_layout(
    axes,
    ncbars: int,
    sizes: dict,
    cbar_mode: str = "each",
    cbar_loc: str = "right",
    direction: str = "col",
):

    sizes_layout = []
    for i, ax in enumerate(axes):
        if direction == "col":
            sizes_layout.append([Size.AxesX(ax, aspect="axes", ref_ax=axes[0])])
        elif direction == "row":
            sizes_layout.append([Size.AxesY(ax, aspect="axes", ref_ax=axes[0])])
        else:
            raise ValueError()

        if (i < len(axes) - 1) and ((ncbars == 0) or (cbar_mode == "single")):
            sizes_layout[-1].append(sizes["padding"])

    if not "cbar" in sizes:
        sizes["cbar"] = Size.from_any("5%", sizes_layout[0][0])

    if cbar_mode == "each":
        if _cbar_orientation(cbar_loc) == "vertical":
            cbar_layouts = [_cbar_layout(ncbars, sizes) for _ in range(len(axes))]
            sizes_layout = interleave(sizes_layout, cbar_layouts)
        elif _cbar_orientation(cbar_loc) == "horizontal":
            cbar_layouts = [_cbar_layout(ncbars, sizes)[::-1] for _ in range(len(axes))]
            sizes_layout = interleave(cbar_layouts, sizes_layout)
    elif cbar_mode == "single":
        if _cbar_orientation(cbar_loc) == "vertical":
            sizes_layout.extend([_cbar_layout(ncbars, sizes)])
        elif _cbar_orientation(cbar_loc) == "horizontal":
            sizes_layout = [_cbar_layout(ncbars, sizes)[::-1]] + sizes_layout
    else:
        raise ValueError()

    sizes_layout = flatten_list_of_lists(sizes_layout)
    return sizes_layout


def _validate_cyclic_cmap(cmap):
    if cmap == "hsluv":
        cmap = hsluv_cmap

    return cmap


class AxesGrid:
    def __init__(
        self,
        fig,
        ncols: int,
        nrows: int,
        ncbars: int = 0,
        cbar_mode: str = "single",
        cbar_loc: str = "right",
        aspect: bool = True,
        anchor: str = "NW",
        sharex: bool = True,
        sharey: bool = True,
        rect: tuple = (0., 0, 1, 1),
    ):

        self._fig = fig
        self._ncols = ncols
        self._nrows = nrows
        self._ncbars = ncbars
        self._aspect = aspect
        self._sharex = sharex
        self._sharey = sharey
        self._rect = rect
        self._anchor = anchor
        self._cbar_loc = cbar_loc
        self._cbar_mode = cbar_mode

        self._sizes = {
            "cbar_spacing": Size.Fixed(0.6),
            "padding": Size.Fixed(0.1),
            "cbar_shift": Size.Fixed(0.1),
            "cbar_width": Size.Fixed(0.1),
            "left": Size.Fixed(0.0),
            "right": Size.Fixed(0.0),
            "top": Size.Fixed(0.0),
            "bottom": Size.Fixed(0.0),
        }

        self._axes = self._make_axes()
        self._caxes = self._make_caxes()
        self._divider = self._make_divider()

        self._set_axes_locators()
        self._set_caxes_locators()

        if sharex:
            for inner_axes in self._axes[:, 1:]:
                for ax in inner_axes:
                    ax._axislines["bottom"].toggle(ticklabels=False, label=False)

        if sharey:
            for inner_axes in self._axes[1:]:
                for ax in inner_axes:
                    ax._axislines["left"].toggle(ticklabels=False, label=False)

        # self.set_layout(ncbars, cbar_loc)

    def _make_axes(self):
        from mpl_toolkits.axes_grid1.mpl_axes import Axes

        ax = Axes(self.fig, (0, 0, 1, 1), sharex=None, sharey=None)

        axes = [ax] + [
            Axes(self.fig, (0, 0, 1, 1), sharex=ax, sharey=ax)
            for _ in range(self._nrows * self._ncols - 1)
        ]

        for ax in axes:
            self._fig.add_axes(ax)

        return np.array(axes, dtype=object).reshape((self._ncols, self._nrows))

    def _remove_caxes(self):
        if self._cbar_mode == "single":
            self._caxes = self._caxes[0, 0]

        for cax in self._caxes.ravel():
            cax.remove()

    def _make_caxes(self):
        orientation = _cbar_orientation(self._cbar_loc)

        if self._cbar_mode == "each":
            caxes = [
                _cbaraxes_class_factory(Axes)(
                    self._fig, self._rect, orientation=orientation
                )
                for _ in range(self._nrows * self._ncols * self._ncbars)
            ]
        else:
            caxes = (
                [
                    _cbaraxes_class_factory(Axes)(
                        self._fig, self._rect, orientation=orientation
                    )
                    for _ in range(self._ncbars)
                ]
                * self._nrows
                * self._ncols
            )

        for cax in caxes:
            self._fig.add_axes(cax)

        caxes = np.array(caxes, dtype=object).reshape(
            (self._ncols, self._nrows, self._ncbars)
        )
        return caxes

    def _make_size(self, axes, line_types, axes_size):
        i = 0
        line_sizes = []
        for row_type in line_types:
            if row_type == "ax":
                line_sizes.append(axes_size(axes[i], aspect="axes", ref_ax=axes[0]))
                i += 1
            else:
                line_sizes.append(self._sizes[row_type])

        return line_sizes

    def _make_divider(self):
        row_types, col_types = self._get_row_types(), self._get_col_types()

        row_sizes = self._make_size(self._axes[0], row_types, Size.AxesY)
        col_sizes = self._make_size(self._axes[:, 0], col_types, Size.AxesX)

        divider = Divider(
            self._fig,
            self._rect,
            horizontal=col_sizes,
            vertical=row_sizes,
            aspect=self._aspect,
            anchor="C",
        )
        return divider

    def _set_axes_locators(self):
        row_types, col_types = self._get_row_types(), self._get_col_types()
        i = 0
        for ny, row_type in enumerate(row_types):
            for nx, col_type in enumerate(col_types):
                if (row_type == "ax") and (col_type == "ax"):
                    locator = self._divider.new_locator(nx=nx, ny=ny)
                    self._axes.ravel()[i].set_axes_locator(locator)
                    i += 1

    def _set_caxes_locators(self):
        row_types, col_types = self._get_row_types(), self._get_col_types()
        i = 0
        if (self._cbar_mode == "single") and (self._cbar_loc == "right"):
            for nx, col_type in enumerate(col_types):
                if col_type == "cbar_width":
                    locator = self._divider.new_locator(nx=nx, ny=1, ny1=-2)
                    self._caxes.ravel()[i].set_axes_locator(locator)
                    i += 1

        elif (self._cbar_mode == "single") and (self._cbar_loc == "below"):
            for ny, row_type in enumerate(row_types):
                if row_type == "cbar_width":
                    locator = self._divider.new_locator(ny=ny, nx=1, nx1=-2)
                    self._caxes.ravel()[i].set_axes_locator(locator)
                    i += 1
        else:
            for ny, row_type in enumerate(row_types):
                for nx, col_type in enumerate(col_types):
                    if ((row_type == "ax") and (col_type == "cbar_width")) or (
                        (row_type == "cbar_width") and (col_type == "ax")
                    ):
                        locator = self._divider.new_locator(nx=nx, ny=ny)
                        self._caxes.ravel()[i].set_axes_locator(locator)
                        i += 1

    def _get_col_types(self):
        if self._cbar_loc == "right":
            return self._get_line_types(
                n=self._ncols, ncbars=self._ncbars, orientation="horizontal"
            )
        else:
            return self._get_line_types(
                n=self._ncols, ncbars=0, orientation="horizontal"
            )

    def _get_row_types(self):
        if self._cbar_loc == "right":
            return self._get_line_types(n=self._nrows, ncbars=0, orientation="vertical")
        else:
            return self._get_line_types(
                n=self._nrows, ncbars=self._ncbars, orientation="vertical"
            )

    def _get_line_types(self, n, ncbars, orientation):
        cbar_types = [
            ["cbar_shift"] * (ncbars > 0) + ["cbar_width", "cbar_spacing"] * ncbars
        ]
        axes_types = [["ax", "padding"]] * (n - 1) + [["ax", "padding"]]

        if ncbars == 0:
            axes_types[-1] = axes_types[-1][:-1]

        if self._cbar_mode == "each":
            line_types = interleave(axes_types, cbar_types * n)
            line_types = flatten_list_of_lists(line_types)
        else:
            line_types = flatten_list_of_lists(axes_types + cbar_types)

        if self._cbar_loc == "below":
            line_types = line_types[::-1]

        if orientation == "horizontal":
            line_types = ["left"] + line_types + ["right"]
        else:
            line_types = ["bottom"] + line_types + ["top"]

        return line_types

    def set_sizes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self._sizes[key], "fixed_size", value)

    def set_cbar_layout(self, **kwargs):
        self._ncbars = kwargs.pop("ncbars", self._ncbars)
        self._cbar_loc = kwargs.pop("cbar_loc", self._cbar_loc)
        self._cbar_mode = kwargs.pop("cbar_mode", self._cbar_mode)

        self._remove_caxes()
        self._caxes = self._make_caxes()
        self._divider = self._make_divider()

        self._set_axes_locators()
        self._set_caxes_locators()

    @property
    def axes(self):
        return self._axes

    @property
    def fig(self):
        return self._fig

    @property
    def ncols(self) -> int:
        return self._axes.shape[0]

    @property
    def nrows(self) -> int:
        return self._axes.shape[1]

    def __getitem__(self, item):
        return self._axes[item]

    def __len__(self) -> int:
        return len(self._axes)

    @property
    def shape(self) -> tuple[int, int]:
        return self._axes.shape


def _axes_grid_cols_and_rows(ensemble_shape, axes_types):

    shape = tuple(
        n
        for n, axes_type in zip(ensemble_shape, axes_types)
        if not axes_type in ("index", "range", "overlay")
    )

    if len(shape) > 0:
        ncols = shape[0]
    else:
        ncols = 1

    if len(shape) > 1:
        nrows = shape[1]
    else:
        nrows = 1

    return ncols, nrows



def _determine_axes_types(
    ensemble_axes_metadata,
    explode: bool | tuple[bool, ...] | None,
    overlay: bool | tuple[bool, ...] | None,
):
    num_ensemble_axes = len(ensemble_axes_metadata)

    axes_types = []
    for axis_metadata in ensemble_axes_metadata:
        if axis_metadata._default_type is not None:
            axes_types.append(axis_metadata._default_type)
        else:
            axes_types.append("index")

    if explode is True:
        explode = tuple(range(max(num_ensemble_axes - 2, 0), num_ensemble_axes))
    elif explode is False:
        explode = ()

    if overlay is True:
        overlay = tuple(range(max(num_ensemble_axes - 2, 0), num_ensemble_axes))
    elif overlay is False:
        overlay = ()

    axes_types = list(axes_types)
    for i, axis_type in enumerate(axes_types):

        if explode is not None:
            if i in explode:
                axes_types[i] = "explode"
            else:
                axes_types[i] = "index"

        if overlay is not None:
            if i in overlay:
                axes_types[i] = "overlay"
            if (explode is not None) and (i not in explode):
                axes_types[i] = "index"

    return axes_types


def _validate_axes(
    ensemble_axes_metadata,
    ax: Axes,
    explode: bool = False,
    overlay: bool = False,
    ncbars: int = 0,
    common_color_scale: bool = False,
    figsize: tuple[float, float] = None,
    ioff: bool = False,
    aspect: bool = True,
    sharex: bool = True,
    sharey: bool = True,
) -> AxesGrid:
    axes_types = _determine_axes_types(ensemble_axes_metadata, explode, overlay)

    if common_color_scale:
        cbar_mode = "single"
    else:
        cbar_mode = "each"

    if ax is None:
        if ioff:
            with plt.ioff():
                fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure(figsize=figsize)
    else:
        fig = ax.get_figure()

    if ax is None:

        ncols, nrows = _axes_grid_cols_and_rows(measurements, axes_types)

        axes = AxesGrid(
            fig=fig,
            ncols=ncols,
            nrows=nrows,
            ncbars=ncbars,
            cbar_mode=cbar_mode,
            aspect=aspect,
            sharex=sharex,
            sharey=sharey,
        )


    elif ax is None:
        ax = fig.add_subplot()
        axes = np.array([[ax]])
    else:
        if explode:
            raise NotImplementedError("`ax` not implemented with `explode = True`.")

        axes = np.array([[ax]])

    return axes


def _format_options(options):

    formatted_options = []
    for option in options:
        if isinstance(option, float):
            formatted_options.append(f"{option:.3f}")
        elif isinstance(option, tuple):
            formatted_options.append(
                ", ".join(tuple(f"{value:.3f}" for value in option))
            )
        else:
            formatted_options.append(option)

    return formatted_options


def discrete_cmap(num_colors, base_cmap):
    if isinstance(base_cmap, str):
        base_cmap = plt.get_cmap(base_cmap)
    colors = base_cmap(range(0, num_colors))
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colors, num_colors)


# def _check_ensemble_axes_equal(masurements):


def _make_update_selection_sliders_button(sliders, indices):
    reset_button = widgets.Button(
        description="Reset sliders",
        disabled=False,
    )

    def reset(*args):
        for slider, index in zip(sliders, indices):
            slider.index = index

    reset_button.on_click(reset)

    return reset_button


def _make_update_indices_function(visualization, sliders):
    def update_indices(change):
        indices = ()
        for slider in sliders:
            idx = slider.index
            if isinstance(idx, tuple):
                idx = slice(*idx)
            indices += (idx,)

        with sliders[0].hold_trait_notifications():
            visualization.set_ensemble_indices(indices)
            if visualization._autoscale:
                min_value, max_value = visualization._get_default_scale_limits()
                visualization.update_scale(min_value, max_value)

    return update_indices


def _make_continuous_update_button(sliders: list, continuous_update=None):
    if continuous_update is None:
        continuous_update = config.get("visualize.continuous_update", False)

    continuous_update_checkbox = widgets.ToggleButton(
        description="Continuous update", value=continuous_update
    )
    for slider in sliders:
        link((continuous_update_checkbox, "value"), (slider, "continuous_update"))
    return continuous_update_checkbox


def make_sliders_from_ensemble_axes(
    visualizations: MeasurementVisualization | Sequence[MeasurementVisualization],
    axes_types: Sequence[str, ...] = None,
    continuous_update: bool = None,
    callbacks: tuple[callable, ...] = (),
    default_values: Any = None,
):
    if continuous_update is None:
        continuous_update = config.get("visualize.continuous_update", False)

    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    ensemble_axes_metadata = visualizations[0].measurements.ensemble_axes_metadata
    ensemble_shape = visualizations[0].measurements.ensemble_shape
    if axes_types is None:
        axes_types = [metadata._default_type for metadata in ensemble_axes_metadata]
    elif not (len(axes_types) == len(ensemble_shape)):
        raise ValueError()

    for visualization in visualizations[1:]:
        # if not isinstance(visualization, MeasurementVisualization):
        #     raise ValueError()

        if not (
            (
                visualization.measurements.ensemble_axes_metadata
                == ensemble_axes_metadata
            )
            and (visualization.measurements.ensemble_shape == ensemble_shape)
        ):
            raise ValueError()

    if not isinstance(default_values, Sequence):
        default_values = (default_values,) * len(ensemble_shape)

    elif not (len(default_values) == len(ensemble_shape)):
        raise ValueError()

    sliders = []
    default_indices = []
    for axes_metadata, n, axes_type, default_value in zip(
        ensemble_axes_metadata, ensemble_shape, axes_types, default_values
    ):
        values = np.array(axes_metadata.coordinates(n))
        options = _format_options(values)

        if default_value is None:
            index = 0
        else:
            index = int(np.argmin(np.abs(values - default_value)))

        default_indices.append(index)

        with config.set({"visualize.use_tex": False}):
            label = axes_metadata.format_label()

        if axes_type == "range":
            sliders.append(
                widgets.SelectionRangeSlider(
                    description=label,
                    options=options,
                    continuous_update=continuous_update,
                    index=(0, len(options) - 1),
                )
            )
        elif axes_type == "index" or axes_type is None:
            sliders.append(
                widgets.SelectionSlider(
                    description=label,
                    options=options,
                    continuous_update=continuous_update,
                    index=index,
                )
            )

    for visualization in visualizations:
        update_indices = _make_update_indices_function(visualization, sliders)

        update_indices({})
        for slider in sliders:
            slider.observe(update_indices, "value")
            for callback in callbacks:
                slider.observe(callback, "value")

    reset_button = _make_update_selection_sliders_button(
        sliders, indices=default_indices
    )

    continuous_update_button = _make_continuous_update_button(
        sliders, continuous_update
    )

    return sliders, reset_button, continuous_update_button


def _get_max_range(array, axes_types):
    if np.iscomplexobj(array):
        array = np.abs(array)

    max_values = array.max(
        tuple(
            i for i, axes_type in enumerate(axes_types) if axes_type not in ("range",)
        )
    )

    positive_indices = np.where(max_values > 0)[0]

    if len(positive_indices) <= 1:
        max_value = np.max(max_values)
    else:
        max_value = np.sum(max_values[positive_indices])

    return max_value


def _make_vmin_vmax_slider(visualization):
    axes_types = (
        tuple(visualization._axes_types)
        + ("base",) * visualization.measurements.num_base_axes
    )

    max_value = _get_max_range(visualization.measurements.array, axes_types)
    min_value = -_get_max_range(-visualization.measurements.array, axes_types)

    step = (max_value - min_value) / 1e6

    vmin_vmax_slider = widgets.FloatRangeSlider(
        value=visualization._get_vmin_vmax(),
        min=min_value,
        max=max_value,
        step=step,
        disabled=visualization._autoscale,
        description="Normalization",
        continuous_update=True,
    )

    def vmin_vmax_slider_changed(change):
        vmin, vmax = change["new"]
        vmax = max(vmax, vmin + step)

        with vmin_vmax_slider.hold_trait_notifications():
            visualization._update_vmin_vmax(vmin, vmax)

    vmin_vmax_slider.observe(vmin_vmax_slider_changed, "value")
    return vmin_vmax_slider


def make_scale_button(
    visualizations: MeasurementVisualization | Sequence[MeasurementVisualization],
):
    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    def scale_button_clicked(*args):
        for visualization in visualizations:
            visualization.update_scale()

    scale_button = widgets.Button(description="Scale")
    scale_button.on_click(scale_button_clicked)
    return scale_button


def make_autoscale_button(
    visualizations: MeasurementVisualization | Sequence[MeasurementVisualization],
):
    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    def autoscale_button_changed(change):
        for visualization in visualizations:
            if change["new"]:
                visualization.autoscale = True
            else:
                visualization.autoscale = False

    autoscale_button = widgets.ToggleButton(
        value=visualizations[0].autoscale,
        description="Autoscale",
        tooltip="Autoscale",
    )
    autoscale_button.observe(autoscale_button_changed, "value")
    return autoscale_button


def make_power_scale_slider(
    visualizations: BaseMeasurementVisualization2D
    | Sequence[BaseMeasurementVisualization2D],
    **kwargs,
):
    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    def powerscale_slider_changed(change):
        for visualization in visualizations:
            visualization.update_power(change["new"])

    default_kwargs = {
        "min": 0.01,
        "max": 2.0,
        "step": 0.01,
        "description": "Power",
        "tooltip": "Power scale",
    }

    kwargs = {**default_kwargs, **kwargs}

    power_scale_slider = widgets.FloatSlider(
        value=visualizations[0]._get_power(), **kwargs
    )
    power_scale_slider.observe(powerscale_slider_changed, "value")
    return power_scale_slider


def make_complex_visualization_dropdown(
    visualizations: MeasurementVisualization | Sequence[MeasurementVisualization],
):

    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    def dropdown_changed(change):
        for visualization in visualizations:
            visualization.set_complex_conversion(change["new"])

    dropdown = widgets.Dropdown(
        options=[
            ("Domain coloring", "none"),
            ("Amplitude", "abs"),
            ("Intensity", "intensity"),
            ("Phase", "phase"),
            ("Real", "real"),
            ("Imaginary", "imag"),
        ],
        value="none",
        description="Complex visualization:",
    )

    dropdown.observe(dropdown_changed, "value")

    return dropdown


def make_cmap_dropdown(
    visualizations,
):
    if not isinstance(visualizations, Sequence):
        visualizations = [visualizations]

    def dropdown_changed(change):
        cmap = change["new"]
        if cmap == "default":
            cmap = None
        for visualization in visualizations:
            visualization.set_cmaps(cmap)

    options = ["default", "viridis", "magma", "gray", "jet", "hsluv", "hsv", "twilight"]

    dropdown = widgets.Dropdown(
        options=options,
        value="default",
        description="Colormap:",
    )

    dropdown.observe(dropdown_changed, "value")

    return dropdown


def _get_joined_titles(measurement, formatting, **kwargs):
    titles = []
    for axes_metadata in measurement.ensemble_axes_metadata:
        titles.append(axes_metadata.format_title(formatting, **kwargs))
    return "\n".join(titles)


class MeasurementVisualization(metaclass=ABCMeta):
    def __init__(
        self,
        measurements: BaseMeasurements,
        axes: AxesGrid | np.ndarray,
        axes_types: Sequence[str] = (),
        autoscale: bool = None,
    ):
        self._measurements = measurements
        self._axes = axes
        self._axes_types = axes_types
        self._indices = self._validate_ensemble_indices()
        self._column_titles = []
        self._row_titles = []
        self._panel_labels = []
        self._metadata_labels = np.array([])
        self._xunits = None
        self._yunits = None
        if autoscale is None:
            autoscale = config.get("visualize.autoscale", False)
        self._autoscale = autoscale
        self._complex_conversion = "none"

        for ax in np.array(self.axes).ravel():
            ax.ticklabel_format(
                style="sci", scilimits=(-3, 3), axis="both", useMathText=True
            )

        self.fig.canvas.header_visible = False

    @property
    def autoscale(self) -> bool:
        return self._autoscale

    @autoscale.setter
    def autoscale(self, value: bool):
        if value is True:
            self.update_scale()
        self._autoscale = value

    @abstractmethod
    def update_scale(self, min_value: float = None, max_value: float = None):
        pass

    @property
    def fig(self):
        return self._axes[0, 0].get_figure()

    @property
    @abstractmethod
    def artists(self):
        pass

    def adjust_figure_to_bbox(self):
        self.axes.set_sizes(left=0., bottom=0.)

        size = self.fig.get_size_inches()
        bbox_inches = self.fig.get_tightbbox()
        pad_inches = plt.rcParams["savefig.pad_inches"]
        bbox_inches.padded(pad_inches)
        aspect = bbox_inches.width / bbox_inches.height
        self.fig.set_size_inches((size[0], size[0] / aspect))

        bbox_inches = self.fig.get_tightbbox().padded(pad_inches)
        self.axes.set_sizes(left=-bbox_inches.xmin)

        bbox_inches = self.fig.get_tightbbox().padded(pad_inches)
        self.axes.set_sizes(bottom=-bbox_inches.ymin)
        self.fig.canvas.draw_idle()

        # bbox_inches = visualization.fig.get_tightbbox()
        # pad_inches = plt.rcParams["savefig.pad_inches"]
        # bbox_inches.padded(pad_inches)
        # r = adjust_bbox(visualization.fig, bbox_inches)
        # visualization.fig.set_size_inches((bbox_inches.width, bbox_inches.height))

    def _generate_measurements(self, keepdims: bool = True):
        indexed_measurements = self._get_display_measurements()

        shape = tuple(
            n if axes_type != "overlay" else 1
            for n, axes_type in zip(
                indexed_measurements.ensemble_shape, self._axes_types
            )
        )

        for indices in np.ndindex(*shape):
            axes_index = ()
            for i, axes_type in zip(indices, self._axes_types):
                if axes_type == "explode":
                    axes_index += (i,)
            axes_index = (axes_index + (0,) * (2 - len(axes_index)))[:2]

            indices = tuple(
                i if axes_type != "overlay" else slice(None)
                for i, axes_type in zip(indices, self._axes_types)
            )
            yield axes_index, indexed_measurements.get_items(indices, keepdims=keepdims)

    def _get_axes_from_axes_types(self, axes_type):
        return tuple(
            i
            for i, checked_axes_type in enumerate(self.axes_types)
            if checked_axes_type == axes_type
        )

    def _get_indexed_measurements(self, keepdims: bool = True):
        indexed = self.measurements.get_items(self._indices, keepdims=keepdims)

        if keepdims:
            summed_axes = tuple(
                i
                for i, axes_type in enumerate(self._axes_types)
                if axes_type == "range"
            )
        else:
            i = 0
            summed_axes = ()
            for axes_type in self._axes_types:
                if axes_type == "range":
                    summed_axes += (i,)
                    i += 1

        indexed = indexed.sum(axis=summed_axes, keepdims=keepdims)

        return indexed

    def _get_display_measurements(self, keepdims: bool = True):
        measurements = self._get_indexed_measurements(keepdims).compute()

        if measurements.is_complex and self._complex_conversion:
            if self._complex_conversion != "none":
                measurements = getattr(measurements, self._complex_conversion)()

        return measurements

    def set_column_titles(
        self,
        titles: str | list[str] = None,
        pad: float = 10.0,
        format: str = ".3g",
        units: str = None,
        fontsize=12,
        **kwargs,
    ):

        indexed_measurements = self._get_indexed_measurements(keepdims=False)

        if titles is None or titles is True:
            if not len(indexed_measurements.ensemble_shape):
                return

            # TODO: same for row titles
            j = 0
            for j, axes_type in enumerate(self.axes_types):
                if not axes_type == "overlay":
                    break

            axes_metadata = indexed_measurements.ensemble_axes_metadata[j]

            if hasattr(axes_metadata, "to_nonlinear_axis"):
                axes_metadata = axes_metadata.to_nonlinear_axis(
                    indexed_measurements.ensemble_shape[j]
                )

            titles = []
            for i, axis_metadata in enumerate(axes_metadata):
                titles.append(
                    axis_metadata.format_title(
                        format, units=units, include_label=i == 0
                    )
                )
                if i == indexed_measurements.ensemble_shape[j]:
                    break

        elif isinstance(titles, str):
            if indexed_measurements.ensemble_shape:
                n = indexed_measurements.ensemble_shape[0]
            else:
                n = 1

            titles = [titles] * n

        for column_title in self._column_titles:
            column_title.remove()

        column_titles = []

        for i, ax in enumerate(self.axes[:, -1]):
            annotation = ax.annotate(
                titles[i],
                xy=(0.5, 1),
                xytext=(0, pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                fontsize=fontsize,
                **kwargs,
            )
            column_titles.append(annotation)

        self._column_titles = column_titles

    @abstractmethod
    def _get_default_xlabel(self, units=None):
        pass

    @abstractmethod
    def _get_default_ylabel(self, units=None):
        pass

    def set_xlabels(self, label: str = None):
        if label is None:
            label = self._get_default_xlabel(units=self._xunits)

        for i, j in np.ndindex(self.axes.shape):  # noqa
            if j == 0:
                ax = self.axes[i, j]
                ax.set_xlabel(label)

    def set_ylabels(self, label: str = None):
        if label is None:
            label = self._get_default_ylabel(units=self._yunits)

        for i, j in np.ndindex(self.axes.shape):  # noqa
            if i == 0:
                ax = self.axes[i, j]
                ax.set_ylabel(label)

    @abstractmethod
    def set_xlim(self):
        pass

    @abstractmethod
    def set_ylim(self):
        pass

    @abstractmethod
    def _get_default_xunits(self):
        pass

    @abstractmethod
    def _get_default_yunits(self):
        pass

    def set_xunits(self, units: str = None):
        """
        Set the units for the x-axis.

        Parameters
        ----------
        units : str
            The name of the units. Must be compatible with existing units.
        """
        if units is None:
            self._xunits = self._get_default_xunits()
        else:
            self._xunits = units

        self.set_xlabels()
        self.set_xlim()

    def set_yunits(self, units: str = None):
        """
        Set the units for the y-axis.

        Parameters
        ----------
        units : str
            The name of the units. Must be compatible with existing units.
        """
        if units is None:
            self._yunits = self._get_default_yunits()
        else:
            self._yunits = units

        self.set_ylabels()
        self.set_ylim()

    def set_row_titles(
        self,
        titles: str | list[str] = None,
        shift: float = 0.0,
        format: str = ".3g",
        units: str = None,
        **kwargs,
    ):
        """
        Set the titles for the rows of the grid of axes.

        Parameters
        ----------
        titles : str or list of str, optional
            If given as list, each item is given as a title for a row, the list must have the same length as the number
            of rows. If given as string the same title is given to all rows. If not given the titles are derived from
            the axes metadata.
        shift : float, optional
            Horizontal shift of the title positions.
        format : str, optional
            String formatting of titles derived from axes metadata.
        units : str, optional
            The units used for titles derived from axes metadata.
        """
        indexed_measurements = self._get_indexed_measurements()

        if not "fontsize" in kwargs:
            kwargs.update({"fontsize": 12})

        if titles is None:
            if not len(indexed_measurements.ensemble_shape) > 1:
                return

            axes_metadata = indexed_measurements.ensemble_axes_metadata[1]

            if hasattr(axes_metadata, "to_nonlinear_axis"):
                axes_metadata = axes_metadata.to_nonlinear_axis(
                    indexed_measurements.ensemble_shape[1]
                )

            titles = []
            for i, axis_metadata in enumerate(axes_metadata):
                titles.append(
                    axis_metadata.format_title(
                        format, units=units, include_label=i == 0
                    )
                )

                if i == indexed_measurements.ensemble_shape[1]:
                    break
        elif isinstance(titles, str):
            titles = [titles] * max(len(indexed_measurements.ensemble_shape), 1)

        for row_title in self._row_titles:
            row_title.remove()

        row_titles = []
        for i, ax in enumerate(self.axes[0, :]):
            annotation = ax.annotate(
                titles[i],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - shift, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=90,
                **kwargs,
            )
            row_titles.append(annotation)

        self._row_titles = row_titles

    @property
    def ncols(self):
        return self._axes.shape[0]

    @property
    def nrows(self):
        return self._axes.shape[1]

    @property
    def axes_types(self):
        return self._axes_types

    @property
    def indices(self):
        return self._indices

    @property
    def measurements(self):
        return self._measurements

    @property
    def axes(self):
        return self._axes

    def _validate_ensemble_indices(self, indices: int | tuple[int, ...] = ()):
        if isinstance(indices, int):
            indices = (indices,)

        num_ensemble_dims = len(self.measurements.ensemble_shape)
        explode_axes = self._get_axes_from_axes_types("explode")
        overlay_axes = self._get_axes_from_axes_types("overlay")
        num_indexing_axes = num_ensemble_dims - len(explode_axes) - len(overlay_axes)

        if len(indices) > num_indexing_axes:
            raise ValueError

        validated_indices = []
        j = 0
        for i, axes_type in enumerate(self.axes_types):
            if axes_type in ("explode", "overlay"):
                validated_indices.append(slice(None))
            elif j < len(indices):
                validated_indices.append(indices[j])
                j += 1
            elif axes_type == "index":
                validated_indices.append(0)
            elif axes_type == "range":
                validated_indices.append(slice(None))
            else:
                raise RuntimeError(
                    "axes type must be one of 'index', 'range', 'explode' or 'overlay'"
                )

        return tuple(validated_indices)

    def set_ensemble_indices(self, indices: int | tuple[int, ...] = ()):
        """
        Set the indices into the ensemble dimensions to select the visualized ensemble members. Interactive
        visualization are updated.

        Parameters
        ----------
        indices : int or tuple of int

        """

        self._indices = self._validate_ensemble_indices(indices)
        self.update_artists()
        self.update_panel_labels()

    @abstractmethod
    def update_artists(self):
        pass

    def _get_default_scale_limits(
        self, min_value: float = None, max_value: float = None
    ) -> tuple[float, float]:

        measurements = self._get_display_measurements()

        if measurements.is_complex:
            measurements = measurements.abs()

        if min_value is None:
            min_value = float(np.nanmin(measurements.array))

        if max_value is None:
            max_value = float(np.nanmax(measurements.array))

        return min_value, max_value

    def set_panel_labels(
        self,
        labels: str = "metadata",
        frameon: bool = True,
        loc: str = "upper left",
        pad: float = 0.1,
        borderpad: float = 0.1,
        prop: dict = None,
        formatting: str = ".3g",
        units: str = None,
        **kwargs,
    ):

        labels_type = labels

        if labels == "alphabetic":
            labels = string.ascii_lowercase
            labels = [f"({label})" for label in labels]
            if config.get("visualize.use_tex", False):
                labels = [f"${label}$" for label in labels]
        elif labels == "metadata":
            labels = []

            for i, measurement in self._generate_measurements(keepdims=True):
                titles = []
                for axes_metadata in measurement.ensemble_axes_metadata:
                    titles.append(
                        axes_metadata.format_title(formatting, units=units, **kwargs)
                    )
                labels.append("\n".join(titles))

            # for i, measurement in self.generate_measurements(keepdims=True):
            #     labels.append(_get_joined_titles(measurement, formatting))
        elif (
            not isinstance(labels, (tuple, list))
            and len(labels) != np.array(self.axes).size
        ):
            raise ValueError()

        if prop is None:
            prop = {}

        for old_label in self._panel_labels:
            old_label.remove()

        anchored_text = []
        for ax, l in zip(np.array(self.axes).ravel(), labels):
            at = AnchoredText(
                l,
                pad=pad,
                borderpad=borderpad,
                frameon=frameon,
                loc=loc,
                prop=prop,
                **kwargs,
            )
            at.formatting = formatting

            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
            anchored_text.append(at)

        self._panel_labels = anchored_text

        if labels_type == "metadata":
            self._metadata_labels = self._panel_labels
        else:
            self._metadata_labels = []

    def update_panel_labels(self):
        for anchored_text, (i, measurement) in zip(
            self._metadata_labels, self._generate_measurements(keepdims=True)
        ):

            label = _get_joined_titles(measurement, anchored_text.formatting)
            anchored_text.txt.set_text(label)

    def animate(
        self,
        interval=20,
        blit=True,
        repeat: bool = False,
        adjust_scale: bool = True,
        **kwargs,
    ):
        def update(i):
            self.set_ensemble_indices((i,))
            if adjust_scale:
                self._update_vmin_vmax()

            return self.artists.ravel()

        index_axes = self._get_axes_from_axes_types("index")

        if len(index_axes) == 0:
            raise RuntimeError()

        frames = self.measurements.shape[index_axes[0]]

        animation = FuncAnimation(
            self.fig,
            update,
            frames=frames,
            interval=interval,
            blit=blit,
            repeat=repeat,
            **kwargs,
        )

        return animation


class BaseMeasurementVisualization2D(MeasurementVisualization):
    def __init__(
        self,
        measurements: _BaseMeasurement2D | IndexedDiffractionPatterns,
        ax: Axes = None,
        common_scale: bool = False,
        cbar: bool = False,
        explode: bool = None,
        figsize: tuple[float, float] = None,
        interact: bool = False,
        autoscale: bool = None,
    ):

        axes_types = _determine_axes_types(
            measurements=measurements, explode=explode, overlay=None
        )

        if "overlay" in axes_types:
            raise NotImplementedError

        axes = _validate_axes(
            measurements=measurements,
            ax=ax,
            explode=explode,
            overlay=None,
            cbar=cbar,
            common_color_scale=common_scale,
            figsize=figsize,
            ioff=interact,
        )

        super().__init__(
            measurements=measurements,
            axes=axes,
            axes_types=axes_types,
            autoscale=autoscale,
        )

        self._xunits = None
        self._yunits = None
        self._scale_units = None
        self._xlabel = None
        self._ylabel = None
        self._column_titles = []
        self._row_titles = []
        self._artists = None
        self._common_scale = common_scale
        self._size_bars = []
        self._cbars = {}

        if self.ncols > 1:
            self.set_column_titles()

        if self.nrows > 1:
            self.set_row_titles()

    def _get_vmin_vmax(self):
        vmin = np.inf
        vmax = -np.inf
        for norm in self._normalization.ravel():
            vmin = min(vmin, norm.vmin)
            vmax = max(vmax, norm.vmax)
        return vmin, vmax

    def _get_power(self):
        power = None
        for norm in self._normalization.ravel():
            if isinstance(norm, colors.PowerNorm):
                if power is None:
                    power = norm.gamma
                else:
                    power = min(power, norm.gamma)
            else:
                if power is None:
                    power = 1.0
                else:
                    power = min(power, 1.0)

        return power

    def _new_normalization(
        self,
        power: float = 1.0,
        vmin: float = None,
        vmax: float = None,
    ):

        if self._common_scale:
            vmin, vmax = self._get_default_scale_limits(min_value=vmin, max_value=vmax)

        self._normalization = np.zeros(self.axes.shape, dtype=object)
        for i, measurement in self._generate_measurements(keepdims=False):

            if power == 1.0:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = colors.PowerNorm(gamma=power, vmin=vmin, vmax=vmax)

            if measurement.is_complex:
                measurement = measurement.abs()

            norm.autoscale_None(measurement.array[np.isnan(measurement.array) == 0])

            self._normalization[i] = norm

    @abstractmethod
    def _set_normalization(self):
        pass

    def update_scale(self, min_value: float = None, max_value: float = None):
        min_value, max_value = self._get_default_scale_limits(min_value, max_value)

        for norm, measurement in zip(
            self._normalization.ravel(), self._generate_measurements(keepdims=False)
        ):

            norm.vmin = min_value
            norm.vmax = max_value

    def add_area_indicator(self, area_indicator, panel="first", **kwargs):

        xlim = self.axes[0, 0].get_xlim()
        ylim = self.axes[0, 0].get_ylim()

        for i, ax in enumerate(np.array(self.axes).ravel()):
            if panel == "first" and i == 0:
                area_indicator._add_to_visualization(ax, **kwargs)
            elif panel == "all":
                area_indicator._add_to_visualization(ax, **kwargs)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    def update_power(self, power: float = 1.0):
        set_normalization = False
        for i, measurement in self._generate_measurements(keepdims=False):
            norm = self._normalization[i]

            if (power != 1.0) and (not hasattr(norm, "gamma")):
                self._normalization[i] = colors.PowerNorm(
                    gamma=power, vmin=norm.vmin, vmax=norm.vmax
                )
                set_normalization = True

            if (power == 1.0) and hasattr(norm, "gamma"):
                self._normalization[i] = colors.Normalize(
                    vmin=norm.vmin, vmax=norm.vmax
                )
                set_normalization = True

            if (power != 1.0) and isinstance(norm, colors.PowerNorm):
                self._normalization[i].gamma = power
                self._normalization[i]._changed()

        if set_normalization:
            self._set_normalization()

    @property
    def artists(self):
        return self._artists

    @abstractmethod
    def _set_artists(self):
        pass

    def set_scale_units(self, units: str = None):
        if units is None:
            units = self._get_display_measurements().metadata.get("units", "")

        self._scale_units = units

    def set_cbar_labels(self, label: str = None, **kwargs):
        if label is None:
            label = self._get_display_measurements().metadata.get("label", "")

        if self._scale_units is None or len(self._scale_units) == 0:
            label = f"{label}"
        else:
            label = f"{label} [{self._scale_units}]"

        for cbars in self._cbars.values():
            for cbar in cbars:
                cbar.set_label(label, **kwargs)
                cbar.formatter.set_powerlimits((-2, 2))
                cbar.formatter.set_useMathText(True)
                cbar.ax.yaxis.set_offset_position("left")

    def set_cbar_layout(self):
        pass

    def _get_cmap(self):
        if self._cmap is not None:
            cmap = self._cmap

        elif self.measurements.is_complex and (
            self._complex_conversion in ("none", "phase")
        ):
            cmap = config.get("visualize.phase_cmap", "hsluv")

        else:
            cmap = config.get("visualize.cmap", "viridis")

        if cmap == "hsluv":
            cmap = hsluv_cmap

        return cmap

    def set_cmaps(self, cmap):
        self._cmap = cmap
        cmap = self._get_cmap()

        for i, _ in self._generate_measurements():
            images = self._artists[i]
            if isinstance(images, np.ndarray):
                images[0].set_cmap(cmap)
            else:
                images.set_cmap(cmap)

    def set_cbars(self, **kwargs):
        cbars = defaultdict(list)

        for i, _ in self._generate_measurements():
            ax = self.axes[i]
            images = self._artists[i]
            orientation = _cbar_orientation(self.axes._cbar_loc)

            if hasattr(self.axes, "set_cbar_layout"):
                caxes = self.axes._caxes[i]

                if isinstance(images, np.ndarray):
                    for j, image in enumerate(images):
                        cbars[ax].append(
                            plt.colorbar(
                                image, cax=caxes[j], orientation=orientation, **kwargs
                            )
                        )
                else:
                    cbars[ax].append(
                        plt.colorbar(
                            images, cax=caxes[0], orientation=orientation, **kwargs
                        )
                    )

            else:
                if isinstance(images, np.ndarray):
                    for j, image in enumerate(images[::-1]):
                        cbars[ax].insert(0, plt.colorbar(image, ax=ax, **kwargs))
                else:
                    cbars[ax].append(plt.colorbar(images, ax=ax, **kwargs))

        self._cbars = cbars

    def set_scalebars(
        self,
        panel_loc: tuple[int, ...] = ((-1, 0),),
        label: str = "",
        size: float = None,
        loc: str = "lower right",
        borderpad: float = 0.5,
        formatting: str = ".3f",
        size_vertical: float = None,
        sep: float = 6,
        pad: float = 0.3,
        label_top: bool = True,
        frameon: bool = False,
        **kwargs,
    ):

        if panel_loc == "all":
            panel_loc = np.ndindex(self.axes.shape)  # noqa
            panel_loc = tuple(panel_loc)
        elif panel_loc == "upper left":
            panel_loc = ((0, -1),)
        elif panel_loc == "upper right":
            panel_loc = ((-1, -1),)
        elif panel_loc == "lower left":
            panel_loc = ((0, 0),)
        elif panel_loc == "lower right":
            panel_loc = ((-1, 0),)
        else:
            panel_loc = ((0, 0),)

        conversion = _get_conversion_factor(
            self._xunits, self.measurements.axes_metadata[-2].units
        )

        if size is None:
            size = (
                self.measurements.base_axes_metadata[-2].sampling
                * self.measurements.base_shape[-2]
                / 3
            )

        if size_vertical is None:
            size_vertical = (
                self.measurements.base_axes_metadata[-1].sampling
                * self.measurements.base_shape[-1]
                / 20
            )

        size = conversion * size
        size_vertical = conversion * size_vertical

        if label is None:
            label = f"{size:>{formatting}} {self._xunits}"

        for size_bar in self._size_bars:
            size_bar.remove()

        self._size_bars = []
        for ax in panel_loc:
            ax = self.axes[ax]
            anchored_size_bar = AnchoredSizeBar(
                ax.transData,
                label=label,
                label_top=label_top,
                size=size,
                borderpad=borderpad,
                loc=loc,
                size_vertical=size_vertical,
                sep=sep,
                pad=pad,
                frameon=frameon,
                **kwargs,
            )
            ax.add_artist(anchored_size_bar)
            self._size_bars.append(anchored_size_bar)

    def axis_off(self, spines: bool = True):
        for ax in np.array(self.axes).ravel():
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            if not spines:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

    def adjust_tight_bbox(self):
        # x_extent = self.measurements._plot_extent_x(self._xunits)
        # y_extent = self.measurements._plot_extent_y(self._yunits)

        # aspect = (y_extent[1] - y_extent[0]) / (x_extent[1] - x_extent[0])
        aspect = 1

        size_x = self.fig.get_size_inches()[0]
        size_y = size_x * aspect

        self.fig.set_size_inches((size_x, size_y))
        self.fig.subplots_adjust(left=0, bottom=0, right=1.0, top=1)


class MeasurementVisualization2D(BaseMeasurementVisualization2D):
    """
    Show the image(s) using matplotlib.

    Parameters
    ----------
    measurements : _BaseMeasurement2D
    ax : matplotlib.axes.Axes, optional
        If given the plots are added to the axis. This is not available for image grids.
    cmap : str, optional
        Matplotlib colormap name used to map scalar data to colors. Ignored if image array is complex.
    power : float
        Show image on a power scale.
    vmin : float, optional
        Minimum of the intensity color scale. Default is the minimum of the array values.
    vmax : float, optional
        Maximum of the intensity color scale. Default is the maximum of the array values.
    common_scale : bool, optional
        If True all images in an image grid are shown on the same colorscale, and a single colorbar is created (if
        it is requested). Default is False.
    cbar : bool, optional
        Add colorbar(s) to the image(s). The position and size of the colorbar(s) may be controlled by passing
        keyword arguments to `mpl_toolkits.axes_grid1.axes_grid.ImageGrid` through `image_grid_kwargs`.

    """

    def __init__(
        self,
        measurements: _BaseMeasurement2D,
        ax: Axes = None,
        cbar: bool = False,
        cmap: str = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        common_scale: bool = False,
        explode: bool = False,
        figsize: tuple[float, float] = None,
        interact: bool = False,
    ):

        super().__init__(
            measurements,
            ax=ax,
            cbar=cbar,
            common_scale=common_scale,
            explode=explode,
            figsize=figsize,
            interact=interact,
        )

        self._normalization = None
        self._cmap = cmap

        self._new_normalization(power=power, vmin=vmin, vmax=vmax)
        self._set_artists()

        if cbar:
            self.set_cbars()
            self.set_scale_units()
            self.set_cbar_labels()

        self.set_extent()
        self.set_xunits()
        self.set_yunits()
        self.set_xlabels()
        self.set_ylabels()
        self.set_column_titles()

    @property
    def _uses_domain_coloring(self):
        return self.measurements.is_complex and (self._complex_conversion == "none")

    @property
    def _artists_per_axes(self):
        if self._uses_domain_coloring:
            return 2
        else:
            return 1

    def set_complex_conversion(self, complex_conversion: str):
        self._complex_conversion = complex_conversion
        self._new_normalization()
        self._set_artists()
        self.update_scale()
        self.set_extent()

        if self._uses_domain_coloring:
            self.axes.set_cbar_layout(ncbars=2)
        else:
            self.axes.set_cbar_layout(ncbars=1)

        self.set_cbars()
        self.set_scale_units()
        self.set_cbar_labels()

    def set_cbar_loc(self, loc):
        self.axes.set_cbar_layout(cbar_loc=loc)
        self.set_cbars()
        self.set_cbar_labels()

    def set_cbar_labels(self, label: str = None, **kwargs):
        if self._uses_domain_coloring:
            for cbar1, cbar2 in self._cbars.values():
                cbar1.set_label("arg", rotation=0, ha="center", va="top")
                cbar1.ax.yaxis.set_label_coords(0.5, -0.02)
                cbar1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
                cbar1.set_ticklabels(
                    [
                        r"$-\pi$",
                        r"$-\dfrac{\pi}{2}$",
                        "$0$",
                        r"$\dfrac{\pi}{2}$",
                        r"$\pi$",
                    ]
                )
                cbar2.set_label("abs", rotation=0, ha="center", va="top")
                cbar2.ax.yaxis.set_label_coords(0.5, -0.02)
                cbar2.formatter.set_powerlimits((0, 0))

                cbar2.formatter.set_useMathText(True)
                # cbar2.ax.xaxis.get_offset_text().set_position((0.4, 1231231321))
                # cbar2.ax.yaxis.get_offset_text().set_visible(False)
                cbar2.ax.yaxis.set_offset_position("right")

        else:
            super().set_cbar_labels(label, **kwargs)

    def _get_default_xlabel(self, units: str = None):
        return self.measurements.axes_metadata[-2].format_label(units=units)

    def _get_default_ylabel(self, units: str = None):
        return self.measurements.axes_metadata[-1].format_label(units=units)

    def _get_default_xunits(self):
        return self.measurements.axes_metadata[-2].units

    def _get_default_yunits(self):
        return self.measurements.axes_metadata[-1].units

    def set_xlim(self):
        self.set_extent()

    def set_ylim(self):
        self.set_extent()

    def set_extent(self, extent=None):

        if extent is None:
            x_extent = self.measurements._plot_extent_x(self._xunits)
            y_extent = self.measurements._plot_extent_y(self._yunits)
            extent = x_extent + y_extent

        for image in self._artists.ravel():
            image.set_extent(extent)

    def _add_domain_coloring_imshow(self, ax, measurement, norm):
        array = measurement.array
        abs_array = np.abs(array)
        alpha = np.clip(norm(abs_array), a_min=0.0, a_max=1.0)

        cmap = self._get_cmap()

        im1 = ax.imshow(
            np.angle(array).T,
            origin="lower",
            interpolation="none",
            alpha=alpha.T,
            vmin=-np.pi,
            vmax=np.pi,
            cmap=cmap,
        )

        im2 = ax.imshow(
            abs_array.T,
            origin="lower",
            interpolation="none",
            cmap="gray",
            zorder=-1,
        )

        return im1, im2

    def _add_real_imshow(self, ax, measurement):
        array = measurement.array
        cmap = self._get_cmap()

        im = ax.imshow(
            array.T,
            origin="lower",
            interpolation="none",
            cmap=cmap,
        )
        return im

    def _set_normalization(self):
        for i, measurement in self._generate_measurements(keepdims=False):
            artists = self._artists[i]
            if self._uses_domain_coloring:
                artists[1].norm = self._normalization[i]
            else:
                artists.norm = self._normalization[i]

    def _remove_artists(self):
        for ax in np.array(self.axes).ravel():
            for child in ax.get_children():
                if isinstance(child, AxesImage):
                    child.remove()

    def _set_artists(
        self,
    ):
        self._remove_artists()

        images = np.zeros(self.axes.shape + (self._artists_per_axes,), dtype=object)
        for i, measurement in self._generate_measurements(keepdims=False):
            ax = self.axes[i]
            norm = self._normalization[i]

            if self._uses_domain_coloring:
                images[i] = self._add_domain_coloring_imshow(ax, measurement, norm)
            else:
                images[i] = self._add_real_imshow(ax, measurement)

        if images.shape[-1] == 1:
            images = np.squeeze(images, -1)

        self._artists = images
        self._set_normalization()

    def _update_domain_coloring_alpha(self):
        for i, measurement in self._generate_measurements(keepdims=False):
            images = self._artists[i]
            abs_array = np.abs(measurement.array).T
            alpha = self._normalization[i](abs_array)
            alpha = np.clip(alpha, a_min=0, a_max=1)
            images[0].set_alpha(alpha)

    def update_power(self, power: float = 1.0):
        super().update_power(power=power)
        if self._uses_domain_coloring:
            self._update_domain_coloring_alpha()

    def update_scale(self, min_value: float = None, max_value: float = None):
        min_value, max_value = self._get_default_scale_limits(min_value, max_value)
        super().update_scale(min_value=min_value, max_value=max_value)
        if self._uses_domain_coloring:
            self._update_domain_coloring_alpha()

    def update_artists(self):
        for i, measurement in self._generate_measurements(keepdims=False):
            images = self._artists[i]

            array = measurement.array.T

            if self._uses_domain_coloring:
                abs_array = np.abs(array)
                alpha = self._normalization[i](abs_array)
                alpha = np.clip(alpha, a_min=0, a_max=1)
                images[0].set_alpha(alpha)
                images[0].set_data(np.angle(array))
                images[1].set_data(abs_array)
            else:
                images.set_data(array)

    @property
    def widgets(self):
        if widgets is None:
            raise ipywidgets_not_installed

        canvas = self.fig.canvas

        sliders = make_sliders_from_ensemble_axes(
            self,
            self.axes_types,
        )

        power_scale_button = make_power_scale_slider(self)

        scale_button = make_scale_button(self)
        autoscale_button = make_autoscale_button(self)
        continuous_update_button = _make_continuous_update_button(sliders)

        scale_button.layout = widgets.Layout(width="20%")
        autoscale_button.layout = widgets.Layout(width="30%")
        continuous_update_button.layout = widgets.Layout(width="50%")

        scale_box = widgets.VBox(
            [widgets.HBox([scale_button, autoscale_button, continuous_update_button])]
        )
        scale_box.layout = widgets.Layout(width="300px")

        gui = widgets.VBox(
            [
                widgets.VBox(sliders),
                scale_box,
                # vmin_vmax_slider,
                power_scale_button,
            ]
        )

        return widgets.HBox([gui, canvas])


class MeasurementVisualization1D(MeasurementVisualization):
    def __init__(
        self,
        measurements: _BaseMeasurement1D,
        ax: Axes = None,
        common_scale: bool = True,
        explode: Sequence[str] | bool = False,
        overlay: Sequence[str] | bool = False,
        figsize: tuple[float, float] = None,
        interact: bool = False,
    ):

        axes_types = _determine_axes_types(
            measurements, explode=explode, overlay=overlay
        )

        axes = _validate_axes(
            measurements=measurements,
            ax=ax,
            explode=explode,
            overlay=overlay,
            cbar=False,
            common_color_scale=False,
            figsize=figsize,
            aspect=False,
            ioff=interact,
            sharey=common_scale,
        )

        super().__init__(measurements=measurements, axes=axes, axes_types=axes_types)

        self._xunits = None
        self._yunits = None
        self._xlabel = None
        self._ylabel = None
        self._column_titles = []
        self._lines = np.array([[]])
        self._common_scale = common_scale
        self.set_artists()
        self.set_xunits()
        self.set_yunits()
        self._autoscale = config.get("visualize.autoscale", False)

        if self.ncols > 1:
            self.set_column_titles()

        #
        # for i, _ in self.iterate_measurements():
        #     #     #self.axes[i].yaxis.set_label_coords(0.5, -0.02)
        #     #     #cbar2.formatter.set_powerlimits((0, 0))
        #     #     #self.axes[i].get_yaxis().formatter.set_useMathText(True)
        #     #     #self.axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
        #     self.axes[i].get_yaxis().get_offset_text().set_horizontalalignment("right")
        # # #
        #     #self.axes[i].yaxis.set_offset_position("right")
        #     #self.axes[i].yaxis.set_offset_position("left")
        #     #self.axes[i].set_ylabel(format_label(self._y_label, self._y_units))

    @property
    def artists(self):
        return self._artists

    def _get_default_xlabel(self, units: str = None):
        return self.measurements.axes_metadata[-1].format_label(units)

    def _get_default_ylabel(self, units: str = None):
        axes = LinearAxis(label=self.measurements.metadata.get("label", ""))
        return format_label(axes, units)

    def _get_default_xunits(self):
        return self.measurements.axes_metadata[-1].units

    def _get_default_yunits(self):
        return self.measurements.metadata.get("units", "")

    def set_xlim(self, xlim=None):
        extent = self.measurements._plot_extent(self._xunits)
        margin = (extent[1] - extent[0]) * 0.05
        if xlim is None:
            xlim = [-extent[0] - margin, extent[1] + margin]

        for i, measurement in self._generate_measurements():
            self.axes[i].set_xlim(xlim)
            artists = self.artists[i]
            for artist in artists:
                x = self._get_xdata()
                artist.set_xdata(x)

    def set_ylim(self, ylim=None):
        def _get_extent(measurements):
            min_value = measurements.min()
            max_value = measurements.max()
            margin = (max_value - min_value) * 0.05
            return [min_value - margin, max_value + margin]

        measurements = self._get_display_measurements()

        if self._common_scale and ylim is None:
            common_ylim = _get_extent(measurements)

        else:
            common_ylim = ylim

        for i, measurement in self._generate_measurements():

            if common_ylim is None:
                ylim = _get_extent(measurement)
            else:
                ylim = common_ylim

            self.axes[i].set_ylim(ylim)

    def set_legends(self, loc: str = "first", **kwargs):

        indices = [index for index in np.ndindex(*self.axes.shape)]

        if loc == "first":
            loc = indices[:1]
        elif loc == "last":
            loc = indices[-1:]
        elif loc == "all":
            loc = indices

        for i, _ in self._generate_measurements():

            if i in loc:
                self.axes[i].legend(**kwargs)

    def update_scale(self, min_value: float = None, max_value: float = None):
        min_value, max_value = self._get_default_scale_limits(min_value, max_value)
        self.set_ylim([min_value, max_value])
        # for _, measurement in self._generate_measurements(keepdims=False):
        #    self.set_ylim([measurement.min(), measurement.max()])

    def set_artists(self):

        artists = np.zeros(self.axes.shape, dtype=object)
        for i, measurement in self._generate_measurements(keepdims=False):
            ax = self.axes[i]
            x = self._get_xdata()
            new_lines = []
            for _, line_profile in measurement.generate_ensemble(keepdims=True):

                labels = []
                for axis in line_profile.ensemble_axes_metadata:
                    labels += [axis.format_title(".3f")]

                label = "-".join(labels)

                new_lines.append(
                    ax.plot(
                        x,
                        line_profile.array[(0,) * (len(line_profile.shape) - 1)],
                        label=label,
                    )[0]
                )
            artists.itemset(i, new_lines)

        self._artists = artists

    def _get_xdata(self):
        extent = self.measurements._plot_extent(self._xunits)
        return np.linspace(
            extent[0],
            extent[1],
            self.measurements.shape[-1],
            endpoint=False,
        )

    def update_artists(self):

        for i, measurements in self._generate_measurements(keepdims=False):
            lines = self._artists[i]
            for line, measurement in zip(lines, measurements):
                y = measurement.array
                x = self._get_xdata()
                line.set_data(x, y)

    @property
    def widgets(self):
        if widgets is None:
            raise ipywidgets_not_installed

        canvas = self.fig.canvas

        # def index_update_callback(change):
        #     if self._autoscale:
        #         vmin, vmax = self.get_global_vmin_vmax()
        #         self._update_vmin_vmax(vmin, vmax)

        (
            sliders,
            reset_button,
            continuous_update_button,
        ) = make_sliders_from_ensemble_axes(
            self,
            self.axes_types,
        )
        # power_scale_button = _make_power_scale_slider(self)
        scale_button = make_scale_button(self)
        autoscale_button = make_autoscale_button(self)
        # continuous_update_button = _make_continuous_update_button(sliders)

        # scale_button.layout = widgets.Layout(width="20%")
        # autoscale_button.layout = widgets.Layout(width="30%")
        # continuous_update_button.layout = widgets.Layout(width="50%")

        scale_box = widgets.VBox(
            [
                widgets.HBox([reset_button, continuous_update_button]),
                widgets.HBox([scale_button, autoscale_button]),
            ]
        )
        # scale_box.layout = widgets.Layout(width="300px")

        gui = widgets.VBox(
            [
                widgets.VBox(sliders),
                scale_box,
                # vmin_vmax_slider,
                # power_scale_button,
            ]
        )

        return widgets.HBox([gui, canvas])


class DiffractionSpotsVisualization(BaseMeasurementVisualization2D):
    """
    Display a diffraction pattern as indexed Bragg reflections.

    Parameters
    ----------
    measurements : IndexedDiffractionPattern
        Diffraction pattern to be displayed.
    scale : float
        Size of the circles representing the diffraction spots.
    ax : matplotlib.axes.Axes, optional
        If given the plots are added to the axis.
    Returns
    -------
    figure, axis_handle : matplotlib.figure.Figure, matplotlib.axis.Axis
    """

    def __init__(
        self,
        measurements: IndexedDiffractionPatterns,
        ax: Axes,
        cbar: bool = False,
        cmap: str = None,
        vmin: float = None,
        vmax: float = None,
        power: float = 1.0,
        scale: float = 0.1,
        common_scale: bool = False,
        explode: bool = False,
        figsize: tuple[float, float] = None,
        interact: bool = False,
    ):

        measurements = measurements.sort(criterion="distance")

        super().__init__(
            measurements,
            ax=ax,
            cbar=cbar,
            common_scale=common_scale,
            explode=explode,
            figsize=figsize,
            interact=interact,
        )

        # positions = measurements.positions[:, :2]

        self._scale = scale

        # (
        #     np.sqrt(np.min(squareform(distance_matrix(positions, positions)))) * scale
        # )

        if cmap is None:
            cmap = config.get("visualize.cmap", "viridis")

        self._normalization = None
        self._scale_normalization = None
        self._annotation_threshold = 0.0
        self._cmap = cmap
        self._size_bars = []
        self._miller_index_annotations = None

        self.set_normalization(power=power, vmin=vmin, vmax=vmax)
        self._set_artists()

        if cbar:
            self.set_cbars()
            self.set_scale_units()
            self.set_cbar_labels()

        self.set_xunits()
        self.set_yunits()
        self.set_xlabels()
        self.set_ylabels()

    def _get_positions(self, indexed_diffraction_spots):
        positions = indexed_diffraction_spots.positions[:, :2].copy()
        positions[:, 0] *= _get_conversion_factor(
            self._xunits, self._get_default_xunits()
        )
        positions[:, 1] *= _get_conversion_factor(
            self._yunits, self._get_default_yunits()
        )
        return positions

    def _get_radii(self, indexed_diffraction_spots, norm):
        conversion = _get_conversion_factor(self._xunits, self._get_default_xunits())
        radii = (
            norm(indexed_diffraction_spots.intensities) ** 0.5
            * self._scale
            * 0.5
            * conversion
        )
        radii = np.clip(radii, a_min=1e-3, a_max=1e3)
        return radii

    def _update_radii(self):
        for i, measurement in self._generate_measurements(keepdims=False):
            artists = self._artists[i]
            norm = self._normalization[i]
            scales = self._get_radii(measurement, norm)
            artists._widths = scales
            artists._heights = scales
            artists.set()

    def update_scale(self, min_value: float = None, max_value: float = None):
        super().update_scale(min_value=min_value, max_value=max_value)
        self._update_radii()

    def update_power(self, power: float = 1.0):
        super().update_power(power)
        self._update_radii()

    def _set_artists(self):

        if self._artists is not None:
            for artist in self.artists.ravel():
                artist.remove()

        self._artists = np.zeros(self.axes.shape, dtype=object)
        for i, measurement in self._generate_measurements(keepdims=False):
            ax = self.axes[i]

            norm = self._normalization[i]

            radii = self._get_radii(measurement, norm)
            positions = self._get_positions(measurement)

            if self._cmap not in plt.colormaps():
                cmap = ListedColormap([self._cmap])
            else:
                cmap = self._cmap

            ellipse_collection = EllipseCollection(
                widths=radii,
                heights=radii,
                angles=0.0,
                units="xy",
                array=measurement.intensities,
                cmap=cmap,
                offsets=positions,
                transOffset=ax.transData,
            )

            ellipse_collection.set_norm(norm)

            ax.add_collection(ellipse_collection)

            self._artists[i] = ellipse_collection

            ax.axis("equal")

    @property
    def _reciprocal_space_axes(self):
        return [
            ReciprocalSpaceAxis(
                label="kx", sampling=1.0, units="1/Å", _tex_label="$k_x$"
            ),
            ReciprocalSpaceAxis(
                label="ky", sampling=1.0, units="1/Å", _tex_label="$k_y$"
            ),
        ]

    def _get_default_xlabel(self, units: str = None):
        return self._reciprocal_space_axes[-2].format_label(units)

    def _get_default_ylabel(self, units: str = None):
        return self._reciprocal_space_axes[-1].format_label(units)

    def _get_default_xunits(self):
        return self._reciprocal_space_axes[-1].units

    def _get_default_yunits(self):
        return self._reciprocal_space_axes[-1].units

    def set_xlim(self):
        for i, measurement in self._generate_measurements():
            x_lim = np.abs(measurement.positions[:, 0]).max() * 1.1
            x_lim = (
                _get_conversion_factor(self._xunits, self._get_default_xunits()) * x_lim
            )
            self.axes[i].set_xlim([-x_lim, x_lim])

    def set_ylim(self):
        for i, measurement in self._generate_measurements():
            x_lim = np.abs(measurement.positions[:, 0]).max() * 1.1
            x_lim = (
                _get_conversion_factor(self._xunits, self._get_default_xunits()) * x_lim
            )
            self.axes[i].set_xlim([-x_lim, x_lim])

    def set_xunits(self, units: str = None):
        super().set_xunits(units)
        self._set_artists()

    def set_yunits(self, units: str = None):
        super().set_yunits(units)
        self._set_artists()

        # for i, measurement in self.iterate_measurements():
        #     artist = self.artists[i]
        #     positions = measurement.positions[:, :2].copy()
        #     positions[:, 0] *= _get_conversion_factor(self._x_units, self._get_default_x_units())
        #     positions[:, 1] *= _get_conversion_factor(self._y_units, self._get_default_y_units())
        #     artist.set(offsets=positions)

    def update_artists(self):
        for i, measurement in self._generate_measurements(keepdims=False):
            artists = self._artists[i]
            norm = self._normalization[i]

            radii = self._get_radii(measurement, norm)

            artists._widths = radii
            artists._heights = radii

            artists.set(array=measurement.intensities)
            self._set_hkl_visibility()

    def remove_miller_index_annotations(self):
        for annotation in self._miller_index_annotations:
            annotation.remove()
        self._miller_index_annotations = []

    def set_hkl_threshold(self, threshold):
        self._annotation_threshold = threshold
        self._set_hkl_visibility()

    def _set_hkl_visibility(self):
        if self._miller_index_annotations is None:
            self.set_miller_index_annotations()

        for i, measurement in self._generate_measurements(keepdims=False):
            visibility = measurement.intensities > self._annotation_threshold
            for annotation, visible in zip(self._miller_index_annotations, visibility):
                annotation.set_visible(visible)

    def set_miller_index_annotations(
        self,
        threshold: float = 1.0,
        size: int = 8,
        alignment: str = "top",
        **kwargs,
    ):

        self._annotation_threshold = threshold
        self._miller_index_annotations = []
        for i, measurement in self._generate_measurements(keepdims=False):
            ax = self.axes[i]
            norm = self._normalization[i]
            visibility = measurement.intensities > threshold
            positions = self._get_positions(measurement)
            radii = self._get_radii(measurement, norm)

            for hkl, position, visible, radius in zip(
                measurement.miller_indices, positions, visibility, radii
            ):
                if alignment == "top":
                    xy = position[:2] + [0, radius / 2]
                    va = "bottom"
                elif alignment == "center":
                    xy = position[:2]
                    va = "center"
                elif alignment == "bottom":
                    xy = position[:2] - [0, radius / 2]
                    va = "top"
                else:
                    raise ValueError()

                if config.get("visualize.use_tex"):
                    text = " \ ".join(
                        [f"\\bar{{{abs(i)}}}" if i < 0 else f"{i}" for i in hkl]
                    )
                    text = f"${text}$"
                else:
                    text = "{} {} {}".format(*hkl)

                annotation = ax.annotate(
                    text,
                    xy=xy,
                    ha="center",
                    va=va,
                    size=size,
                    visible=visible,
                    **kwargs,
                )
                self._miller_index_annotations.append(annotation)

    def pick_events(self):

        self._pick_annotations = {}
        for ax, artist in zip(np.array(self.axes).ravel(), self.artists.ravel()):
            artist.set_picker(True)
            annotation = ax.annotate(
                "",
                xy=(0, 0),
                xycoords="data",
                xytext=(20.0, 20.0),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
                visible=False,
            )
            self._pick_annotations[artist] = annotation

        def onpick(event):
            hkl = self.measurements.miller_indices[event.ind][0].tolist()
            position = self.measurements.positions[event.ind][0]
            intensity = event.artist.get_array()[event.ind].item()
            annotation = self._pick_annotations[event.artist]

            annotation.set_text(
                "\n".join(
                    (
                        f"hkl: {' '.join(map(str, hkl))}",
                        f"coordinate: {'{:.2f}, {:.2f}, {:.2f}'.format(*position.tolist())}",
                        f"intensity: {intensity:.4g}",
                    )
                )
            )
            annotation.xy = position[:2]
            annotation.set_visible(True)

        self.fig.canvas.mpl_connect("pick_event", onpick)

    @property
    def widgets(self):
        if widgets is None:
            raise ipywidgets_not_installed

        canvas = self.fig.canvas

        sliders = make_sliders_from_ensemble_axes(self, self.axes_types)

        def index_update_callback(change):
            if self._autoscale:
                vmin, vmax = self.get_global_vmin_vmax()
                self._update_vmin_vmax(vmin, vmax)

        _set_update_indices_callback(sliders, self, callbacks=(index_update_callback,))

        def hkl_slider_changed(change):
            self.set_hkl_threshold(change["new"])

        hkl_slider = widgets.FloatLogSlider(
            description="Index threshold", min=-10, max=0, value=1, step=1e-6
        )
        hkl_slider.observe(hkl_slider_changed, "value")

        power_scale_slider = make_power_scale_slider(self)
        scale_button = make_scale_button(self)
        autoscale_button = make_autoscale_button(self)
        continuous_update_button = _make_continuous_update_button(sliders)

        scale_button.layout = widgets.Layout(width="20%")
        autoscale_button.layout = widgets.Layout(width="30%")
        continuous_update_button.layout = widgets.Layout(width="50%")

        scale_box = widgets.VBox(
            [widgets.HBox([scale_button, autoscale_button, continuous_update_button])]
        )
        scale_box.layout = widgets.Layout(width="300px")

        gui = widgets.VBox(
            [
                widgets.VBox(sliders),
                scale_box,
                # vmin_vmax_slider,
                power_scale_slider,
                hkl_slider,
            ]
        )

        return widgets.HBox([gui, canvas])


def make_toggle_hkl_button(visualization):
    toggle_hkl_button = widgets.ToggleButton(description="Toggle hkl", value=False)

    def update_toggle_hkl_button(change):
        if change["new"]:
            visualization.set_miller_index_annotations()
        else:
            visualization.remove_miller_index_annotations()

    toggle_hkl_button.observe(update_toggle_hkl_button, "value")

    return toggle_hkl_button


_cube = np.array(
    [
        [[0, 0, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 1]],
        [[0, 0, 1], [1, 0, 1]],
        [[0, 1, 0], [1, 1, 0]],
        [[0, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [1, 1, 0]],
        [[1, 0, 0], [1, 0, 1]],
        [[0, 1, 1], [1, 1, 1]],
        [[1, 0, 1], [1, 1, 1]],
        [[1, 1, 0], [1, 1, 1]],
    ]
)


def _merge_columns(atoms: Atoms, plane, tol: float = 1e-7) -> Atoms:
    uniques, labels = np.unique(atoms.numbers, return_inverse=True)

    new_atoms = Atoms(cell=atoms.cell)
    for unique, indices in zip(uniques, label_to_index(labels)):
        positions = atoms.positions[indices]
        positions = _merge_positions(positions, plane, tol)
        numbers = np.full((len(positions),), unique)
        new_atoms += Atoms(positions=positions, numbers=numbers)

    return new_atoms


def _merge_positions(positions, plane, tol: float = 1e-7) -> np.ndarray:
    axes = plane_to_axes(plane)
    rounded_positions = tol * np.round(positions[:, axes[:2]] / tol)
    unique, labels = np.unique(rounded_positions, axis=0, return_inverse=True)

    new_positions = np.zeros((len(unique), 3))
    for i, label in enumerate(label_to_index(labels)):
        top_atom = np.argmax(-positions[label][:, axes[2]])
        new_positions[i] = positions[label][top_atom]
        # new_positions[i, axes[2]] = np.max(positions[label][top_atom, axes[2]])

    return new_positions


def show_atoms(
    atoms: Atoms,
    plane: tuple[float, float] | str = "xy",
    ax: Axes = None,
    scale: float = 0.75,
    title: str = None,
    numbering: bool = False,
    show_periodic: bool = False,
    figsize: tuple[float, float] = None,
    legend: bool = False,
    merge: float = 1e-2,
    tight_limits: bool = False,
    show_cell: bool = None,
    **kwargs,
):
    """
    Display 2D projection of atoms as a matplotlib plot.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms to be shown.
    plane : str, two float
        The projection plane given as a concatenation of 'x' 'y' and 'z', e.g. 'xy', or as two floats representing the
        azimuth and elevation angles of the viewing direction [degrees], e.g. (45, 45).
    ax : matplotlib.axes.Axes, optional
        If given the plots are added to the axes.
    scale : float
        Factor scaling their covalent radii for the atom display sizes (default is 0.75).
    title : str
        Title of the displayed image. Default is None.
    numbering : bool
        Display the index of the Atoms as a number. Default is False.
    show_periodic : bool
        If True, show the periodic images of the atoms at the cell boundary.
    figsize : two int, optional
        The figure size given as width and height in inches, passed to `matplotlib.pyplot.figure`.
    legend : bool
        If True, add a legend indicating the color of the atomic species.
    merge: float
        To speed up plotting large numbers of atoms, those closer than the given value [Å] are merged.
    tight_limits : bool
        If True the limits of the plot are adjusted
    kwargs : Keyword arguments for matplotlib.collections.PatchCollection.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    if show_periodic:
        atoms = atoms.copy()
        atoms = pad_atoms(atoms, margins=1e-3)

    if merge > 0.0:
        atoms = _merge_columns(atoms, plane, merge)

    if tight_limits and show_cell is None:
        show_cell = False
    elif show_cell is None:
        show_cell = True

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    cell = atoms.cell
    axes = plane_to_axes(plane)

    cell_lines = np.array(
        [[np.dot(line[0], cell), np.dot(line[1], cell)] for line in _cube]
    )
    cell_lines_x, cell_lines_y = cell_lines[..., axes[0]], cell_lines[..., axes[1]]

    if show_cell:
        for cell_line_x, cell_line_y in zip(cell_lines_x, cell_lines_y):
            ax.plot(cell_line_x, cell_line_y, "k-")

    if len(atoms) > 0:
        positions = atoms.positions[:, axes[:2]]
        order = np.argsort(-atoms.positions[:, axes[2]])
        positions = positions[order]

        colors = jmol_colors[atoms.numbers[order]]
        sizes = covalent_radii[atoms.numbers[order]] * scale

        circles = []
        for position, size in zip(positions, sizes):
            circles.append(Circle(position, size))

        coll = PatchCollection(circles, facecolors=colors, edgecolors="black", **kwargs)
        ax.add_collection(coll)

        ax.axis("equal")
        ax.set_xlabel(plane[0] + " [Å]")
        ax.set_ylabel(plane[1] + " [Å]")

        ax.set_title(title)

        if numbering:
            if merge:
                raise ValueError("atom numbering requires 'merge' to be False")

            for i, (position, size) in enumerate(zip(positions, sizes)):
                ax.annotate(
                    "{}".format(order[i]), xy=position, ha="center", va="center"
                )

    if legend:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markeredgecolor="k",
                label=chemical_symbols[unique],
                markerfacecolor=jmol_colors[unique],
                markersize=12,
            )
            for unique in np.unique(atoms.numbers)
        ]

        ax.legend(handles=legend_elements, loc="upper right")

    if tight_limits:
        ax.set_adjustable("box")
        ax.set_xlim([np.min(cell_lines_x), np.max(cell_lines_x)])
        ax.set_ylim([np.min(cell_lines_y), np.max(cell_lines_y)])

    return fig, ax
