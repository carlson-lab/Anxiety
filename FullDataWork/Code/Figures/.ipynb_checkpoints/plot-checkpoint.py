# ----------------------------------------------------------------------------------------------------------------------
# FILE DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------

# File:  plot.py
# Author:  anonymous
# Date written:  01-18-2022
# Last modified:  10-25-2023

r"""
Description:  
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT STATEMENTS
# ----------------------------------------------------------------------------------------------------------------------

# Import statements
import numpy as np
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from matplotlib.patches import Polygon
import seaborn as sns

# Constants
R1 = 1.0  # inner radius of power plots


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# Chord plot
def chord_plot(
    x,
    rois=None,
    freqs=None,
    freq_ticks=None,
    max_alpha=0.7,
    buffer_percent=1.0,
    outer_radius=1.2,
    min_max_quantiles=(0.5, 0.9),
    color=None,
    cmap=None,
    roi_fontsize=13,
    roi_extent=0.28,
    tick_extent=0.03,
    tick_label_extent=0.11,
    tick_label_fontsize=10.0,
    fontfamily='sans-serif',
    figsize=(7, 7)):
    r"""
    
    Parameters
    ----------
    
    Returns
    -------
    """
    
    # Check arguments
    assert x.ndim == 3
    assert x.shape[0] == x.shape[1]
    assert max_alpha >= 0.0 and max_alpha <= 1.0, f"{max_alpha}"
    n_roi, n_freq = x.shape[1:]
    assert freqs is None or len(freqs) == n_freq, f"{len(freqs)} != {n_freq}"
    
    # Replace ROI underscores with spaces
    if rois is not None:
        assert len(rois) == n_roi, f"{len(rois)} != {n_roi}"
        pretty_rois = [roi.replace("_", " ") for roi in rois]
    
    # Default color
    if color is None and cmap is None:
        color = 'tab:blue'
    
    # Set color to None if color map is provided
    if cmap is not None:
        color = None
    
    #  variables
    r2 = outer_radius
    center_angles = np.linspace(0, 2 * np.pi, n_roi + 1)
    buffer = buffer_percent / 100.0 * 2.0 * np.pi
    start_angles = center_angles[:-1] + buffer
    stop_angles = center_angles[1:] - buffer
    freq_diff = (stop_angles[0] - start_angles[0]) / (n_freq + 1)
    min_val, max_val = np.quantile(x, min_max_quantiles)
    x = max_alpha * np.clip((x - min_val) / (max_val - min_val), 0.0, 1.0)

    # Set up axes and labels and ticks
    _, ax = _set_up_chord_plot(
        start_angles=start_angles,
        stop_angles=stop_angles,
        r1=R1,
        r2=r2,
        pretty_rois=pretty_rois,
        freqs=freqs,
        freq_ticks=freq_ticks,
        tick_extent=tick_extent,
        tick_label_extent=tick_label_extent,
        tick_label_fontsize=tick_label_fontsize,
        roi_fontsize=roi_fontsize,
        roi_extent=roi_extent,
        fontfamily=fontfamily,
        figsize=figsize)

    # Add the power and chord plots
    _update_chord_plot(
        x=x,
        ax=ax,
        start_angles=start_angles,
        stop_angles=stop_angles,
        freq_diff=freq_diff,
        outer_radius=outer_radius,
        color=color,
        cmap=cmap)
    
    # Return plot axis
    return ax


# Update chord plot
def _update_chord_plot(
    x,
    ax,
    start_angles,
    stop_angles,
    freq_diff,
    outer_radius,
    color,
    cmap=None):
    r"""
    
    Parameters
    ----------
    
    Returns
    -------
    """
    
    # Initialize variables
    r2 = outer_radius
    handles = []
    n_roi, n_freq = x.shape[1:]
    
    # Create colormap array for different frequencies
    if cmap is not None:
        if isinstance(cmap, str):
            cmap_idx = np.linspace(0, 1, n_freq)
            # cmap_color_arr = mpl.colormaps[cmap](cmap_idx)[:, :3]
            cmap_color_arr = mpl.cm.get_cmap(cmap)(cmap_idx)[:, :3]
        elif isinstance(cmap, LinearSegmentedColormap):
            cmap_idx = np.linspace(0, 1, n_freq)
            cmap_color_arr = cmap(cmap_idx)[:, :3]
        else:
            cmap_color_arr = None
            color = 'b'
    else:
        cmap_color_arr = None
    
    # Draw the power plots
    for i, (c1, c2) in enumerate(zip(start_angles, stop_angles)):
        # Iterate over frequencies
        for j in range(n_freq):
            # Retrieve colormap color for corresponding frequency value
            if cmap_color_arr is not None:
                color = cmap_color_arr[j]
            
            # Generate arc power patch
            if x[i, i, j] > 0:
                # Arc power patch rotation
                diff1 = j * (c2 - c1) / n_freq
                diff2 = (j + 1) * (c2 - c1) / n_freq
                
                # Transparency value
                alpha = x[i, i, j]
                
                # Append arc power patch to handles
                h = _arc_patch(
                    r1=R1,
                    r2=r2,
                    theta1=c1 + diff1,
                    theta2=c1 + diff2,
                    ax=ax,
                    color=color,
                    cmap=cmap,
                    n=5,
                    alpha=alpha)
                handles.append(h)
    
    # Draw the chords to represent cross-power
    for i in range(n_roi - 1):
        for j in range(i + 1, n_roi):
            # Iterate over frequency values
            for k in range(n_freq):
                # Frequency-specific color map color
                if cmap_color_arr is not None:
                    color = cmap_color_arr[k]
                
                # Generate chord connection
                if x[i, j, k] > 0.0:
                    # Chord connection rotation
                    theta1 = start_angles[i] + freq_diff * k
                    theta2 = start_angles[j] + freq_diff * k
                    
                    # Transparency value
                    alpha = x[i, j, k]
                    
                    # Append chord connection polygon to handles
                    h = _plot_poly_chord(
                        theta1=theta1,
                        theta2=theta2,
                        diff=freq_diff,
                        ax=ax,
                        color=color,
                        alpha=alpha)
                    handles.append(h)
    
    # Return collection of power arc and chord connection handles
    return handles


# Set up chord plot
def _set_up_chord_plot(
    start_angles,
    stop_angles,
    r1,
    r2,
    pretty_rois,
    freqs,
    freq_ticks,
    tick_extent,
    tick_label_extent,
    tick_label_fontsize,
    roi_fontsize,
    roi_extent,
    fontfamily,
    figsize):
    r"""
    
    Parameters
    ----------
    
    Returns
    -------
    """
    
    # Initialize figure
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Set up axes and draw power plots
    for i, (c1, c2) in enumerate(zip(start_angles, stop_angles)):
        # Draw power axis
        _draw_power_axis(
            r1=r1,
            r2=r2,
            theta1=c1,
            theta2=c2,
            ax=ax)
        
        # Plot ticks
        if freqs is not None and freq_ticks is not None:
            _plot_ticks(
                r=r2,
                theta1=c1,
                theta2=c2, 
                ax=ax,
                freqs=freqs,
                freq_ticks=freq_ticks,
                tick_extent=tick_extent,
                tick_label_extent=tick_label_extent,
                tick_label_fontsize=tick_label_fontsize,
                fontfamily=fontfamily)
        
        # Annotate ROIs
        if pretty_rois is not None:
            _plot_roi_name(
                r=r2,
                theta=0.5 * (c1 + c2),
                ax=ax,
                roi=pretty_rois[i],
                extent=roi_extent,
                fontsize=roi_fontsize,
                fontfamily=fontfamily)
    
    # Axis limits
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-1.5, 1.5)
    plt.axis("off")
    
    # Return figure and axis variables
    return fig, ax


# Plot poly chord
def _plot_poly_chord(
    theta1,
    theta2,
    diff,
    ax,
    color,
    n=50,
    alpha=0.5):
    r"""
    
    Parameters
    ----------
    
    Returns
    -------
    """
    
    # Chord chonnection points
    points1 = _chord_helper(theta1, theta2, n=n)
    rot_mat = np.array([[np.cos(diff), -np.sin(diff)], [np.sin(diff), np.cos(diff)]])
    points2 = rot_mat @ points1
    points = np.concatenate([points1, points2[:, ::-1]], axis=1).T
    
    # Chord connection polygon
    poly = Polygon(points, closed=True, fc=to_rgba(c=color, alpha=alpha))
    ax.add_patch(poly)
    
    # Return chord connection polygon
    return poly


# Chord helper
def _chord_helper(theta1, theta2, n=50):
    r"""
    
    Parameters
    ----------
    
    Returns
    -------
    """
    
    # Chord helper coordinates calculations
    a1, a2 = np.cos(theta1), np.sin(theta1)
    b1, b2 = np.cos(theta2), np.sin(theta2)
    denom = a1 * b2 - a2 * b1
    if np.abs(denom) < 1e-5:
        xs = np.linspace(a1, b1, n)
        ys = np.linspace(a2, b2, n)
        
        return np.vstack([xs, ys])
    v, w = 2.0 * (a2 - b2) / denom, 2.0 * (b1 - a1) / denom
    center = (-v / 2.0, -w / 2.0)
    radius = np.sqrt(((v ** 2.0 + w ** 2.0) / 4.0) - 1.0)
    angle1 = np.arctan2(a2 - center[1], a1 - center[0])
    angle2 = np.arctan2(b2 - center[1], b1 - center[0])
    angle1, angle2 = min(angle1, angle2), max(angle1, angle2)
    if angle2 - angle1 > np.pi:
        angle1, angle2 = angle2, angle1 + 2 * np.pi
    theta = np.linspace(angle1, angle2, n)
    xs = radius * np.cos(theta) + center[0]
    ys = radius * np.sin(theta) + center[1]
    
    # Return coordinates
    return np.vstack([xs, ys])


# Arc patch
def _arc_patch(
    r1,
    r2,
    theta1,
    theta2,
    ax,
    color,
    cmap=None,
    n=50,
    alpha=1.0,
    **kwargs):
    r"""
    
    Parameters
    ----------
    
    Returns
    -------
    """
    
    # Power arc points
    thetas = np.linspace(theta1, theta2, n)
    sin_thetas, cos_thetas = np.sin(thetas), np.cos(thetas)
    points = np.vstack([cos_thetas, sin_thetas]).T
    points = np.concatenate([r1 * points, r2 * points[::-1]], axis=0)
    
    # Power arc polygon
    poly = Polygon(
        points,
        closed=True,
        fc=to_rgba(color, alpha=alpha),
        **kwargs)
    ax.add_patch(poly)
    
    # Return power arc polygon
    return poly


# Draw power axis
def _draw_power_axis(
    r1,
    r2,
    theta1,
    theta2,
    ax,
    n=50,
    **kwargs):
    r"""
    
    Parameters
    ----------
    
    Returns
    -------
    """
    
    # Power axis points
    thetas = np.linspace(theta1, theta2, n)
    sin_thetas, cos_thetas = np.sin(thetas), np.cos(thetas)
    points = np.vstack([cos_thetas, sin_thetas]).T
    points = np.concatenate([r1 * points, r2 * points[::-1]], axis=0)
    points = np.concatenate([points, points[:1]], axis=0)
    
    # Power axis handle
    handle = ax.plot(points[:, 0], points[:, 1], c='k', **kwargs)
    
    # Return handle
    return handle


# Plot ticks
def _plot_ticks(
    r,
    theta1,
    theta2,
    ax,
    freqs,
    freq_ticks,
    tick_extent=0.03,
    tick_label_extent=0.11,
    tick_label_fontsize=10.0,
    n=5,
    fontfamily='sans-serif',
    **kwargs):
    r"""
    
    Parameters
    ----------
    
    Returns
    -------
    """
    
    # Tick offset
    offset = 0.0 if np.cos((theta1 + theta2) / 2.0) > 0.0 else 180.0
    
    # Iterate over frequency tick values
    for freq in freq_ticks:
        # Tick location and rotation
        theta = theta1 + (theta2 - theta1) * (freq - freqs[0]) / (freqs[-1] - freqs[0])
        x = [r * np.cos(theta), (r + tick_extent) * np.cos(theta)]
        y = [r * np.sin(theta), (r + tick_extent) * np.sin(theta)]
        
        # Plot tick
        ax.plot(x, y, c="k", **kwargs)
        
        # Tick label location and rotation
        x = (r + tick_label_extent) * np.cos(theta)
        y = (r + tick_label_extent) * np.sin(theta)
        rotation = (theta * 180.0 / np.pi) + offset
        
        # Tick text / label
        ax.text(
            x=x,
            y=y,
            s=str(freq),
            rotation=rotation,
            fontfamily=fontfamily,
            fontsize=tick_label_fontsize,
            ha='center',
            va='center')


# Plot ROI name
def _plot_roi_name(
    r,
    theta,
    ax,
    roi,
    extent=0.3,
    fontsize=13,
    fontfamily='sans-serif'):
    r"""
    
    Parameters
    ----------
    
    Returns
    -------
    """
    
    # ROI location and rotation
    x, y = (r + extent) * np.cos(theta), (r + extent) * np.sin(theta)
    rotation = (theta * 180.0 / np.pi) - 90.0
    
    # Offset rotation
    if np.sin(theta) < 0.0:
        rotation += 180.0
    
    # ROI text
    ax.text(
        x=x,
        y=y,
        s=roi,
        rotation=rotation,
        ha='center',
        va='center',
        fontfamily=fontfamily,
        fontsize=fontsize)


