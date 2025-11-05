# UTILS AND COLORMAP

from django.http import JsonResponse, HttpResponse
from PIL import Image
from django.core.cache import cache
import io
import matplotlib.cm as cm
import xarray as xr
import numpy as np
from ..services.interactive_map_service import get_cached_zarr_path
from functools import lru_cache
from django.core.cache import cache
import hashlib
from pyproj import Transformer
TRANSFORMER_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def get_cache_key(validation_id, var_name):
    """Generate consistent cache keys"""
    return f"validation_{validation_id}_{var_name}"

def compute_norm_params(validation_id, zarr_path, variable_name):
    #TODO: integrate with global limits from qa4sm-reader.globals.py - guess it makes sense to calculate min max for vars witout limit
    """Compute normalization parameters for a specific variable"""
    cache_key = get_cache_key(validation_id, variable_name)
    cached_params = cache.get(cache_key)
    
    if cached_params:
    #if False:
        return cached_params
    
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
        
        # Select tsw='bulk' if needed
        if 'tsw' in ds.dims:
            ds = ds.sel(tsw='bulk')
        
        var_data = ds[variable_name].values
        vmin = float(np.nanmin(var_data))
        vmax = float(np.nanmax(var_data))
        
        # Cache for 24 hours
        cache.set(cache_key, (vmin, vmax), 86400)
        return vmin, vmax
    
    except Exception as e:
        print(f"Error computing normalization parameters: {e}")
        return 0.0, 1.0

def create_transparent_tile():
    """Create a transparent 256x256 PNG tile"""
    # Create a transparent image
    img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))

    # Save to buffer
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return HttpResponse(img_buffer.getvalue(), content_type='image/png')

def get_transparent_tile():
    """Get cached transparent tile bytes"""
    cache_key = 'transparent_tile_png'
    tile_bytes = cache.get(cache_key)
    
    if not tile_bytes:
        img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        tile_bytes = img_buffer.getvalue()
        cache.set(cache_key, tile_bytes, timeout=None)  # Cache forever
    
    return HttpResponse(tile_bytes, content_type='image/png')


from validator.validation.globals import QR_COLORMAPS, QR_STATUS_DICT
def get_colormap(metric_name):
    """Get cached colormap for a specific metric"""
    cache_key = f"colormap_{metric_name}"
    cached_colormap = cache.get(cache_key)
    
    if cached_colormap:
        return cached_colormap
    
    try:
        # Get the matplotlib colormap from QR_COLORMAPS
        if metric_name in QR_COLORMAPS:
            mpl_colormap = QR_COLORMAPS[metric_name]
        else:
            # Fallback to a default colormap if metric not found
            mpl_colormap = cm.get_cmap('viridis')  # or whatever default you prefer
        
        # Cache for 24 hours (colormaps don't change)
        cache.set(cache_key, mpl_colormap, 86400)
        return mpl_colormap
    
    except Exception as e:
        print(f"Error getting colormap for metric {metric_name}: {e}")
        # Return a safe default colormap
        return cm.get_cmap('viridis')
    

def get_colormap_type(mpl_colormap):
    """Get information about the colormap type and properties"""
    if hasattr(mpl_colormap, 'colors') and hasattr(mpl_colormap, 'N'):
        return {
            'type': 'ListedColormap',
            'num_colors': mpl_colormap.N,
            'colors': mpl_colormap.colors
        }
    else:
        return {
            'type': 'LinearSegmentedColormap',
            'name': getattr(mpl_colormap, 'name', 'unknown')
        }


def generate_css_gradient(mpl_colormap, metric_name=None, num_colors=10):
    """Convert matplotlib colormap to CSS gradient string"""
    try:
        # Get colormap information
        colormap_info = get_colormap_type(mpl_colormap)
        
        if colormap_info['type'] == 'ListedColormap':
            # For ListedColormap, use the actual discrete colors
            colors = []
            actual_colors = colormap_info['colors']
            
            for rgba in actual_colors:
                # Convert to CSS rgba format
                css_color = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3] if len(rgba) > 3 else 1.0})"
                colors.append(css_color)
            
            # For categorical data like 'status', create equal segments
            if metric_name == 'status' and len(colors) > 1:
                # Create discrete segments with equal width for categorical data
                segment_width = 100 / len(colors)
                gradient_parts = []
                
                for i, color in enumerate(colors):
                    start_pos = i * segment_width
                    end_pos = (i + 1) * segment_width
                    
                    # Create sharp transitions between categories
                    gradient_parts.append(f"{color} {start_pos}%")
                    gradient_parts.append(f"{color} {end_pos}%")
                
                gradient = f"linear-gradient(to right, {', '.join(gradient_parts)})"
            elif len(colors) > 1:
                # For other discrete colormaps, still create segments but maybe smoother
                segment_width = 100 / len(colors)
                gradient_parts = []
                
                for i, color in enumerate(colors):
                    start_pos = i * segment_width
                    end_pos = (i + 1) * segment_width
                    
                    if i == 0:
                        gradient_parts.append(f"{color} {start_pos}%")
                    
                    gradient_parts.append(f"{color} {end_pos}%")
                
                gradient = f"linear-gradient(to right, {', '.join(gradient_parts)})"
            else:
                # Single color case
                gradient = f"linear-gradient(to right, {colors[0]}, {colors[0]})"
                
        else:
            # For LinearSegmentedColormap, sample colors continuously
            colors = []
            for i in range(num_colors):
                # Sample from 0 to 1
                normalized_pos = i / (num_colors - 1)
                rgba = mpl_colormap(normalized_pos)
                # Convert to CSS rgba format
                css_color = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3] if len(rgba) > 3 else 1.0})"
                colors.append(css_color)
            
            # Create smooth CSS linear gradient
            gradient = f"linear-gradient(to right, {', '.join(colors)})"
        
        return gradient
    
    except Exception as e:
        print(f"Error generating CSS gradient: {e}")
        return "linear-gradient(to right, blue, cyan, yellow, red)"  # fallback
    

def get_colormap_metadata(metric_name):
    """Get static colormap metadata (gradient, type, categories) - NO file I/O"""
    cache_key = f"colormap_metadata_{metric_name}"
    cached = cache.get(cache_key)

    if cached:
        return cached

    # Get the colormap for this metric
    colormap = get_colormap(metric_name)
    colormap_info = get_colormap_type(colormap)

    # Generate CSS gradient colors from the matplotlib colormap
    gradient_colors = generate_css_gradient(colormap, metric_name)

    result = {
        'gradient': gradient_colors,
        'is_categorical': metric_name == 'status',  # Adjust based on your logic
        'colormap_info': colormap_info
    }

    # Add category definitions for status
    if metric_name == 'status':
        result['categories'] = QR_STATUS_DICT

    # Cache indefinitely - this never changes
    cache.set(cache_key, result, None)

    return result



def get_layernames(zarr_path):
    """Get information about all variables in the Zarr dataset"""
    cache_key = f"dataset_info_{zarr_path}"
    cached_info = cache.get(cache_key)
    
    #if cached_info:
    #    return cached_info
    
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
        
        excluded = {'tsw', 'lon', 'lat', 'idx', 'gpi'}
        descriptions = {}
        
        for var_name in ds.data_vars:
            if var_name not in excluded:
                descriptions[var_name] = ds[var_name].attrs.get('long_name', var_name)
        
        # Cache for 1 hour
        cache.set(cache_key, descriptions, 3600)
        return descriptions
    
    except Exception as e:
        print(f"Error reading zarr dataset info: {e}")
        return {}

def get_status_legend_data(geotiff_path, variable_name):
    pass


def get_colorbar_data(request, validation_id, metric_name, var_name):
    """Get all colorbar data: colormap, min/max values, and gradient colors"""

    zarr_path = get_cached_zarr_path(validation_id)

    # Get the colormap for this metric
    colormap = get_colormap(metric_name)

    vmin, vmax = compute_norm_params(validation_id, zarr_path, var_name)

    # Generate CSS gradient colors from the matplotlib colormap
    gradient_colors = generate_css_gradient(colormap, metric_name)

    return {
        'vmin': vmin,
        'vmax': vmax,
        'gradient': gradient_colors,
        'metric_name': metric_name,
        'is_categorical': False
    }

from django.core.cache import cache


def calculate_spread(zoom_level, plot_resolution):
    """
    Calculate spread pixels based on zoom level and minimum distance.

    Args:
        zoom_level: Zoom level (1-11)
        plot_resolution: Minimum distance (plot_resolution from dataset)

    Returns:
        int: Number of pixels to spread
    """
    if plot_resolution is None:
        plot_resolution = 0.1  # Default fallback value

    return int((plot_resolution + 10) * zoom_level ** 2.2 / 86)


def get_spread_and_buffer(validation_id, zoom_level, plot_resolution):
    """
    Get cached spread_px and buffer_px for a given validation, zoom level, and resolution.

    Args:
        validation_id: Validation run identifier
        zoom_level: Zoom level (1-11)
        plot_resolution: Minimum distance from dataset (can be None)

    Returns:
        tuple: (spread_px, buffer_px)
    """
    # Use plot_resolution in cache key, handle None case
    resolution_key = plot_resolution if plot_resolution is not None else 'none'
    cache_key = f'spread_buffer:{validation_id}:{zoom_level}:{resolution_key}'

    result = cache.get(cache_key)

    if result is None:
        spread_px = calculate_spread(zoom_level, plot_resolution)
        buffer_px = spread_px + 1
        result = (spread_px, buffer_px)

        # Cache for 1 hour
        cache.set(cache_key, result, 3600)

    return result




# 1. In-memory LRU cache for DataFrames (most recent validations)
@lru_cache(maxsize=10)  # Keep last 10 validation_id + var_name combos in memory
def get_cached_dataframe(validation_id, var_name, zarr_path):
    """
    Load and cache DataFrame in memory.
    Cache key is based on validation_id and var_name.
    """
    ds_zarr = xr.open_zarr(zarr_path)

    if var_name not in ds_zarr.data_vars:
        return None

    da = ds_zarr[var_name]
    df = da.to_dataframe(name='value').reset_index()
    df = df.dropna(subset=['value'])

    print(f"Loaded DataFrame for {validation_id}/{var_name}: {len(df)} points, {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


# 2. Spatial indexing for faster filtering
def create_spatial_index(df, coord_cols=('lon', 'lat')):
    """
    Create a simple spatial index using binning.
    This helps quickly filter points for a given tile.
    """
    from scipy.spatial import cKDTree

    # Create KD-tree for spatial queries
    coords = df[list(coord_cols)].values
    tree = cKDTree(coords)

    return tree, coords


@lru_cache(maxsize=10)
def get_cached_dataframe_with_index(validation_id, var_name, zarr_path, projection):
    """
    Load DataFrame and create spatial index.
    """
    ds_zarr = xr.open_zarr(zarr_path)

    if var_name not in ds_zarr.data_vars:
        return None, None, None

    da = ds_zarr[var_name]
    df = da.to_dataframe(name='value').reset_index()
    df = df.dropna(subset=['value'])

    # Transform if needed
    if projection == 3857:
        df = df.copy()
        df['x'], df['y'] = TRANSFORMER_TO_3857.transform(
            df['lon'].values, 
            df['lat'].values
        )
        coord_cols = ('x', 'y')
    else:
        coord_cols = ('lon', 'lat')

    # Create spatial index
    tree, coords = create_spatial_index(df, coord_cols)

    print(f"Loaded DataFrame with spatial index for {validation_id}/{var_name}: {len(df)} points")

    return df, tree, coord_cols