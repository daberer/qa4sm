# interactive_map_view.py
import os
import numpy as np
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.cache import cache
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from .interactive_map_utils import get_transparent_tile, compute_norm_params, get_layernames, get_colormap_metadata, get_spread_and_buffer, get_colormap, get_cached_dataframe_with_index
import io
from pyproj import Transformer
from PIL import Image

import json

import xarray as xr
import morecantile
import datashader as ds
import datashader.transfer_functions as tf
import zarr
from ..services.interactive_map_service import get_cached_zarr_path, get_plot_resolution



# Cache TMS and transformer objects 
TMS_4326 = morecantile.tms.get("WorldCRS84Quad")  # EPSG:4326
TMS_3857 = morecantile.tms.get("WebMercatorQuad")  # EPSG:3857
TRANSFORMER_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


@require_http_methods(["GET"])
def get_pixel_value(request):
    return 1


# LEGEND 


    # """Get status legend data by reading unique values from the GeoTIFF"""
    # cache_key = f"status_legend_{geotiff_path}_{var_name}"
    # cached_data = cache.get(cache_key)

    # if cached_data:
    #     return cached_data

    # try:
    #     with rasterio.open(geotiff_path) as dataset:
    #         # Read the band data
    #         band_data = dataset.read(band_index, masked=True)

    #         # Get unique values, excluding NaN/masked values
    #         unique_values = np.unique(band_data.compressed())
    #         unique_values = unique_values[~np.isnan(unique_values)]
    #         unique_values = sorted([int(val) for val in unique_values if not np.isnan(val)])

    #     # Get the colormap for status
    #     status_colormap = get_colormap('status')
    #     colormap_info = get_colormap_type(status_colormap)

    #     # Create legend entries for each unique status value found in the data
    #     legend_entries = []

    #     for status_value in unique_values:
    #         if status_value in QR_STATUS_DICT:
    #             # Map status value to colormap index (shift by 1: -1→0, 0→1, etc.)
    #             colormap_index = status_value + 1

    #             if colormap_info['type'] == 'ListedColormap' and colormap_index < len(colormap_info['colors']):
    #                 rgba = colormap_info['colors'][colormap_index]
    #                 # Convert to hex color for frontend
    #                 hex_color = f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"

    #                 legend_entries.append({
    #                     'value': status_value,
    #                     'label': QR_STATUS_DICT[status_value],
    #                     'color': hex_color,
    #                     'rgba': rgba
    #                 })

    #     legend_data = {
    #         'type': 'categorical',
    #         'entries': legend_entries,
    #         'total_categories': len(QR_STATUS_DICT)
    #     }

    #     # Cache for 1 hour
    #     cache.set(cache_key, legend_data, 3600)
    #     return legend_data

    # except Exception as e:
    #     print(f"Error getting status legend data: {e}")
    #     return {
    #         'type': 'categorical',
    #         'entries': [],
    #         'total_categories': len(QR_STATUS_DICT)
    #     }


# API endpoints


@require_http_methods(["GET"])
def get_layer_bounds(request, validation_id):
    """Get geographic bounds from Zarr lat/lon arrays (same for all metrics)"""
    
    # Cache key only uses validation_id since bounds are the same for all metrics
    bounds_cache_key = f"bounds_{validation_id}"
    bounds_data = cache.get(bounds_cache_key)
    
    if not bounds_data:
    #if True:
        try:
            zarr_path = get_cached_zarr_path(validation_id)
            
            # Open the Zarr store
            store = zarr.open(zarr_path, mode='r')
            
            # Read lat/lon arrays
            if 'lat' not in store or 'lon' not in store:
                return JsonResponse(
                    {"error": "lat/lon arrays not found in Zarr store"}, 
                    status=404
                )
            
            lat = store['lat'][:]
            lon = store['lon'][:]
            
            # Filter out fill values (NaN) if present
            lat_valid = lat[~np.isnan(lat)]
            lon_valid = lon[~np.isnan(lon)]
            
            if len(lat_valid) == 0 or len(lon_valid) == 0:
                return JsonResponse(
                    {"error": "No valid coordinates found"}, 
                    status=404
                )
            
            # Calculate bounds
            min_lon = float(np.min(lon_valid))
            max_lon = float(np.max(lon_valid))
            min_lat = float(np.min(lat_valid))
            max_lat = float(np.max(lat_valid))
            
            # Add a small buffer (1% of the range) to ensure all points are visible
            lon_buffer = (max_lon - min_lon) * 0.01 or 0.1
            lat_buffer = (max_lat - min_lat) * 0.01 or 0.1
            
            bounds_data = {
                'extent': [
                    min_lon - lon_buffer,
                    min_lat - lat_buffer,
                    max_lon + lon_buffer,
                    max_lat + lat_buffer
                ],
                'center': [
                    (min_lon + max_lon) / 2,
                    (min_lat + max_lat) / 2
                ],
                'num_points': len(lat_valid)
            }
            
            # Cache for longer (24 hours) since this never changes for a validation
            cache.set(bounds_cache_key, bounds_data, timeout=86400)
            
        except Exception as e:
            return JsonResponse(
                {"error": f"Error reading Zarr bounds: {str(e)}"}, 
                status=500
            )
    
    return JsonResponse(bounds_data)

@require_http_methods(["GET"])
def get_layer_range(request, validation_id, metric_name, var_name):
    """Get vmin/vmax for a specific layer by var_name (lazy-loaded)"""

    # Cache key for this specific layer's range
    range_cache_key = f"range_{validation_id}_{metric_name}_{var_name}"
    range_data = cache.get(range_cache_key)

    if not range_data:
    #if True:
        zarr_path = get_cached_zarr_path(validation_id)

        # Special handling for categorical data
        if metric_name == 'status':
            # Get metadata to access colormap info
            metadata_cache_key = f"layer_metadata_{validation_id}"
            metadata = cache.get(metadata_cache_key)

            #if metadata:
            if True:
                colormap_metadata = metadata.get('colormap_metadata', {}).get(metric_name, {})
                colormap_info = colormap_metadata.get('colormap_info', {})

                if colormap_info.get('type') == 'ListedColormap':
                    vmin = 0
                    vmax = colormap_info.get('num_colors', 1) - 1
                else:
                    vmin, vmax = compute_norm_params(
                        validation_id, 
                        zarr_path, 
                        var_name
                    )
            else:
                # Fallback if metadata not cached
                vmin, vmax = compute_norm_params(
                    validation_id, 
                    zarr_path, 
                    var_name
                )

            range_data = {
                'vmin': vmin,
                'vmax': vmax
            }
        else:
            # Continuous data - compute from GeoTIFF
            vmin, vmax = compute_norm_params(
                validation_id, 
                zarr_path, 
                var_name
            )

            range_data = {
                'vmin': vmin,
                'vmax': vmax
            }

        # Cache for 2 hours (longer than metadata)
        cache.set(range_cache_key, range_data, timeout=7200)

    return JsonResponse(range_data)




@require_http_methods(["POST"])
def get_validation_layer_metadata(request, validation_id):
    """Fetch metadata including layer mappings, gradients, and status legends (NO vmin/vmax)"""
    cache_key = f"layer_metadata_{validation_id}"
    metadata = cache.get(cache_key)

    if not metadata:
    #if True:
        data = json.loads(request.body)
        possible_layers = data.get('possible_layers', {})

        zarr_path = get_cached_zarr_path(validation_id)
        available_layers_mapping = get_layernames(zarr_path)

        # Simplified: Just use variable names
        layers = []
        status_metadata = {}

        for metric, possible_layer_list in possible_layers.items():
            for possible_layer_name in possible_layer_list:
                # Check if layer exists in available layers
                if possible_layer_name in available_layers_mapping.values():
                    # Get colormap info for this layer
                    colormap_info = get_colormap_metadata(metric)

                    # Add layer to list using variable name
                    layers.append({
                        'name': possible_layer_name,
                        'metric': metric,
                        'colormap': colormap_info
                    })

                    #TODO: handle status
                    # # Handle status legends using variable name
                    # if metric == 'status':
                    #     status_legend = get_status_legend_data(
                    #         zarr_path,
                    #         possible_layer_name  # Now using variable name
                    #     )
                    #     status_metadata[possible_layer_name] = status_legend

        metadata = {
            'validation_id': validation_id, 
            'layers': layers,  # Simple list with variable names
            'status_metadata': status_metadata
        }

        cache.set(cache_key, metadata, timeout=3600)

    return JsonResponse(metadata)





    
@require_http_methods(["GET"])
def get_tile(request, validation_id, metric_name, var_name, projection, z, x, y):
    """
    Serve a datashader tile with efficient DataFrame caching.
    """

    if projection not in (4326, 3857):
        return HttpResponse("Invalid projection. Use '4326' or '3857'", status=400)

    zarr_path = get_cached_zarr_path(validation_id)
    if zarr_path is None or not os.path.exists(zarr_path):
        return HttpResponse(status=404)

    try:
        # Get cached DataFrame with spatial index
        df, tree, coord_cols = get_cached_dataframe_with_index(
            validation_id, var_name, zarr_path, projection
        )

        if df is None or len(df) == 0:
            return get_transparent_tile()

        # Select TMS based on projection
        tms = TMS_3857 if projection == 3857 else TMS_4326

        # Get tile bounds
        bbox = tms.xy_bounds(x, y, z)

        # Get spread parameters
        plot_resolution = get_plot_resolution(validation_id) * 100 # my demo fidgeting was using degree * 100. so the buffer function needs a *100 thing somewhere
        spread_px, buffer_px = get_spread_and_buffer(validation_id, z, plot_resolution)

        # Calculate ranges and buffers
        x_range = bbox.right - bbox.left
        y_range = bbox.top - bbox.bottom
        x_buffer = x_range * (buffer_px / 256)
        y_buffer = y_range * (buffer_px / 256)

        # Filter data with buffered bounds (this is still the bottleneck)
        mask = ((df[coord_cols[0]] >= bbox.left - x_buffer) & 
                (df[coord_cols[0]] <= bbox.right + x_buffer) &
                (df[coord_cols[1]] >= bbox.bottom - y_buffer) & 
                (df[coord_cols[1]] <= bbox.top + y_buffer))
        df_tile = df[mask].copy()

        if len(df_tile) == 0:
            return get_transparent_tile()

        # Map to buffered pixel space
        df_tile['px'] = ((df_tile[coord_cols[0]] - (bbox.left - x_buffer)) / 
                        (x_range + 2*x_buffer) * (256 + 2*buffer_px))
        df_tile['py'] = ((df_tile[coord_cols[1]] - (bbox.bottom - y_buffer)) / 
                        (y_range + 2*y_buffer) * (256 + 2*buffer_px))

        # Create buffered canvas
        cvs = ds.Canvas(plot_width=256 + 2*buffer_px,
                        plot_height=256 + 2*buffer_px,
                        x_range=(0, 256 + 2*buffer_px),
                        y_range=(0, 256 + 2*buffer_px))

        # Aggregate points
        agg = cvs.points(df_tile, 'px', 'py', ds.mean('value'))

        # Get normalization parameters and colormap
        vmin, vmax = compute_norm_params(validation_id, zarr_path, var_name)
        mpl_cmap = get_colormap(metric_name)

        # Shade and spread
        img = tf.shade(agg, cmap=mpl_cmap, how='linear', span=(vmin, vmax))
        img = tf.spread(img, px=spread_px, shape='square')

        # Crop and process
        pil_img = img.to_pil().convert('RGBA')
        pil_img = pil_img.crop((buffer_px, buffer_px, 256 + buffer_px, 256 + buffer_px))
        img_array = np.array(pil_img)

        # Set transparency
        alpha = np.where(
            (img_array[:, :, 0] == 0) & 
            (img_array[:, :, 1] == 0) & 
            (img_array[:, :, 2] == 0),
            0, 255
        ).astype(np.uint8)

        img_array[:, :, 3] = alpha
        pil_img = Image.fromarray(img_array, 'RGBA')

        # Save to buffer
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        return HttpResponse(img_buffer.getvalue(), content_type='image/png')

    except Exception as e:
        print(f"Error generating tile: {str(e)}")
        return HttpResponse(status=500)

    except Exception as e:
        print(f"Error generating tile: {e}")
        import traceback
        traceback.print_exc()
        return get_transparent_tile()


from numba import jit

@jit(nopython=True)
def create_border_mask(alpha_channel):
    h, w = alpha_channel.shape
    border_mask = np.zeros((h, w), dtype=np.bool_)

    for i in range(1, h-1):
        for j in range(1, w-1):
            if alpha_channel[i, j] == 0:
                # Check 4-connected neighbors
                if (alpha_channel[i-1, j] > 0 or 
                    alpha_channel[i+1, j] > 0 or
                    alpha_channel[i, j-1] > 0 or 
                    alpha_channel[i, j+1] > 0):
                    border_mask[i, j] = True

    return border_mask