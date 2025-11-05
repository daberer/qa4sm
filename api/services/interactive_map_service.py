
from django.core.cache import cache
from django.conf import settings
from django.shortcuts import get_object_or_404
from validator.models import ValidationRun, Dataset
import os
import math


def get_cached_zarr_path(validation_id):
    """Handles caching + model access for zarr paths"""
    cache_key = f'zarr_path_{validation_id}'
    zarr_path = cache.get(cache_key)

    if not zarr_path:
        validation_run = get_object_or_404(ValidationRun, id=validation_id)
        zarr_path = os.path.join(settings.MEDIA_ROOT, validation_run.zarr_path)
        cache.set(cache_key, zarr_path, timeout=3600)

    return zarr_path


def get_plot_resolution(validation_id):
    """
    Retrieves the plot_resolution from the spatial reference dataset.
    Returns float or None (for NaN values).
    Not cached since it's only used as input to cached calculations.
    """
    validation_run = get_object_or_404(ValidationRun, id=validation_id)
    dataset_id = validation_run.spatial_reference_configuration.dataset_id
    dataset = get_object_or_404(Dataset, id=dataset_id)

    plot_resolution = dataset.plot_resolution

    # Handle NaN - return None or a default value
    if plot_resolution is None or math.isnan(plot_resolution):
        return None

    return plot_resolution