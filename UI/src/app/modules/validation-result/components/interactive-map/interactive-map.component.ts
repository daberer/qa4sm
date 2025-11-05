// interactive-map.component.ts
import { Component, input, OnInit, Input, inject, AfterViewInit, ChangeDetectorRef, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { DropdownModule } from 'primeng/dropdown';
import { ButtonModule } from 'primeng/button';
import { MapModule } from '../../../map/map.module';
import { Map, View } from 'ol';
import TileLayer from 'ol/layer/Tile';
import TileGrid from 'ol/tilegrid/TileGrid';
import { XYZ, Cluster, TileWMS } from 'ol/source';
import VectorSource from 'ol/source/Vector';
import { transformExtent } from 'ol/proj'
import { FullScreen } from 'ol/control';
import { Extent } from 'ol/extent';
import { WebsiteGraphicsService } from '../../../core/services/global/website-graphics.service';
import { TiffLayer } from './interfaces/layer.interfaces';
import { ValidationrunDto } from '../../../core/services/validation-run/validationrun.dto';
import { MetricsPlotsDto } from '../../../core/services/validation-run/metrics-plots.dto';
import { LegendControl } from './legend-control/legend-control';
import { ValidationrunService } from '../../../core/services/validation-run/validationrun.service';

@Component({
  selector: 'qa-interactive-map',
  standalone: true,
  imports: [MapModule, CommonModule, FormsModule, DropdownModule, ButtonModule],
  templateUrl: './interactive-map.component.html',
  styleUrls: ['./interactive-map.component.scss']
})

export class InteractiveMapComponent implements AfterViewInit, OnDestroy {
  Object = Object;
  private httpClient = inject(HttpClient);
  validationId = input('');
  @Input() validationRun: ValidationrunDto;
  // Inject ChangeDetectorRef
  private cdr = inject(ChangeDetectorRef);

  Map: Map;
  clusteredSource: VectorSource = new VectorSource();
  availableLayers: TiffLayer[] = [];
  currentLayer: TiffLayer | null = null;
  currentTileLayer: any = null;
  isLoading = false;
  private currentLayerKey: string = '';
  cachedMetadata: any = null;
  selectedLayerForMetric: { [metric: string]: LayerDetail } = {};
  colorbarData: any = null;
  private legendControl: LegendControl | null = null;
  statusMetadata: any = null;
  selectedMetric: MetricsPlotsDto = {} as MetricsPlotsDto;
  metrics: MetricsPlotsDto[] = [];
  public currentProjection: 'EPSG:4326' | 'EPSG:3857' = 'EPSG:4326';
  private baseLayer4326!: TileLayer<TileWMS>;
  private baseLayer3857!: TileLayer<XYZ>;
  private shouldFitToBounds = true;

  constructor(public plotService: WebsiteGraphicsService, private validationRunService: ValidationrunService) { }



  ngAfterViewInit() {
    // now the template exists, safe to build OpenLayers Map
    this.initMap();
    this.loadInitialMetric();
  }

  ngOnDestroy() {
    if (this.Map) {
      this.Map.dispose();
    }
  }


  private loadMetadataAndInitializeLayers() {
    const zarrMetrics = this.selectedMetric.zarr_metrics || {};

    this.plotService.getValidationMetadata(
      this.validationId(),
      zarrMetrics
    ).subscribe({
      next: (metadata) => {
        console.log('Received metadata:', metadata);
        this.cachedMetadata = metadata;
        this.statusMetadata = metadata.status_metadata || {};

        if (!metadata.layers || metadata.layers.length === 0) {
          console.error('No layers found in metadata');
          this.isLoading = false;
          this.cdr.detectChanges(); // Force detection
          return;
        }

        metadata.layers.forEach(layer => {
          const metric = layer.metric;
          if (!this.selectedLayerForMetric[metric]) {
            this.selectedLayerForMetric[metric] = {
              name: layer.name,
              metricName: metric,
              colormap: layer.colormap
            };
          }
        });

        if (this.selectedMetric) {
          const currentMetric = this.selectedMetric.metric_query_name;
          if (this.selectedLayerForMetric[currentMetric]) {
            this.isLoading = false;
            this.cdr.detectChanges(); // Force detection
            this.addTileLayerForMetric();
          } else {
            this.clearMapAndLegend();
          }
        } else {
          this.isLoading = false;
          this.cdr.detectChanges(); // Force detection
        }
      },
      error: (error) => {
        console.error('Error loading metadata:', error);
        this.isLoading = false;
        this.cdr.detectChanges(); // Force detection
      }
    });
  }


  // Method to handle layer selection change
  onLayerSelectionChange(metric: string, selectedLayer: LayerDetail | null) {
    if (!selectedLayer) {
      console.warn(`No layer found for selection`);
      return;
    }

    console.log(`Layer selection changed for metric ${metric}:`, selectedLayer);
    this.selectedLayerForMetric[metric] = selectedLayer;

    // If this is the currently displayed metric, update the map (but not the legend)
    if (this.selectedMetric?.metric_query_name === metric) {
      console.log('Updating map for current metric');
      this.addTileLayerForMetric();
    }
  }

  onMetricChange(event: any): void {
    console.log('[onMetricChange] Metric changed to:', event.value);
    this.shouldFitToBounds = false;
    this.selectedMetric = event.value;

    // DON'T set isLoading here - let updateTileLayer handle it
    // this.isLoading = true; // REMOVE THIS

    if (this.cachedMetadata) {
      if (this.selectedLayerForMetric[this.selectedMetric.metric_query_name]) {
        this.addTileLayerForMetric(); // This calls updateTileLayer which handles spinner
      } else {
        this.clearMapAndLegend();
      }
    } else {
      this.isLoading = true; // Only show for metadata loading
      this.loadMetadataAndInitializeLayers();
    }
  }

  private initMap() {
    console.log('[initMap] initializing map...');

    // 1. Define EOX WMS base layer (EPSG:4326)
    this.baseLayer4326 = new TileLayer({
      source: new TileWMS({
        url: 'https://tiles.maps.eox.at/wms',
        params: {
          LAYERS: 'terrain-light',
          TILED: true,
          CRS: 'EPSG:4326'
        },
        serverType: 'geoserver',
        attributions: 'Data © OpenStreetMap contributors and others, Rendering © EOX'
      }),
      visible: true, // start with this
    });

    // 2. Define OSM base layer (EPSG:3857)
    this.baseLayer3857 = new TileLayer({
      source: new XYZ({
        url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        attributions: '© OpenStreetMap contributors'
      }),
      visible: false // hidden initially
    });

    // 3. Initial view (use EPSG:4326 for now)
    const initialView = new View({
      projection: 'EPSG:4326',
      center: [0, 0],
      zoom: 2,
      minZoom: 0,
      maxZoom: 12
    });

    // 4. Construct map
    this.Map = new Map({
      target: 'imap',
      layers: [this.baseLayer4326, this.baseLayer3857],
      view: initialView,
      controls: [new FullScreen()]
    });
    this.legendControl = new LegendControl({
      colorbarData: null,
      metricName: ''
    });
    this.Map.addControl(this.legendControl);

    console.log('[initMap] map created with EOX base layer visible');
  }


  private addTileLayerForMetric() {
    const validationIdValue = this.validationId();
    console.log('addTileLayerForMetric - validationId:', validationIdValue);
    console.log('addTileLayerForMetric - selectedMetric:', this.selectedMetric);
    console.log('addTileLayerForMetric - cachedMetadata exists:', !!this.cachedMetadata);

    if (!validationIdValue || !this.selectedMetric) {
      console.error('No validationId or selectedMetric provided - cannot add tile layer');
      return;
    }

    if (!this.cachedMetadata) {
      console.warn('Metadata not loaded yet - cannot add tile layer');
      return;
    }


    const currentMetric = this.selectedMetric.metric_query_name;
    const selectedLayer = this.selectedLayerForMetric[currentMetric];

    console.log('Current metric:', currentMetric);
    console.log('Selected layer for metric:', selectedLayer);
    console.log('All selectedLayerForMetric:', this.selectedLayerForMetric);

    if (selectedLayer) {
      const metricLayer: TiffLayer = {
        name: selectedLayer.name || this.selectedMetric.metric_pretty_name || 'Metric Layer',
        metricName: selectedLayer.metricName,
        opacity: 0.7,
        isLoaded: false
      };

      console.log(`Adding tile layer for metric: ${this.selectedMetric.metric_pretty_name}, layer: ${selectedLayer.name}`);

      // Always add the tile layer
      this.updateTileLayer(metricLayer);

      // Always update colorbar for any metric with a tile layer
      this.updateVisualizationForCurrentMetric();

    } else {
      console.error(`No layer selected for metric: ${currentMetric}`);
      console.error('Available metrics in cachedMetadata:', Object.keys(this.cachedMetadata.metrics || {}));
    }


  }


  isCurrentMetricStatus(): boolean {
    return this.selectedMetric?.metric_query_name === 'status';
  }
  getStatusLegendEntries(): any[] {
    if (!this.colorbarData || !this.colorbarData.legend_data) {
      return [];
    }
    return this.colorbarData.legend_data.entries || [];
  }
  // helper to create resolutions and tileGrid depending on projection
  private createTileGridForProjection(projection: 'EPSG:4326' | 'EPSG:3857') {
    if (projection === 'EPSG:4326') {
      const resolutions: number[] = [];
      for (let z = 0; z <= 11; z++) {
        resolutions[z] = 180 / (256 * Math.pow(2, z));
      }
      return new TileGrid({
        extent: [-180, -90, 180, 90],
        origin: [-180, 90],
        resolutions,
        tileSize: 256
      });
    } else {
      // WebMercator / EPSG:3857 tile grid (global)
      const WEBMERCATOR_EXTENT = [-20037508.342789244, -20037508.342789244, 20037508.342789244, 20037508.342789244];
      const origin: [number, number] = [WEBMERCATOR_EXTENT[0], WEBMERCATOR_EXTENT[3]];
      const initialResolution = 156543.03392804097; // resolution at zoom 0 for 256 tiles
      const resolutions: number[] = [];
      for (let z = 0; z <= 11; z++) {
        resolutions[z] = initialResolution / Math.pow(2, z);
      }
      return new TileGrid({
        extent: WEBMERCATOR_EXTENT,
        origin,
        resolutions,
        tileSize: 256
      });
    }
  }


  private async updateTileLayer(
    layer: TiffLayer,
    projection: 'EPSG:4326' | 'EPSG:3857' = 'EPSG:4326',
    fitToBounds: boolean = false
  ) {
    const layerKey = `${layer.metricName}_${layer.name}_${projection}`;
    console.log(`[updateTileLayer] Starting update for ${layerKey}`);

    const isNewLayer = this.currentLayerKey !== layerKey;

    if (isNewLayer) {
      this.currentLayerKey = layerKey;
      this.isLoading = true;
      console.log(`[updateTileLayer] Set isLoading = true for new layer`);
    }

    if (this.currentTileLayer) {
      this.Map.removeLayer(this.currentTileLayer);
    }

    const validationIdValue = this.validationId();
    if (!validationIdValue) {
      console.error('[updateTileLayer] No validationId for tile layer');
      this.isLoading = false;
      this.cdr.detectChanges(); // Force detection
      return;
    }

    const epsgCode = projection.replace('EPSG:', '');
    const tileUrl = `/api/${validationIdValue}/tiles/${layer.metricName}/${layer.name}/${epsgCode}/{z}/{x}/{y}.png`;

    const tileGrid = this.createTileGridForProjection(projection);

    this.currentTileLayer = new TileLayer({
      source: new XYZ({
        url: tileUrl,
        tileSize: 256,
        projection: projection,
        tileGrid: tileGrid
      }),
      opacity: layer.opacity ?? 0.7
    });

    const source = this.currentTileLayer.getSource();

    let firstTileReceived = false;

    const hideSpinner = () => {
      if (!firstTileReceived && this.isLoading) {
        firstTileReceived = true;
        console.log('[updateTileLayer] ✓ First tile received, hiding spinner NOW');
        this.isLoading = false;

        // THIS IS THE KEY FIX - Force Angular to detect the change
        this.cdr.detectChanges();

        console.log(`[updateTileLayer] isLoading set to: ${this.isLoading}`);

        // Clean up listeners
        source.un('tileloadend', tileLoadHandler);
        source.un('tileloaderror', errorHandler);
        console.log('[updateTileLayer] Cleaned up tile listeners');
      }
    };

    const tileLoadHandler = () => {
      console.log('[updateTileLayer] ✓✓✓ TILE LOAD END EVENT FIRED ✓✓✓');
      hideSpinner();
    };

    const errorHandler = () => {
      console.log('[updateTileLayer] ✗✗✗ TILE LOAD ERROR EVENT FIRED ✗✗✗');
      hideSpinner();
    };

    source.on('tileloadend', tileLoadHandler);
    source.on('tileloaderror', errorHandler);

    this.Map.addLayer(this.currentTileLayer);
    this.currentLayer = layer;
    this.currentProjection = projection;

    console.log(`[updateTileLayer] Layer added to map: ${layerKey}`);

    if (fitToBounds || this.shouldFitToBounds) {
      setTimeout(async () => {
        await this.fitToLayerBounds(validationIdValue, projection);
      }, 200);
    }
  }




  public reloadCurrentLayer() {
    if (!this.currentTileLayer || !this.currentLayer) {
      console.warn('No current tile layer to reload');
      return;
    }

    // Don't show spinner for refresh
    const source = this.currentTileLayer.getSource();
    if (source && typeof (source as any).refresh === 'function') {
      (source as any).refresh();
      console.log('Requested tile source refresh');
    } else {
      console.warn('Tile source has no refresh method; re-create layer');
      if (this.currentLayer) {
        // Preserve the key to prevent spinner on refresh
        const preservedKey = this.currentLayerKey;
        this.updateTileLayer(this.currentLayer, this.currentProjection);
        this.currentLayerKey = preservedKey;
      }
    }
  }




  private async fitToLayerBounds(validationId: string, layerProjection: 'EPSG:4326' | 'EPSG:3857' = 'EPSG:4326') {
    console.log(`[fitToLayerBounds] Fetching bounds for ${validationId} (${layerProjection})`);
    try {
      const response = await fetch(`/api/${validationId}/bounds/`);
      if (!response.ok) {
        console.warn('[fitToLayerBounds] Could not fetch bounds');
        return;
      }

      const data = await response.json();
      if (!data.extent) {
        console.warn('[fitToLayerBounds] No extent in response');
        return;
      }

      let extent = data.extent;
      const backendCRS = data.crs || 'EPSG:4326';
      const viewProj = this.Map.getView().getProjection().getCode();

      console.log(`[fitToLayerBounds] Backend CRS=${backendCRS}, View CRS=${viewProj}`);

      if (backendCRS !== viewProj) {
        extent = transformExtent(extent, backendCRS, viewProj);
        console.log('[fitToLayerBounds] Transformed extent to view CRS');
      }

      this.Map.renderSync();
      this.Map.getView().fit(extent, {
        padding: [200, 200, 200, 200],
        duration: 1000,
        maxZoom: 9
      });
      console.log('[fitToLayerBounds] Fit view completed');
    } catch (error) {
      console.error('[fitToLayerBounds] Error:', error);
    }
  }





  async resetMapView() {
    const validationIdValue = this.validationId();
    if (this.Map && this.currentLayer && validationIdValue) {
      this.shouldFitToBounds = true; // allow fit on next update
      await this.fitToLayerBounds(validationIdValue, this.currentProjection);
    } else if (this.Map) {
      this.Map.getView().animate({
        center: [0, 0],
        zoom: 1,
        duration: 1000
      });
    }
  }


  public async toggleProjection() {
    if (!this.currentLayer) {
      console.warn('[toggleProjection] No current layer to toggle');
      return;
    }
    const validationIdValue = this.validationId();
    const newProj = this.currentProjection === 'EPSG:4326' ? 'EPSG:3857' : 'EPSG:4326';
    console.log(`[toggleProjection] Switching from ${this.currentProjection} to ${newProj}`);

    // Toggle base layers
    this.baseLayer4326.setVisible(newProj === 'EPSG:4326');
    this.baseLayer3857.setVisible(newProj === 'EPSG:3857');

    // Change the map's view projection
    const currentView = this.Map.getView();
    const newView = new View({
      projection: newProj,
      center: currentView.getCenter(),
      zoom: currentView.getZoom(),
      minZoom: 0,
      maxZoom: 12
    });

    this.Map.setView(newView);
    console.log(`[toggleProjection] Map view changed to ${newProj}`);

    // Reload the content tile layer in the new projection
    // This will trigger spinner because it's a "new" layer (different projection)
    await this.updateTileLayer(this.currentLayer, newProj);
    await this.fitToLayerBounds(validationIdValue, newProj);
  }



  private updateVisualizationForCurrentMetric() {
    if (!this.selectedMetric || !this.cachedMetadata) {
      console.warn('No selectedMetric or cachedMetadata for visualization update');
      this.clearMapAndLegend();
      return;
    }

    const currentMetric = this.selectedMetric.metric_query_name;
    const selectedLayer = this.selectedLayerForMetric[currentMetric];

    if (!selectedLayer) {
      console.log(`No layer selected for metric: ${currentMetric}`);
      this.clearMapAndLegend();
      return;
    }

    console.log(`Updating visualization for metric: ${currentMetric}, layer: ${selectedLayer.name}`);

    // Find the layer in cached metadata to get its colormap info
    let colormapInfo = null;
    if (this.cachedMetadata.layers) {
      const layer = this.cachedMetadata.layers.find((l: any) =>
        l.name === selectedLayer.name && l.metric === currentMetric
      );
      if (layer && layer.colormap) {
        colormapInfo = layer.colormap;
      }
    }

    // Lazy-load vmin/vmax
    this.plotService.getLayerRange(
      this.validationId(),
      currentMetric,
      selectedLayer.name
    ).subscribe({
      next: (rangeData) => {
        // Combine colormap info + range data
        const completeColormap = {
          ...(colormapInfo || {}),
          vmin: rangeData.vmin,
          vmax: rangeData.vmax,
          metric_name: currentMetric
        };

        // Safely check if it's categorical with fallback
        const isCategorical = completeColormap.is_categorical === true;

        // Decide: colorbar or legend?
        if (isCategorical) {
          // Show legend for categorical data (e.g., status)
          this.showLegend(completeColormap, selectedLayer);
        } else {
          // Show colorbar for continuous data
          this.showColorbar(completeColormap);
        }

        // Store for tile rendering and template access
        this.colorbarData = completeColormap;
      },
      error: (error) => {
        console.error('Error fetching layer range:', error);
        // Fallback to default visualization
        this.showColorbar({
          vmin: 0,
          vmax: 1,
          metric_name: currentMetric,
          is_categorical: false
        });
      }
    });
  }

  // Helper method to get available layers for current metric
  getAvailableLayersForCurrentMetric(): LayerDetail[] {
    if (!this.cachedMetadata || !this.selectedMetric) {
      console.log('No cached metadata or selected metric for layers');
      return [];
    }

    // Try layer_mapping first (if it exists)
    if (this.cachedMetadata.layer_mapping) {
      const layerNames = this.cachedMetadata.layer_mapping[this.selectedMetric.metric_query_name];
      if (Array.isArray(layerNames) && layerNames.length > 0) {
        return layerNames.map(layerName => ({
          name: layerName,
          metricName: this.selectedMetric.metric_query_name
        }));
      }
    }

    // Fallback: filter layers by metric
    if (this.cachedMetadata.layers && Array.isArray(this.cachedMetadata.layers)) {
      const metricLayers = this.cachedMetadata.layers.filter(layer =>
        layer.metric === this.selectedMetric.metric_query_name
      );

      if (metricLayers.length > 0) {
        return metricLayers.map(layer => ({
          name: layer.name,
          metricName: this.selectedMetric.metric_query_name,
          colormap: layer.colormap
        }));
      }
    }

    console.log(`No layers found for metric: ${this.selectedMetric.metric_query_name}`);
    return [];
  }

  currentMetricHasMultipleLayers(): boolean {
    if (!this.cachedMetadata || !this.selectedMetric) return false;

    const metricData = this.cachedMetadata.layer_mapping[this.selectedMetric.metric_query_name];
    return metricData ? Object.keys(metricData).length > 1 : false;
  }

  getLayerByName(layerName: string): LayerDetail | null {
    const availableLayers = this.getAvailableLayersForCurrentMetric();
    return availableLayers.find(layer => layer.name === layerName) || null;
  }


  getColorbarData(metricName: string, index: number): Observable<any> {
    const validationId = this.validationId();
    return this.httpClient.get(`/api/${validationId}/colorbar/${metricName}/${index}/`);
  }

  getColorbarGradient(): string {
    return this.colorbarData?.gradient || 'linear-gradient(to right, blue, cyan, yellow, red)';
  }

  getColorbarMin(): string {
    return this.colorbarData?.vmin?.toFixed(2) || '0.0';
  }

  getColorbarMax(): string {
    return this.colorbarData?.vmax?.toFixed(2) || '1.0';
  }


  private clearMapAndLegend() {
    if (this.currentTileLayer) {
      this.Map.removeLayer(this.currentTileLayer);
      this.currentTileLayer = null;
      this.currentLayer = null;
    }
    if (this.legendControl) {
      this.legendControl.updateLegend(null, '');
    }
    this.colorbarData = null;
    this.currentLayerKey = '';
    this.isLoading = false;
    this.cdr.detectChanges(); // Force detection
  }

  private showColorbar(colorbarData: any) {
    console.log('Showing colorbar for continuous data:', colorbarData);

    // Hide legend
    if (this.legendControl) {
      this.legendControl.updateLegend(null, '');
    }

    // Store colorbar data (your template already uses this)
    this.colorbarData = colorbarData;
  }

  private showLegend(colorbarData: any, selectedLayer: LayerDetail) {
    console.log('Showing legend for categorical data:', colorbarData);

    // Get status legend entries from cached metadata
    const statusLegend = this.cachedMetadata.status_metadata?.[selectedLayer.name];

    if (statusLegend && this.legendControl) {
      // Combine colorbar data with status legend entries
      const legendData = {
        ...colorbarData,
        legend_data: statusLegend
      };

      this.legendControl.updateLegend(legendData, selectedLayer.metricName);

      // Store for template access (if needed)
      this.colorbarData = legendData;
    } else {
      console.warn('No status legend data found for layer:', selectedLayer.name);
      // Fallback: clear legend
      if (this.legendControl) {
        this.legendControl.updateLegend(null, '');
      }
    }
  }


  private loadInitialMetric(): void {
    const params = new HttpParams().set('validationId', this.validationId());
    this.validationRunService.getMetricsAndPlotsNames(params)
      .subscribe({
        next: (metrics) => {
          this.metrics = metrics;  // Store the array

          if (metrics.length > 0) {
            this.selectedMetric = metrics[0];  // Set first as default
            this.loadMetadataAndInitializeLayers();
          }
        },
        error: (error) => console.error('Error loading metrics:', error)
      });
  }
}

export interface LayerDetail {
  name: string;           // Variable name (var_name) for backend requests
  metricName: string;     // Metric this layer belongs to
  colormap?: any;         // Colormap metadata from backend (optional)
}



export interface ValidationMetadata {
  validation_id: string;
  layer_mapping: { [metric: string]: string[] };
}

