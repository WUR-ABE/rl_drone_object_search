#!/bin/bash

if [[ -z "${DATA_HOME}" ]]; then
    echo "DATA_HOME is not set. Cannot draw field borders."
    exit 1
fi

draw_field --scheme_file $DATA_HOME/20240213_clustered_1/orthomosaic_12m/12m_utm-scheme.kml --objects_file $DATA_HOME/20240213_clustered_1/plants_clustered_1_RDNAPTRANS2008.csv --raster_cells 48 48 
draw_field --scheme_file $DATA_HOME/20240423_clustered_2/orthomosaic_12m/12m_utm-scheme.kml --objects_file $DATA_HOME/20240423_clustered_2/plants_clustered_2_RDNAPTRANS2008.csv --raster_cells 48 48 
draw_field --scheme_file $DATA_HOME/20240718_clustered_3/orthomosaic_12m/12m_utm-scheme.kml --objects_file $DATA_HOME/20240718_clustered_3/plants_clustered_3_RDNAPTRANS2008.csv --raster_cells 48 48 
draw_field --scheme_file $DATA_HOME/20240801_clustered_4/orthomosaic_12m/12m_utm-scheme.kml --objects_file $DATA_HOME/20240801_clustered_4/plants_clustered_4_RDNAPTRANS2008.csv --raster_cells 48 48 
