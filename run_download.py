import geopandas as gpd
import numpy as np
import ee
import os
import geemap

from processors.landsat_processor import LandsatProcessor
from processors.modis_processor import MODISProcessor


START_DATE = '2017-04-09'
END_DATE = '2025-05-30'

SHP_PATH = r"./aoi/major_cities_global.shp"
FIELD = "city"

PIXEL_THRESHOLD = 92

OUTPUT_PATH = "./data"



def main():

    #ee.Authenticate(auth_mode='localhost', force=True) #Needed only once
    ee.Initialize(project='elegant-cipher-451117-m8',opt_url='https://earthengine-highvolume.googleapis.com')

    #load the SHP and get one city polygon
    shp = gpd.read_file("./aoi/major_cities_global.shp")
    for id, cities in enumerate(shp["city"],start=0):
        polygon,city_name = (shp.loc[id, 'geometry'],cities)

        try:

            aoi = ee.Geometry.Rectangle(list(polygon.bounds))

            print(f"Looking at {city_name} for overlapping images")

            # ------------------------------
            # Landsat 8 
            # ------------------------------
            Landsat_LST_preprocess = LandsatProcessor(
                                    start_date=START_DATE,
                                    end_date=END_DATE,
                                    bounds=aoi
                                )

            # Load image collection
            L8_data = Landsat_LST_preprocess.get_Landsat_collection()
            print('Landsat 8 images before filtering:', Landsat_LST_preprocess.countImages(L8_data))

            # Filter by pixel availability
            L8_data_filtered = Landsat_LST_preprocess.filter_available_images(L8_data, PIXEL_THRESHOLD)
            print('Landsat 8 images after filtering:', Landsat_LST_preprocess.countImages(L8_data_filtered))

            # Extract LST, NDVI, NDBI, NDWI
            L8_LST_index_data_filtered = Landsat_LST_preprocess.get_LST_index(L8_data_filtered)

            # Get timestamps
            L8_times_filtered = Landsat_LST_preprocess.get_times(L8_LST_index_data_filtered)


            # ------------------------------
            # MODIS
            # ------------------------------
            MODIS_LST_preprocess = MODISProcessor(
                start_date=START_DATE,
                end_date=END_DATE,
                bounds=aoi
            )

            # Load both collections
            MODIS_data = MODIS_LST_preprocess.get_MODIS_collection()
            modis_reflectance = MODIS_LST_preprocess.get_MODIS_reflectance_collection()
            print('MODIS images before filtering:', MODIS_LST_preprocess.countImages(MODIS_data))

            # Join LST + reflectance, apply joint mask
            MODIS_merged = MODIS_LST_preprocess.merge_LST_reflectance(MODIS_data, modis_reflectance)

            # Filter on JOINT availability
            MODIS_data_filtered = MODIS_LST_preprocess.filter_available_images(MODIS_merged, PIXEL_THRESHOLD)
            print('MODIS images after joint filtering:', MODIS_LST_preprocess.countImages(MODIS_data_filtered))

            # Times — call before get_LST_index, needs Day_view_time
            MODIS_times = MODIS_LST_preprocess.get_formatted_times(MODIS_data_filtered)

            # Final band selection
            MODIS_LST_data = MODIS_LST_preprocess.get_LST_index(MODIS_data_filtered)

            # Extract date-only strings (YYYY-MM-DD) from each satellite time list
            dates_modis = np.array([date.split(' ')[0] for date in MODIS_times])
            dates_landsat = np.array([date.split('T')[0] for date in L8_times_filtered])

            # Find the intersection across all three satellites
            common_dates = np.intersect1d(dates_landsat, dates_modis)

            print(f"Total common dates found: {len(common_dates)}")

            # Filter each dataset to keep only the common dates
            MODIS_LST_data_common = MODIS_LST_preprocess.filter_by_common_dates(MODIS_LST_data, common_dates)
            Landsat_LST_data_common = MODIS_LST_preprocess.filter_by_common_dates(L8_LST_index_data_filtered, common_dates)

            modis_img = ee.Image(MODIS_LST_data_common.first())
            landsat_img = ee.Image(Landsat_LST_data_common.first())

            modis_date = ee.Date(modis_img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            landsat_date = ee.Date(landsat_img.get('system:time_start')).format('YYYY-MM-dd').getInfo()

            # ----------------- Export MODIS -----------------
            modis_dir = os.path.join(OUTPUT_PATH,city_name)
            if not os.path.exists(modis_dir):
                os.makedirs(modis_dir)

            modis_filename = os.path.join(modis_dir,f'modis_1k_{modis_date}.tif')

            geemap.download_ee_image(
                modis_img,
                filename=modis_filename,
                scale=1000,  # MODIS resolution: 1km
                region=MODIS_LST_preprocess.aoi,
                num_threads=8
            )

            # ----------------- Export Landsat -----------------
            landsat_dir = os.path.join(OUTPUT_PATH,city_name)
            if not os.path.exists(landsat_dir):
                os.makedirs(landsat_dir)

            landsat_filename = os.path.join(landsat_dir,f'landsat_30m_{landsat_date}.tif')

            geemap.download_ee_image(
                landsat_img,
                filename=landsat_filename,
                scale=30,  # Landsat 8resolution: 30m
                region=Landsat_LST_preprocess.aoi,
                num_threads=8
            )
        except Exception as e:
            print(f"FAILED {city_name}: {type(e).__name__}: {e}")
            continue


if __name__ == "__main__":
    main()