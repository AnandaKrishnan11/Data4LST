import ee 
import fiona

class LandsatProcessor:
    def __init__(self,start_date,end_date,bounds):

        self.start_date = start_date
        self.end_date = end_date
        self.aoi = bounds


    def applyScaleFactors(self,image):
            optical_bands = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
            thermal_band = image.select("ST_B.*").multiply(0.00341802).add(149.0)

            return image.addBands(optical_bands, None, True).addBands(thermal_band, None, True)
 

    def cloudMask(self,image):
        """
        Masking cloud and cloud shadow to get a clear image
        """

        qa = image.select('QA_PIXEL')

        #Define bits for cloud shadow, and cloud
        cloudShadowBitMask = (1 << 4)
        cloudsBitMask = (1 << 3)

        #All flags should be set to zero (indicating clear conditions)
        mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
        return image.updateMask(mask)
        

    def addTimeBand(self,image):
        """
        Add a time band to each image for interpolation.
        """
        timeImage = image.metadata('system:time_start').rename('timestamp')
        timeImageMasked = timeImage.updateMask(image.mask().select(0))
        return image.addBands(timeImageMasked)
    

    def calculatePixelAvailability(self, image):
        """
        Calculate the percentage of valid pixels (non-masked) in the image.
        """
        totalPixels = image.select('QA_PIXEL').mask().reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=self.aoi,
            scale=30,
            maxPixels=1e9
        ).values().get(0)

        validPixels = image.select('QA_PIXEL').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=self.aoi,
            scale=30,
            maxPixels=1e9
        ).values().get(0)

        pixelAvailability = ee.Number(validPixels).divide(totalPixels).multiply(100)
        return image.set('pixelAvailability', pixelAvailability)
    

    def countImages(self, collection):
        """
        Count the number of images in the filtered collection.
        """
        return collection.size().getInfo()
    

    def filter_available_images(self, collection, percentage):
        """
        Filter the image collection based on the percentage of valid (non-masked) pixels.

        Parameters:
        - collection (ee.ImageCollection): The image collection to filter.
        - pourcentage (float): The minimum percentage of valid pixels required.

        Returns:
        - ee.ImageCollection: The filtered image collection.
        """
        # Apply the calculatePixelAvailability function to each image in the collection
        L8_data_filtered = collection.map(self.calculatePixelAvailability)

        # Filter the images based on the pixelAvailability property (at least the specified percentage)
        L8_data_filtered = L8_data_filtered.filter(ee.Filter.gte('pixelAvailability', percentage))
        return L8_data_filtered


    def filter_by_common_dates(self, collection, common_dates_array):
        """
        Filter the Landsat 8 collection using the common dates.
        """
        filters = ee.Filter.Or(*[self.date_filter(date) for date in common_dates_array])
        return collection.filter(filters)
    

    def calculateLST(self, image):
        """
        Calculate the LST from the thermal band.
        """
        thermalLST = image.select('ST_B10').subtract(273.15).rename('LST_thermal')
        return image.addBands(thermalLST)
    

    def calculate_indices(self, image):
        """
        Calculate the spectral indices.
        """   
        ndvi = image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        ndwi = image.normalizedDifference(["SR_B3", "SR_B5"]).rename("NDWI")
        ndbi = image.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
        return image.addBands([ndvi, ndwi, ndbi])
    

    def get_image(self, collection, image_index):
        count = self.countImages(collection)
        return ee.Image(collection.toList(count).get(image_index))
    
    
    def get_LST(self, collection):
        collection = collection.map(self.calculateLST)
        return collection.select('LST_thermal')
    
    
    def get_LST_index(self, collection):
        collection = collection.map(self.calculateLST)
        return collection.select(['LST_thermal', 'NDVI', 'NDWI', 'NDBI','SR_B2', 'SR_B3', 'SR_B4', 'SR_B5'])


    def get_times(self, collection):
        dates = collection.aggregate_array("system:time_start")

        # Format the dates using map with a lambda function
        dates = dates.map(lambda ele: ee.Date(ele).format())

        # Print the formatted dates
        return dates.getInfo()
    

    def get_Landsat_collection(self):

        """
        Get the Landsat collection filtered by date and bounds.
        """
        return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(self.aoi) \
            .filterDate(self.start_date, self.end_date) \
            .map(self.applyScaleFactors) \
            .map(self.cloudMask) \
            .map(self.addTimeBand) \
            .map(self.calculate_indices)
            

        