import ee

class MODISProcessor:

    def __init__(self, start_date, end_date, bounds):
        """
        Initialize the MODISProcessor with study area and date range.
        
        Parameters:
        - start_date (str): The start date of the image collection (format: 'YYYY-MM-DD').
        - end_date (str): The end date of the image collection (format: 'YYYY-MM-DD').
        - bounds (ee.Geometry): The area of interest (AOI).
        """
    
        # Define the study area and date range
        self.aoi = bounds
        self.start_date = start_date
        self.end_date = end_date


    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def get_MODIS_collection(self):  # SW algo is used
        """
        Get the MODIS LST collection filtered by date and bounds.
        """
        return ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterBounds(self.aoi) \
            .filterDate(self.start_date, self.end_date) \
            .map(self.applyQDMask)


    def get_MODIS_reflectance_collection(self):
        """
        Get the MOD09GA surface reflectance collection, cloud-masked and
        scaled to 0-1. b01=Red, b02=NIR, b03=Blue, b04=Green, b06=SWIR1.
        """
        return ee.ImageCollection('MODIS/061/MOD09GA') \
            .filterBounds(self.aoi) \
            .filterDate(self.start_date, self.end_date) \
            .map(self.applyReflectanceMask) \
            .map(self.scale_reflectance)


    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def bitwiseExtract(self, value, fromBit, toBit):
        """
        Extract bit range from a given value.
        """
        maskSize = ee.Number(1).add(toBit).subtract(fromBit)
        mask = ee.Number(1).leftShift(maskSize).subtract(1)
        return value.rightShift(fromBit).bitwiseAnd(mask)


    def applyQDMask(self, image):
        """
        Apply a quality mask to MODIS LST daytime images based on the QC_Day band.
        """
        qcDay = image.select('QC_Day')
        qaMask = self.bitwiseExtract(qcDay, 0, 1).lte(1)
        dataQualityMask = self.bitwiseExtract(qcDay, 2, 3).eq(0)
        lstErrorMask = self.bitwiseExtract(qcDay, 6, 7).lte(2)
        mask = qaMask.And(dataQualityMask).And(lstErrorMask)
        return image.updateMask(mask)


    def applyReflectanceMask(self, image):
        """
        Cloud/shadow/cirrus mask for MOD09GA using the state_1km QA band.
        """
        state = image.select('state_1km')

        cloudState    = self.bitwiseExtract(state, 0, 1).eq(0)    # 0 = clear
        cloudShadow   = self.bitwiseExtract(state, 2, 2).eq(0)    # 0 = no shadow
        cirrus        = self.bitwiseExtract(state, 8, 9).lte(1)   # none or small
        internalCloud = self.bitwiseExtract(state, 10, 10).eq(0)
        adjacent      = self.bitwiseExtract(state, 13, 13).eq(0)

        mask = cloudState.And(cloudShadow).And(cirrus) \
                         .And(internalCloud).And(adjacent)
        return image.updateMask(mask)


    def applyJointMask(self, image):
        """
        Keep only pixels valid in BOTH the LST band and all reflectance bands.
        Applied to the whole image, so every band ends up identically masked
        (including QC_Day, which pixelAvailability is measured on).
        """
        lst_mask = image.select('LST_Day_1km').mask()
        refl_mask = image.select(['Blue', 'Green', 'Red', 'NIR']) \
                         .mask().reduce(ee.Reducer.min())  # 1 only where all valid

        return image.updateMask(lst_mask.And(refl_mask))


    # ------------------------------------------------------------------
    # Scaling and indices
    # ------------------------------------------------------------------

    def toCelsiusDay(self, image):
        """
        Convert LST from Kelvin to Celsius for daytime images.
        """
        lst = image.select('LST_Day_1km').multiply(0.02).subtract(273.15)
        return image.addBands(lst.rename('LST_Day_1km'), None, True)


    def scale_reflectance(self, image):
        """
        Apply the 0.0001 scale factor to MOD09GA reflectance bands.
        """
        bands = image.select(['sur_refl_b01', 'sur_refl_b02',
                              'sur_refl_b03', 'sur_refl_b04',
                              'sur_refl_b06']).multiply(0.0001)
        return image.addBands(bands, None, True)


    def calculate_indices_MODIS(self, image):
        """
        Calculate NDVI, NDWI, NDBI from MODIS reflectance bands.
        Same formulas as Landsat, using MODIS band equivalents.
        """
        ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']).rename('NDVI')
        ndwi = image.normalizedDifference(['sur_refl_b04', 'sur_refl_b02']).rename('NDWI')
        ndbi = image.normalizedDifference(['sur_refl_b06', 'sur_refl_b02']).rename('NDBI')
        return image.addBands([ndvi, ndwi, ndbi])


    def apply_scale_factors_time(self, image):
        """
        Apply scale factors to the Day_view_time band for time calculations.
        """
        optical_bands = image.select('Day_view_time').multiply(0.1)
        return image.addBands(optical_bands, None, True)


    # ------------------------------------------------------------------
    # Join LST + reflectance
    # ------------------------------------------------------------------

    def merge_bands(self, image):
        """
        Attach the matched reflectance image's indices and raw bands to the LST image.
        """
        image = ee.Image(image)
        match = ee.Image(image.get('reflectance_match'))
        renamed = match.select(
            ['NDVI', 'NDWI', 'NDBI', 'sur_refl_b03', 'sur_refl_b04',
            'sur_refl_b01', 'sur_refl_b02'],
            ['NDVI', 'NDWI', 'NDBI', 'Blue', 'Green', 'Red', 'NIR']
        )
        return image.addBands(renamed)


    def merge_LST_reflectance(self, lst_collection, reflectance_collection):
        reflectance_with_indices = reflectance_collection.map(self.calculate_indices_MODIS)

        time_filter = ee.Filter.maxDifference(
            difference=12 * 60 * 60 * 1000,
            leftField='system:time_start',
            rightField='system:time_start'
        )

        join = ee.Join.saveFirst(matchKey='reflectance_match')
        joined = join.apply(lst_collection, reflectance_with_indices, time_filter)
        joined = joined.filter(ee.Filter.notNull(['reflectance_match']))

        # Cast the join result back to an ImageCollection before mapping
        merged = ee.ImageCollection(joined).map(self.merge_bands)
        return merged.map(self.applyJointMask)

    # ------------------------------------------------------------------
    # Pixel availability and filtering
    # ------------------------------------------------------------------

    def calculatePixelAvailability_MODIS(self, image):
        """
        Calculate the percentage of valid (non-masked) pixels.
        After applyJointMask, QC_Day carries the joint mask, so this measures
        availability across LST *and* all spectral bands together.
        """
        stats = image.select('QC_Day').mask().rename('total') \
            .addBands(image.select('QC_Day').rename('valid')) \
            .reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=self.aoi,
                scale=1000,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4
            )

        totalPixels = ee.Number(stats.get('total'))
        validPixels = ee.Number(stats.get('valid'))

        pixelAvailability = validPixels.divide(totalPixels).multiply(100)
        return image.set('pixelAvailability', pixelAvailability)


    def filter_available_images(self, collection, percentage):
        """
        Filter the MODIS image collection based on the percentage of valid
        (non-masked) pixels.
        
        Parameters:
        - collection (ee.ImageCollection): The image collection to filter.
        - percentage (float): The minimum percentage of valid pixels required.

        Returns:
        - ee.ImageCollection: The filtered image collection.
        """
        modis_filtered = collection.map(self.calculatePixelAvailability_MODIS)
        modis_filtered = modis_filtered.filter(ee.Filter.gte('pixelAvailability', percentage))
        return modis_filtered


    # ------------------------------------------------------------------
    # Times
    # ------------------------------------------------------------------

    def format_time(self, image):
        """
        Compute and format the Day_view_time for each MODIS image.
        """
        day_view_time_band = image.select('Day_view_time')
        
        stats = day_view_time_band.reduceRegion(
            reducer=ee.Reducer.mean()
              .combine(reducer2=ee.Reducer.max(), sharedInputs=True)
              .combine(reducer2=ee.Reducer.min(), sharedInputs=True),
            geometry=self.aoi,
            scale=1000,  # Adjust the scale based on the resolution of MODIS
            maxPixels=1e4
        )
        
        time = stats.get('Day_view_time_mean')

        hours = ee.Number(time).floor().divide(1).int()
        minutes = ee.Number(time).subtract(hours).multiply(60).round().int()
        result = ee.String(hours.format('%02d')).cat(':').cat(minutes.format('%02d')).cat(':00')

        system_time_start = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd ')
        sts = system_time_start.cat(result)

        return ee.Feature(None, {'formattedTime': sts})


    def get_formatted_times(self, collection):
        """
        Retrieve and format the times for each image in the MODIS collection.
        Must be called on a collection that still has Day_view_time
        (i.e. before get_LST_index).
        """
        modis_lst_day = collection.map(self.apply_scale_factors_time)
        formatted_times = modis_lst_day.map(self.format_time)
        times_list = formatted_times.aggregate_array('formattedTime')
        return times_list.getInfo()


    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def countImages(self, collection):
        """
        Count the number of images in the filtered collection.
        """
        return collection.size().getInfo()


    def get_image(self, collection, image_index):
        """
        Get a single image from a collection by index.
        """
        count = self.countImages(collection)
        return ee.Image(collection.toList(count).get(image_index))


    def date_filter(self, date_str):
        """
        Convert the date strings to ee.Date objects.
        """
        return ee.Filter.date(ee.Date(date_str), ee.Date(date_str).advance(1, 'day'))


    def filter_by_common_dates(self, collection, common_dates_array):
        """
        Filter the MODIS_LST_data collection using the common dates.
        """
        filters = ee.Filter.Or(*[self.date_filter(date) for date in common_dates_array])
        return collection.filter(filters)


    def addTimeBand(self, image):
        """
        Add a time band to each image for interpolation.
        """
        timeImage = image.metadata('system:time_start').rename('timestamp')
        timeImageMasked = timeImage.updateMask(image.mask().select(0))
        return image.addBands(timeImageMasked)


    # ------------------------------------------------------------------
    # Band selection
    # ------------------------------------------------------------------

    def get_LST(self, collection):
        collection = collection.map(self.toCelsiusDay)
        return collection.select('LST_Day_1km')


    def get_LST_index(self, collection):
        """
        LST, then indices, then raw spectral bands — same order as Landsat.
        Call this AFTER merge_LST_reflectance and filtering.
        """
        collection = collection.map(self.toCelsiusDay)
        return collection.select(['LST_Day_1km', 'NDVI', 'NDWI', 'NDBI',
                                  'Blue', 'Green', 'Red', 'NIR'])