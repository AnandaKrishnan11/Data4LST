"""
Landsat8Processor -- WGAST-style API, Collection-2 correct, gap-free by default.

Method names follow the reference class so this drops into existing call sites.
What changed:

  cloudMask                  QA_PIXEL is Collection-2: bit 3 is CLOUD, bit 5 is
                             SNOW. The reference used Collection-1 pixel_qa
                             numbering and so never masked cloud shadow,
                             dilated cloud or cirrus at all.
  mask_mode                  'flag' by default -- NOTHING is masked out. Every
                             pixel keeps a value and QA_VALID records whether
                             it is trustworthy. Landsat ST_B10 and SR_B* are
                             produced under cloud, so a complete raster costs
                             you nothing but honesty about quality.
  coverage                   measured separately from cloud. A scene can be
                             100% clear over the part of the box it covers and
                             still leave half the box with no observation.
  calculatePixelAvailability measured over the snapped EXPORT RECTANGLE.
  calculate_indices          safe division, so no index-only holes.
  never clip()               clipping to a polygon writes nodata outside it.
  filter_by_common_dates     date_filter now exists on this class.
  snow                       not masked -- NDSI exists to find it.
  L9                         merged in, halving the revisit gap to ~8 days.

Band order, identical to modis.BAND_ORDER for the first 11 channels:

    blue, green, red, nir, swir1, swir2, LST, NDVI, NDWI, NDSI, QA_VALID
"""

import ee

from utils import bitwise_extract, fraction, safe_nd, fill_gaps, resolve_aoi

L8 = "LANDSAT/LC08/C02/T1_L2"
L9 = "LANDSAT/LC09/C02/T1_L2"

SRC_OPT = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
DST_OPT = ["blue", "green", "red", "nir", "swir1", "swir2"]
BAND_ORDER = DST_OPT + ["LST", "NDVI", "NDWI", "NDSI", "QA_VALID"]

# Collection-2 Level-2 QA_PIXEL. Do not reuse Collection-1 constants here.
QA_FILL, QA_DILATED, QA_CIRRUS, QA_CLOUD, QA_SHADOW, QA_SNOW = 0, 1, 2, 3, 4, 5


class Landsat8Processor:
    def __init__(
        self,
        start_date,
        end_date,
        aoi,
        mask_mode="flag",
        mask_snow=False,
        use_l9=True,
        lst_celsius=True,
        scene_cloud_prefilter=80,
        stat_scale=120,
    ):
        """
        aoi       : aoi.City, ee.Geometry, or [xmin, ymin, xmax, ymax].
        mask_mode : 'flag' -> keep every pixel, QA_VALID marks quality (default,
                              and the only mode that is gap-free by construction)
                    'fill' -> mask bad pixels, then interpolate them back
                    'mask' -> classic behaviour, holes where invalid
        """
        if mask_mode not in ("flag", "fill", "mask"):
            raise ValueError("mask_mode must be 'flag', 'fill' or 'mask'")
        self.region, self.city = resolve_aoi(aoi)
        self.aoi = self.region
        self.crs = self.city.crs if self.city is not None else None
        self.start_date = start_date
        self.end_date = end_date
        self.mask_mode = mask_mode
        self.mask_snow = mask_snow
        self.use_l9 = use_l9
        self.lst_celsius = lst_celsius
        self.scene_cloud_prefilter = scene_cloud_prefilter
        self.stat_scale = stat_scale

    # -------------------------------------------------------- per-image steps

    def applyScaleFactors(self, image):
        optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2).clamp(0, 1)
        thermal = image.select("ST_B.*").multiply(0.00341802).add(149.0)
        return image.addBands(optical, None, True).addBands(thermal, None, True)

    def qaMask(self, image):
        """1 where the pixel is good. Collection-2 bit numbering."""
        qa = image.select("QA_PIXEL")
        bad = (
            bitwise_extract(qa, QA_FILL).eq(1)
            .Or(bitwise_extract(qa, QA_DILATED).eq(1))
            .Or(bitwise_extract(qa, QA_CIRRUS).eq(1))
            .Or(bitwise_extract(qa, QA_CLOUD).eq(1))
            .Or(bitwise_extract(qa, QA_SHADOW).eq(1))
        )
        if self.mask_snow:
            bad = bad.Or(bitwise_extract(qa, QA_SNOW).eq(1))
        return bad.Not().rename("QA_VALID")

    def cloudMask(self, image):
        """
        Attach QA_VALID. Only applies updateMask in 'mask'/'fill' mode -- in
        'flag' mode the values stay, which is what keeps the raster complete.
        """
        valid = self.qaMask(image)
        out = image.addBands(valid)
        return out.updateMask(valid) if self.mask_mode != "flag" else out

    def calculateLST(self, image):
        lst = image.select("ST_B10")
        if self.lst_celsius:
            lst = lst.subtract(273.15)
        return image.addBands(lst.rename("LST_thermal"))

    def calculate_indices(self, image):
        """
        NDSI replaces WGAST's NDBI. Both pair SWIR1 with a visible/NIR band, so
        adding the NDBI line back costs nothing.
        Safe division: normalizedDifference() would mask any zero-denominator
        pixel and punch a hole into an otherwise complete image.
        """
        ndvi = safe_nd(image, "SR_B5", "SR_B4", "NDVI")
        ndwi = safe_nd(image, "SR_B3", "SR_B5", "NDWI")
        ndsi = safe_nd(image, "SR_B3", "SR_B6", "NDSI")
        return image.addBands([ndvi, ndwi, ndsi])

    def addTimeBand(self, image):
        t = image.metadata("system:time_start").rename("timestamp")
        return image.addBands(t.updateMask(image.mask().select(0)))

    def calculatePixelAvailability(self, image):
        avail = fraction(image.select("QA_VALID"), self.region, self.stat_scale)
        return image.set("pixelAvailability", avail)

    # ------------------------------------------------------------ collections

    def get_Landsat_collection(self):
        colls = [ee.ImageCollection(L8)]
        if self.use_l9:
            colls.append(ee.ImageCollection(L9))
        coll = colls[0]
        for c in colls[1:]:
            coll = coll.merge(c)
        return (
            coll.filterBounds(self.region)
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.lt("CLOUD_COVER", self.scene_cloud_prefilter))
            .map(self.applyScaleFactors)
            .map(self.cloudMask)
            .map(self.calculateLST)
            .map(self.calculate_indices)
        )

    def filter_disponible_images(self, collection, pourcentage):
        return collection.map(self.calculatePixelAvailability).filter(
            ee.Filter.gte("pixelAvailability", pourcentage)
        )

    def countImages(self, collection):
        return collection.size().getInfo()

    def get_image(self, collection, image_index):
        return ee.Image(collection.toList(collection.size()).get(image_index))

    def get_times(self, collection):
        return (
            collection.aggregate_array("system:time_start")
            .map(lambda e: ee.Date(e).format())
            .getInfo()
        )

    def date_filter(self, date_str):
        d = ee.Date(date_str)
        return ee.Filter.date(d, d.advance(1, "day"))

    def filter_by_common_dates(self, collection, common_dates_array):
        return collection.filter(
            ee.Filter.Or(*[self.date_filter(d) for d in common_dates_array])
        )

    # ---------------------------------------------- daily mosaics for pairing

    def to_stack(self, image):
        opt = image.select(SRC_OPT, DST_OPT)
        lst = image.select("LST_thermal").rename("LST")
        idx = image.select(["NDVI", "NDWI", "NDSI"])
        qa = image.select("QA_VALID")
        return opt.addBands(lst).addBands(idx).addBands(qa).select(BAND_ORDER)

    def _mosaic_for_date(self, date_str, fill=None):
        """
        One image per calendar day. Adjacent WRS-2 paths overlap, so a city can
        be covered by two scenes acquired the same day; mosaicking is also what
        closes the gap when neither scene covers the box on its own.
        """
        fill = (self.mask_mode == "fill") if fill is None else fill
        d = ee.Date(date_str)
        day = self.get_Landsat_collection().filterDate(d, d.advance(1, "day"))
        stack = self.to_stack(day.mosaic())

        # coverage: an observation exists at all, regardless of quality.
        coverage = fraction(stack.select("LST").mask(), self.region, self.stat_scale)
        avail = fraction(stack.select("QA_VALID"), self.region, self.stat_scale)

        if fill:
            qa = stack.select("QA_VALID").unmask(0)
            data = fill_gaps(
                stack.select(BAND_ORDER[:-1]), self.region, self.stat_scale,
                crs=self.crs,
            )
            stack = data.addBands(qa).select(BAND_ORDER)

        return stack.set({
            "date": date_str,
            "system:time_start": d.millis(),
            "pixelAvailability": avail,
            "coverage": coverage,
            "N_SCENES": day.size(),
            "SENSORS": day.aggregate_array("SPACECRAFT_ID"),
            "SOURCE_IDS": day.aggregate_array("LANDSAT_PRODUCT_ID"),
        })

    def daily_mosaics(self, fill=False):
        """fill=False by default: date screening never needs the gap filler."""
        dates = (
            self.get_Landsat_collection()
            .aggregate_array("system:time_start")
            .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
            .distinct()
            .sort()
        )
        return ee.ImageCollection(
            dates.map(lambda d: self._mosaic_for_date(ee.String(d), fill=fill))
        )

    def available_dates(self, pourcentage=90, min_coverage=100.0):
        """
        Dates passing BOTH tests. min_coverage=100 is the no-gaps guarantee:
        every pixel of the export rectangle has an observation that day.
        """
        return (
            self.daily_mosaics()
            .filter(ee.Filter.gte("pixelAvailability", pourcentage))
            .filter(ee.Filter.gte("coverage", min_coverage))
            .aggregate_array("date")
            .getInfo()
        )

    def date_stats(self):
        """
        [{date, avail, coverage}] for every candidate date, in ONE round trip
        (the previous version made three).
        """
        c = self.daily_mosaics()
        out = ee.Dictionary({
            "date": c.aggregate_array("date"),
            "avail": c.aggregate_array("pixelAvailability"),
            "cov": c.aggregate_array("coverage"),
        }).getInfo()
        return [
            {"date": d, "avail": a, "coverage": cv}
            for d, a, cv in zip(out["date"], out["avail"], out["cov"])
        ]

    def image_for_date(self, date_str, fill=None):
        return self._mosaic_for_date(date_str, fill=fill)
