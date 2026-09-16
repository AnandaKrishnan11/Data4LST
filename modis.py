"""
MODISProcessor -- WGAST-style API, extended with reflectance, gap-aware.

The reference class only handles MOD11A1 (LST). Fine for WGAST, which takes its
optical bands from Sentinel-2, but NDVI/NDWI/NDSI on the MODIS side need surface
reflectance, and MODIS splits the two across products:

    MOD11A1  1 km, daily  -> LST_Day_1km, QC_Day, Day_view_time
    MOD09GA  500 m, daily -> sur_refl_b01..b07, state_1km

Both are mosaicked per date, since a city can straddle two sinusoidal tiles.

THE GAP CAVEAT, which is specific to this sensor:
MOD09GA reflectance is produced under cloud (contaminated but present), so
'flag' mode returns it complete. MOD11A1 LST is NOT -- when QC_Day says "not
produced due to cloud" there is no number there at all. So unlike Landsat, a
gap-free MODIS LST cannot be had just by declining to mask. Your options are
to require near-total clear sky (min availability ~99-100) or to accept
synthesised pixels via mask_mode='fill'. QA_VALID tells the two apart.

Use Terra when pairing with Landsat: its descending node is ~10:30 local solar
time against Landsat's ~10:15-10:30, so both sample nearly the same point on
the diurnal curve. Aqua at ~13:30 does not, and that offset will exceed the
resolution effect you are trying to model.

Band order, matching landsat.BAND_ORDER on the first 11 channels:

    blue, green, red, nir, swir1, swir2, LST, NDVI, NDWI, NDSI, QA_VALID, view_time
"""

import ee

from utils import bitwise_extract, fraction, safe_nd, fill_gaps, resolve_aoi

# Two different LST algorithms, not two versions of one.
#
#   'sw'  MOD11A1 / MYD11A1   generalised split-window on bands 31/32.
#                             Emissivity assigned from a LAND-COVER CLASS
#                             lookup. Well validated over vegetated surfaces;
#                             known cold bias of roughly 2-4 K over arid and
#                             bare ground, because the lookup does not capture
#                             the emissivity of exposed sand and rock.
#
#   'tes' MOD21A1D / MYD21A1D ASTER Temperature-Emissivity Separation on bands
#                             29/31/32. RETRIEVES emissivity per pixel, so it
#                             handles bare surfaces far better, and hands you
#                             Emis_29/31/32 as physical covariates. Noisier,
#                             and its cloud screening is less aggressive than
#                             MOD11's, so keep the QC threshold tight.
#
# Neither matches Landsat C2 ST, which uses ASTER GED emissivity adjusted by
# NDVI. That mismatch is a land-cover-correlated bias between your LR and HR
# targets, and a downscaling model cannot tell it apart from a scale effect.
PRODUCTS = {
    ("terra", "sw"): {"lst": "MODIS/061/MOD11A1", "sr": "MODIS/061/MOD09GA"},
    ("terra", "tes"): {"lst": "MODIS/061/MOD21A1D", "sr": "MODIS/061/MOD09GA"},
    ("aqua", "sw"): {"lst": "MODIS/061/MYD11A1", "sr": "MODIS/061/MYD09GA"},
    ("aqua", "tes"): {"lst": "MODIS/061/MYD21A1D", "sr": "MODIS/061/MYD09GA"},
}

# Band names and scalings differ between the two products.
ALGO_BANDS = {
    "sw": {
        "lst": "LST_Day_1km", "qc": "QC_Day", "time": "Day_view_time",
        "lst_scale": 0.02, "lst_offset": 0.0, "time_scale": 0.1,
        "emis": [],
    },
    "tes": {
        "lst": "LST", "qc": "QC", "time": "View_Time",
        "lst_scale": 0.02, "lst_offset": 0.0, "time_scale": 0.1,
        "emis": ["Emis_31", "Emis_32"],   # scale 0.002, offset 0.49
    },
}
EMIS_SCALE, EMIS_OFFSET = 0.002, 0.49

# MOD09GA numbering does not follow wavelength order: b01 is RED, b02 is NIR.
SRC_SR = [
    "sur_refl_b03",  # blue    459-479
    "sur_refl_b04",  # green   545-565
    "sur_refl_b01",  # red     620-670
    "sur_refl_b02",  # nir     841-876
    "sur_refl_b06",  # swir1  1628-1652
    "sur_refl_b07",  # swir2  2105-2155
]
DST_SR = ["blue", "green", "red", "nir", "swir1", "swir2"]
BAND_ORDER = DST_SR + ["LST", "NDVI", "NDWI", "NDSI", "QA_VALID", "view_time"]


class MODISProcessor:
    def __init__(
        self,
        start_date,
        end_date,
        aoi,
        platform="terra",
        algorithm="sw",
        include_emissivity=False,
        mask_mode="fill",
        lst_celsius=True,
        strict_qc=False,
        assume_clear_unset=False,
        stat_scale=1000,
    ):
        """
        mask_mode : 'fill' is the default here, not 'flag'. MOD11A1 has genuine
                    holes under cloud that no amount of not-masking recovers, so
                    filling is the only way to guarantee a complete raster.
                    Set 'flag' if you would rather see the holes and reject
                    those dates outright.
        strict_qc : QC_Day mandatory flag == 00 and error <= 1 K. The default
                    also accepts 'produced, other quality' and error <= 2 K,
                    keeping far more pixels over heterogeneous terrain; the
                    reference used dataQualityMask.eq(0) with error <= 3 K.
        """
        if algorithm not in ALGO_BANDS:
            raise ValueError("algorithm must be 'sw' (MOD11A1) or 'tes' (MOD21A1D)")
        if (platform, algorithm) not in PRODUCTS:
            raise ValueError("platform must be 'terra' or 'aqua'")
        if include_emissivity and algorithm != "tes":
            raise ValueError("emissivity bands exist only in the TES product")
        if mask_mode not in ("flag", "fill", "mask"):
            raise ValueError("mask_mode must be 'flag', 'fill' or 'mask'")
        self.region, self.city = resolve_aoi(aoi)
        self.aoi = self.region
        self.crs = self.city.crs if self.city is not None else None
        self.start_date = start_date
        self.end_date = end_date
        self.platform = platform
        self.algorithm = algorithm
        self.bands = ALGO_BANDS[algorithm]
        self.include_emissivity = include_emissivity
        self.band_order = list(BAND_ORDER)
        if include_emissivity:
            self.band_order += self.bands["emis"]
        self.mask_mode = mask_mode
        self.lst_celsius = lst_celsius
        self.strict_qc = strict_qc
        self.assume_clear_unset = assume_clear_unset
        self.stat_scale = stat_scale
        self.lst_coll = ee.ImageCollection(PRODUCTS[(platform, algorithm)]["lst"])
        self.sr_coll = ee.ImageCollection(PRODUCTS[(platform, algorithm)]["sr"])

    # -------------------------------------------------------- per-image steps

    def bitwiseExtract(self, value, fromBit, toBit):
        return bitwise_extract(value, fromBit, toBit)

    def lstQAMask(self, image):
        """
        Both products share the QC layout: bits 0-1 mandatory, 2-3 data
        quality, 6-7 LST accuracy. Only the band name differs.
        """
        qc = image.select(self.bands["qc"])
        mandatory = bitwise_extract(qc, 0, 1)
        quality = bitwise_extract(qc, 2, 3)
        err = bitwise_extract(qc, 6, 7)
        if self.strict_qc:
            return mandatory.eq(0).And(quality.eq(0)).And(err.eq(0))
        return mandatory.lte(1).And(err.lte(1))

    def applyQDMask(self, image):
        return image.updateMask(self.lstQAMask(image))

    def srQAMask(self, image):
        state = image.select("state_1km")
        cloud = bitwise_extract(state, 0, 1)
        shadow = bitwise_extract(state, 2)
        cirrus = bitwise_extract(state, 8, 9)
        internal = bitwise_extract(state, 10)
        clear = cloud.eq(0).Or(cloud.eq(3)) if self.assume_clear_unset else cloud.eq(0)
        return clear.And(shadow.eq(0)).And(cirrus.lte(1)).And(internal.eq(0))

    def applySRMask(self, image):
        return image.updateMask(self.srQAMask(image))

    def toCelsiusDay(self, image):
        b = self.bands
        lst = image.select(b["lst"]).multiply(b["lst_scale"]).add(b["lst_offset"])
        if self.lst_celsius:
            lst = lst.subtract(273.15)
        return image.addBands(lst.rename(b["lst"]), None, True)

    def apply_scale_factors_time(self, image):
        b = self.bands
        t = image.select(b["time"]).multiply(b["time_scale"])
        return image.addBands(t, None, True)

    def emissivity(self, image):
        """Emis_31/Emis_32 in physical units. TES only."""
        return (
            image.select(self.bands["emis"])
            .multiply(EMIS_SCALE)
            .add(EMIS_OFFSET)
        )

    def calculate_indices(self, image):
        ndvi = safe_nd(image, "nir", "red", "NDVI")
        ndwi = safe_nd(image, "green", "nir", "NDWI")
        ndsi = safe_nd(image, "green", "swir1", "NDSI")
        return image.addBands([ndvi, ndwi, ndsi])

    # ------------------------------------------------------------ collections

    def get_MODIS_collection(self):
        """LST-only collection, matching the reference method."""
        return (
            self.lst_coll.filterBounds(self.region)
            .filterDate(self.start_date, self.end_date)
            .map(self.applyQDMask)
            .map(self.toCelsiusDay)
            .map(self.apply_scale_factors_time)
        )

    def calculatePixelAvailability_MODIS(self, image):
        avail = fraction(
            image.select(self.bands["lst"]).mask(), self.region, self.stat_scale
        )
        return image.set("pixelAvailability", avail)

    def filter_disponible_images(self, collection, pourcentage):
        return collection.map(self.calculatePixelAvailability_MODIS).filter(
            ee.Filter.gte("pixelAvailability", pourcentage)
        )

    def countImages(self, collection):
        return collection.size().getInfo()

    def date_filter(self, date_str):
        d = ee.Date(date_str)
        return ee.Filter.date(d, d.advance(1, "day"))

    def filter_by_common_dates(self, collection, common_dates_array):
        return collection.filter(
            ee.Filter.Or(*[self.date_filter(d) for d in common_dates_array])
        )

    def get_LST(self, collection):
        return collection.select(self.bands["lst"])

    def get_times(self, collection):
        return (
            collection.aggregate_array("system:time_start")
            .map(lambda e: ee.Date(e).format())
            .getInfo()
        )

    # ----------------------------------------------- joined stack for pairing

    def image_for_date(self, date_str, fill=None):
        """
        fill=None follows mask_mode. Pass fill=False to get the statistics
        without paying for (or crashing on) the gap filler -- a date that is
        about to be rejected must not be filled first.
        """
        fill = (self.mask_mode == "fill") if fill is None else fill
        d = ee.Date(date_str)
        lst_day = self.lst_coll.filterDate(d, d.advance(1, "day")).filterBounds(self.region)
        sr_day = self.sr_coll.filterDate(d, d.advance(1, "day")).filterBounds(self.region)

        lst_raw = self.apply_scale_factors_time(self.toCelsiusDay(lst_day.mosaic()))
        lst_ok = self.lstQAMask(lst_raw)
        # The LST band is already absent where the retrieval failed, so the
        # product's own mask has to be part of QA_VALID, not just QC.
        lst = lst_raw.select(self.bands["lst"]).updateMask(lst_ok).rename("LST")
        vtime = lst_raw.select(self.bands["time"]).rename("view_time")
        emis = self.emissivity(lst_day.mosaic()) if self.include_emissivity else None

        sr_raw = sr_day.mosaic()
        sr_ok = self.srQAMask(sr_raw)
        opt = sr_raw.select(SRC_SR, DST_SR).multiply(0.0001).clamp(0, 1)
        if self.mask_mode != "flag":
            opt = opt.updateMask(sr_ok)

        stack = self.calculate_indices(opt).addBands(lst)
        valid = lst.mask().And(sr_ok).rename("QA_VALID")

        coverage = fraction(
            lst_raw.select(self.bands["lst"]).mask().And(sr_raw.select("state_1km").mask()),
            self.region, self.stat_scale,
        )
        avail = fraction(valid, self.region, self.stat_scale)

        stack = stack.addBands(vtime)
        if emis is not None:
            stack = stack.addBands(emis)
        if fill:
            stack = fill_gaps(stack, self.region, self.stat_scale, crs=self.crs)
        stack = stack.addBands(valid.unmask(0)).select(self.band_order)

        return stack.set({
            "date": date_str,
            "system:time_start": d.millis(),
            "pixelAvailability": avail,
            "coverage": coverage,
            "N_LST": lst_day.size(),
            "N_SR": sr_day.size(),
            "PLATFORM": self.platform,
            "ALGORITHM": self.algorithm,
        })

    def date_stats(self, dates):
        """
        Availability and coverage for a whole list of dates in ONE round trip.

        check_date() is one getInfo per date. At 148 cities x ~20 candidate
        dates that is 3000 blocking calls; this collapses each city to a single
        call by building the collection server-side. fill=False throughout --
        screening never needs the gap filler.
        """
        dates = list(dates)
        if not dates:
            return []
        coll = ee.ImageCollection(
            ee.List(dates).map(lambda d: self.image_for_date(ee.String(d), fill=False))
        )
        out = ee.Dictionary({
            "date": coll.aggregate_array("date"),
            "avail": coll.aggregate_array("pixelAvailability"),
            "cov": coll.aggregate_array("coverage"),
            "n_lst": coll.aggregate_array("N_LST"),
            "n_sr": coll.aggregate_array("N_SR"),
        }).getInfo()
        return [
            {"date": d, "avail": a, "coverage": c, "n_lst": nl, "n_sr": ns}
            for d, a, c, nl, ns in zip(
                out["date"], out["avail"], out["cov"], out["n_lst"], out["n_sr"]
            )
        ]

    def check_date(self, date_str, pourcentage=90, min_coverage=100.0):
        """One round trip: (passes, availability%, coverage%)."""
        info = (
            self.image_for_date(date_str, fill=False)
            .toDictionary(["pixelAvailability", "coverage", "N_LST", "N_SR"])
            .getInfo()
        )
        avail = info.get("pixelAvailability")
        cov = info.get("coverage")
        ok = (
            info.get("N_LST", 0) > 0
            and info.get("N_SR", 0) > 0
            and avail is not None
            and cov is not None
            and avail >= pourcentage
            and cov >= min_coverage
        )
        return ok, avail, cov
