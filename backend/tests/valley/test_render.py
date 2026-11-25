import io

import numpy as np
from PIL import Image

from fundoscope.valley import NO_CLEAR_IMAGES, PLANTED
from fundoscope.valley.render import (
    FARMLAND_COLOR,
    NO_DATA_COLOR,
    SLOT_COLORS,
    class_color,
    class_png,
    corners,
    farmland_png,
    image_name,
)

CODES = np.array([[0, 1, 1, 2], [0, 1, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]], np.uint8)


def rgba(png: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(png))
    assert image.mode == "P"  # a few palette colors keep the file small
    return np.asarray(image.convert("RGBA"))


def test_a_class_image_shows_only_that_class():
    pixels = rgba(class_png(CODES, 1, "#2a78d6"))

    assert pixels[0, 1].tolist() == [0x2A, 0x78, 0xD6, 235]
    assert pixels[:, :, 3].astype(bool).tolist() == (CODES == 1).tolist()


def test_planted_is_striped():
    pixels = rgba(class_png(CODES, 3, "#008300", striped=True))

    colors = {tuple(pixel) for pixel in pixels[2:].reshape(-1, 4)}
    assert len(colors) == 2  # the color and a lighter stripe
    assert (0x00, 0x83, 0x00, 235) in colors


def test_the_farmland_image_is_a_light_wash():
    pixels = rgba(farmland_png(CODES))

    assert pixels[0, 0, 3] == 0
    assert tuple(pixels[0, 1]) == FARMLAND_COLOR


def test_each_class_keeps_its_color():
    classes = (
        "alfalfa",
        "avocado",
        "citrus",
        "fallow/bare",
        "grape",
        "mango",
        PLANTED,
        NO_CLEAR_IMAGES,
    )

    assert class_color("alfalfa") == SLOT_COLORS[0]
    assert class_color("other annual crops") == SLOT_COLORS[1]
    assert class_color(PLANTED) == SLOT_COLORS[5]
    assert class_color(NO_CLEAR_IMAGES) == NO_DATA_COLOR
    assert class_color("citrus", classes) == SLOT_COLORS[6]  # unnamed classes: the free slots
    assert class_color("mango", classes) == SLOT_COLORS[7]
    assert class_color("olive", classes) == NO_DATA_COLOR  # past the palette


def test_image_names():
    assert image_name("other annual crops") == "other-annual-crops"
    assert image_name("fallow/bare") == "fallow-bare"
    assert image_name(PLANTED) == "planted-not-yet-identifiable"


def test_corners_in_longitude_and_latitude():
    north_west, north_east, south_east, south_west = corners(
        (779_740, 8_177_460, 805_720, 8_201_070), 32718
    )

    assert north_west[0] < north_east[0] and south_west[0] < south_east[0]
    assert south_west[1] < north_west[1] and south_east[1] < north_east[1]
    assert -72.39 < north_west[0] < -72.37 and -16.26 < north_west[1] < -16.25
