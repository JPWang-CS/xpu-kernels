"""RMSNorm cases covering serving row counts, model widths and padding tails."""

MODEL_WIDTHS = (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 16384)
ALIGNED_SHAPES = tuple((rows, width) for width in MODEL_WIDTHS for rows in (1, 128, 1024))
SMALL_SHAPES = tuple((rows, width) for width in (128, 256, 768) for rows in (1, 128))
TAIL_SHAPES = (
    (3, 1), (1, 127), (33, 129), (1, 255), (129, 257),
    (3, 1023), (33, 1025), (127, 4095), (129, 4097),
    (33, 8191), (129, 8193), (17, 16383), (17, 16385),
)
LARGE_SHAPES = ((4096, 4096), (4096, 8192))
RMSNORM_SHAPES = SMALL_SHAPES + ALIGNED_SHAPES + TAIL_SHAPES + LARGE_SHAPES


def shape_group(shape):
    shape = tuple(shape)
    if shape in TAIL_SHAPES:
        return "tail"
    if shape in LARGE_SHAPES:
        return "large_rows"
    return "aligned"
