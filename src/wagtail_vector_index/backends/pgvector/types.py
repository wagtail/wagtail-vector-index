import enum


class DistanceMethod(enum.Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "max_inner_product"
