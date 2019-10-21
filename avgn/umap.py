from cuml.manifold.umap import UMAP as cumlUMAP
from umap import UMAP
import warnings


def umap_reduce(data, **kwargs):
    try:
        reducer = cumlUMAP(**kwargs)
        embedding = reducer.fit_transform(data)
    except (RuntimeError, TypeError) as e:
        warnings.warn(e)
        reducer = UMAP(**kwargs)
        embedding = reducer.fit_transform(data)
    return embedding, reducer
