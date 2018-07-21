import drqa.retriever.utils as utils
import sys
import pickle
import numpy as np
import tqdm

x, meta = utils.load_sparse_csr(sys.argv[1])
def convert(x, prefix):
    for t in tqdm.trange(x.shape[0]):
        start, end = x.indptr[t:t+2]
        if start != end:
            data = x.data[start:end]
            indices = x.indices[start:end]
            order = data.argsort()[::-1]
            x.data[start:end] = data[order]
            x.indices[start:end] = indices[order]

    # Binary sparse CSR data format:
    # First 8 int32 values are:
    # Number of rows
    # Number of columns
    # Number of index pointers (should always be number of rows plus 1)
    # Number of indices
    # Number of data entries
    # Index pointer alignment
    # Indices alignment
    # Data alignment

    with open(sys.argv[1] + '.' + prefix, 'wb') as f:
        f.write(np.array(
            [x.shape[0], x.shape[1], len(x.indptr), len(x.indices), len(x.data),
             x.indptr.dtype.alignment, x.indices.dtype.alignment, x.data.dtype.alignment],
            dtype='int64'
            ).tobytes()
            )
        f.write(x.indptr.tobytes())
        f.write(x.indices.tobytes())
        f.write(x.data.tobytes())

with open(sys.argv[1] + '.meta', 'wb') as f:
    pickle.dump(meta, f)
convert(x, 'csr')
convert(x.T.tocsr(), 'csc')
