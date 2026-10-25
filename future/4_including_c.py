# now lets bring c into the mix
# many python libs use c in the background to speed up everything
# lets do the same

# unfortunately for this we will need a import
# if anyone knows how to do this without import let me know!



import ctypes
testlib = ctypes.CDLL("./4_include_c.so")
testlib.dot_product.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
testlib.dot_product.restype = ctypes.c_float

def linear(input, weight, bias):
    assert input.shape[-1] == weights.shape[0]
    assert weight.shape[1] == bias.shape[0]
    

class Tensor:
    def __init__(self, shape):
        self.shape = shape
        self.size = 1
        for _ in shape:
            self.size *= _
        self.data = (ctypes.c_float * self.size)()

    def _flat_index(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != len(self.shape):
            if len(key) == 1 and len(self.shape) == 1:
                pass
            else:
                raise IndexError("Incorrect number of indices")
        
        flat = 0
        stride = 1

        for i in range(len(self.shape) - 1, -1, -1):
            idx = key[i]
            dim = self.shape[i]

            if idx < 0:
                idx += dim

            if not 0 <= idx < dim:
                raise IndexError("Tensor index out of range")

            flat += idx * stride
            stride *= dim

        return flat


    def __getitem__(self, key):
        return self.data[self._flat_index(key)]

    def __setitem__(self, key, value):
        self.data[self._flat_index(key)] = value

    @staticmethod
    def mat_mult_2d(tensor1, tensor2):
        assert len(tensor1.shape) == 2 and len(tensor2.shape) == 2
        assert tensor1.shape[-1] == tensor2.shape[0]
        

        