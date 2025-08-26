# Custom tensor class in python without any dependency (not even numpy)
import random

class Tensor:

    def __init__(self, data): #, req_grad=False):
        if isinstance(data, list) and data and not isinstance(data[0], list):
            data = [data]
        self._check_rectangular(data)
        self.data = data
        self.shape = self._get_shape(data)
        if len(self.shape) != 2:
            raise ValueError("Supporting upto 2D Tensors (Matrices) for now")

    def _check_rectangular(self, data):
        """Recursively ensure all sublists have the same length."""
        if isinstance(data, list):
            # All elements must be same type (either list or scalar)
            if not all(isinstance(x, type(data[0])) for x in data):
                raise ValueError("Inconsistent nesting in tensor data")
            
            # All sublists must have same length
            if all(isinstance(x, list) for x in data):
                first_len = len(data[0])
                for sub in data:
                    if len(sub) != first_len:
                        raise ValueError("Ragged tensor: inconsistent sublist lengths")
                    self._check_rectangular(sub)

    def _get_shape(self, data):
        if isinstance(data, list):
            if len(data) == 0:
                return (0,)
            return (len(data), ) + self._get_shape(data[0])
        else:
            return ()
         
    def __repr__(self):
        # return (f"tensor: {self.data}, shape: {self.shape}")

        if len(self.shape) == 2:
            rows = ",\n ".join(str(row) for row in self.data)
            return f"tensor:\n[{rows}], shape: {self.shape}"
        else:
            # Fallback for non-2D tensors
            return f"tensor: {self.data}, shape: {self.shape}"
    
# OPERATIONS
    def _elementwise_op(self, other, op):

        other_data = other.data if isinstance(other, Tensor) else other
        result_shape = self._broadcast_shape(self.shape, other.shape if isinstance(other, Tensor) else ())

        self_broadcasted = self._broadcast_to(self.data, self.shape, result_shape)
        other_broadcasted = other_data if not isinstance(other, Tensor) else self._broadcast_to(other_data, other.shape, result_shape)

        result = self._apply_elementwise(self_broadcasted, other_broadcasted, op)
        
        return Tensor(result)

    
    def _apply_elementwise(self, a, b, op):
        if not isinstance(a, list) and not isinstance(b, list):
            return op(a,b)
        elif not isinstance(a, list):  # broadcast scalar a
            return [self._apply_elementwise(a, y, op) for y in b]
        elif not isinstance(b, list):  # broadcast scalar b
            return [self._apply_elementwise(x, b, op) for x in a]
        
        return [self._apply_elementwise(x,y,op) for x,y in zip(a,b)]
    
    def _broadcast_shape(self, shape1, shape2):
    # Right-align shapes and apply broadcasting rules
        result = []
        for i in range(max(len(shape1), len(shape2))):
            dim1 = shape1[-1 - i] if i < len(shape1) else 1
            dim2 = shape2[-1 - i] if i < len(shape2) else 1
            
            if dim1 == dim2 or dim1 == 1 or dim2 == 1:
                result.append(max(dim1, dim2))
            else:
                raise ValueError(f"Shapes {shape1} and {shape2} not broadcastable")
        return tuple(reversed(result))

    def _broadcast_to(self, data, from_shape, to_shape):
        # Recursively replicate data to match to_shape
        if len(to_shape) == 0:
            return data  # scalar
        if len(from_shape) < len(to_shape):
            from_shape = (1,) * (len(to_shape) - len(from_shape)) + from_shape
        if from_shape[0] == to_shape[0]:
            # Broadcast each sublist
            return [self._broadcast_to(d, from_shape[1:], to_shape[1:]) for d in data]
        elif from_shape[0] == 1:
            # Repeat the same sublist to match size
            return [self._broadcast_to(data[0], from_shape[1:], to_shape[1:]) for _ in range(to_shape[0])]
        else:
            # Should not reach here if shapes check succeeded
            raise ValueError("Incompatible shapes during broadcasting")
    
    def _apply_unary(self, a, op):
        if not isinstance(a, list):
            return op(a)
        return [self._apply_unary(x, op) for x in a]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __add__(self, other):
        return self._elementwise_op(other, lambda x,y: x+y)
    
    def __mul__(self, other):
        return self._elementwise_op(other, lambda x,y: x*y)
    
    def __sub__(self, other):
        return self._elementwise_op(other, lambda x,y: x-y)
    
    def __neg__(self):
        return Tensor(self._apply_unary(self.data, lambda x: -x))
    
    def __pow__(self, other):
        return Tensor(self._apply_unary(self.data, lambda x : x**other))
    
# DEFINING RANDOM TENSORS
    @classmethod
    def unit_tensor(cls, unit: float, shape):
        """Create a tensor filled with ones or zeros."""
        if unit not in (0, 1):
            raise ValueError("unit must be 0 or 1")
        unit = float(unit)  # ensure float type
        def build(s):
            if len(s) == 1:
                return [unit] * s[0]
            return [build(s[1:]) for _ in range(s[0])]
        return cls(build(shape))
    
    @classmethod
    def random_tensor(cls, shape):
        '''Creating a tensor with random values'''
        def build(s):
            if len(s) == 1:  # last dimension
                return [random.random() for _ in range(s[0])]
            return [build(s[1:]) for _ in range(s[0])]
        return cls(build(shape))
    

# MATRIX OPERATIONS
    def matmul(self, other):
        assert isinstance(self, Tensor) and isinstance(other, Tensor), "Not a tensor"
        assert len(self.shape) == 2 and len(other.shape) == 2, "Not a matrix" # For UPTO 2d tensors
        if self.shape[1] != other.shape[0]:
            raise ValueError("Cannot multiply, order not compatible")
        else:    
            result = [
            [sum(self.data[i][k] * other.data[k][j] for k in range(self.shape[1]))
             for j in range(other.shape[1])]
            for i in range(self.shape[0])
        ]
        
        return Tensor(result)
    
    def transpose(self):
        '''Creating Transpose of the tensor
        
        input: Tensor of dimension 2 (shape: row, column)
        output: Tensor of dimension 2 (shape: column, row)

        Workings:
        The i loop -> will populate the new tensor's inner list with len(row)
        The j loop -> will iterate to num of columns

        [[a, b, c], [d, e, f]] --> transpose --> [[a, d], [b, e], [c, f]]
        i will populate inner list with m times
        j will initiate creating n inner lists
        '''

        row, col = self.shape
        tranposed_tensor = [
            [self.data[i][j] for i in range(row)]  
            for j in range(col)]
        
        return Tensor(tranposed_tensor)

    def flatten(self):
        m,n = self.shape
        flat_tensor = [self.data[i][j] for i in range(m) for j in range(n)]
        return flat_tensor
    

    def reshape(self, new_shape: tuple):
        m, n = self.shape
        new_m, new_n = new_shape

        if m*n != new_m*new_n:
            raise ValueError(
            f"Incompatible Size for reshape. "
            f"New size {new_m, new_n} should have {m * n} elements"
        )
        flat = self.flatten()

        reshaped_tensor = [flat[i* new_n:(i+1) * new_n]
                           for i in range(new_m)]
            
        return Tensor(reshaped_tensor)