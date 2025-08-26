from tensort_v2_1_cuda import TensorT

A = TensorT([[1,2,3],[4,5,6]])      # (2,3)
B = TensorT([[0.5,0.0],[1.0,1.5],[2.0,3.0]])  # (3,2)

C = A.tmatmul(B)    # (2,2) on GPU
C.print()

# simple scalar loss: sum all entries, backprop
# loss = TensorT(1.0) * C   # broadcasting not enabled on GPU path yet; pass grad explicitly per element if needed
# For now, call backward with a ones-like grad:
C.backward(TensorT([[1.0, 1.0],[1.0, 1.0]]))
# grads land on A.grad / B.grad as TensorT objects (device-backed)
