#### IPython

##### ShortCut
```python
Ctrl + A # go to head of line
Ctrl + E # go to end of line
Ctrl + L # clean screen
```
##### Magic
```python
np.*load*? # introspection
%run python_script.py # run python script
%debug # debug
%prun # deeply profile
%lprun -f funcName func_call(x,y) # line code profile
%timeit np.dot(np.random.randn(100,100), np.random.randn(100,100)) # time test


$ipython --pylab # no wait when rendering image

print _, _27 # last output and no.27 output
```

#### Numpy

##### Create ndarray
```python
data1 = [[6, 7.5], [8, 0]]
arr1 = np.array(data1)
print arr1.ndim
print arr1.shape
print arr1.dtype

##### ways of creating ndarray
array(list)
asarray(data)
arange(start, end) # create array from start to end
ones(dim)
ones_like(data) # create 1 array which has the same dim with data
zeros(dim)
zeros_like(data)
empty(dim)
empty_like(data)
eye(dim)
identity(dim)
```

##### ndarray operation
```python
arr * arr # element-wise multiplication
arr - arr # element-wise substract
1 / arr 
arr ** 0.5
```

##### slice
```python
arr2d[0][2] == arr2d[0,2] # same meaning
# arr2d.shape = (3,3)
arr2d[:2,1:] # shape(2,2)
arr[2], arr[2,:] # shape(3,)
arr[:,:2] # shape(3,2)
arr[1,:2] # shape(2,)
arr.T # transpose
```

