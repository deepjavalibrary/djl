# PyTorch NDArray operators

In the following examples, we assume

- `manager` is an instance of `ai.djl.ndarray.NDManager`
    - it is recommended to look through https://docs.djl.ai/master/docs/development/memory_management.html
      in advance so that you can get better insight of `NDManager`
- `$tensor` is a placeholder for an instance of `torch.tensor`
- `$ndarray` is a placeholder for an instance of `ai.djl.ndarray.NDArray`
- you import the following packages

    ```
    import ai.djl.ndarray.*;
    import ai.djl.ndarray.types.*;
    ```

## Data types

| torch                          | djl                |
|--------------------------------|--------------------|
| `torch.bool`                   | `DataType.BOOLEAN` |
| `torch.uint8`                  | `DataType.UINT8`   |
| `torch.int8`                   | `DataType.INT8`    |
| `torch.int16`,`torch.short`    | `DataType.INT16`   |
| `torch.int32`,`torch.int`      | `DataType.INT32`   |
| `torch.int64`,`torch.long`     | `DataType.INT64`   |
| `torch.float16`,`torch.half`   | `DataType.FLOAT16` |
| `torch.float32`,`torch.float`  | `DataType.FLOAT32` |
| `torch.float64`,`torch.double` | `DataType.FLOAT64` |

## NDArray creation

| torch                            | djl                                                          | note                                                |
|----------------------------------|--------------------------------------------------------------|-----------------------------------------------------|
| `tensor.tensor(array)`           | `manager.create(array)`                                      | `array` may be `[v0,v1,...]` or `[[v0,v1,...],...]` |
| `$tensor.to('cpu')`              | `$ndarray.toDevice(Device.cpu(),false)`                      |                                                     |
| `torch.zeros(p,q,r)`             | `manager.zeros(new Shape(p,q,r))`                            |                                                     |
| `torch.ones(p,q,r)`              | `manager.ones(new Shape(p,q,r))`                             |                                                     |
| `torch.zeros_like($tensor)`      | `$ndarray.zerosLike()`, `manager.zeros($ndarray.getShape())` |                                                     |
| `torch.ones_like($tensor)`       | `$ndarray.onesLike()`, `manager.ones($ndarray.getShape())`   |                                                     |
| `torch.full((p,q),fill_value=s)` | `manager.full(new Shape(p,q),r)`                             |                                                     |
| `torch.rand(p,q,r)`              | `manager.randomUniform(from,to,new Shape(p,q,r))`            | `randomUniform` requires a range                    |
| `torch.randn(p,q,r)`             | `manager.randomNormal(new Shape(p,q,r))`                     |                                                     |
| `torch.arange(p,q,r)`            | `manager.arange(p,q,r)`                                      |                                                     |
| `torch.linspace(p,q,r)`          | `manager.linspace(p,q,r)`                                    |                                                     |
| `torch.eye(n)`                   | `manager.eye(n)`                                             |                                                     |

### create a tensor from array

#### `torch.tensor`/`manager.create`

```python
a = [
    [1, 2, 3],
    [4, 5, 6]
]
tensor.tensor(a)
```

```
float[][] array = { {1, 2, 3}, {4, 5, 6} };
NDArray nd = manager.create(array);
```

<details>
<summary>
show output
</summary>

```
ND: (2, 3) cpu() float32
[
  [1., 2., 3.],
  [4., 5., 6.],
]
```

</details>

---

#### `to('cpu')`/`toDevice(Device.cpu())`

```
NDArray nd = manager.create(new float[] {1, 2, 3, 4, 5});
nd = nd.toDevice(Device.cpu(), false);
// OR
nd = nd.toDevice(Device.gpu(), false);
```

### create zero and one tensor

#### ones, zeros, ones_like and zeros_like

```python
torch.ones(1, 2, 3)
torch.zeros(1, 2, 3)
a = [
    [1, 2, 3],
    [4, 5, 6]
]
torch.ones_like(a)
torch.zeros_like(a)
```

```
NDArray ones = manager.ones(new Shape(1, 2, 3));
NDarray zeros = manager.zeros(new Shape(1, 2, 3));

NDArray a = manager.create(new float[][] { {1, 2, 3}, {4, 5, 6} });
ones = a.onesLike();
zeros= a.zerosLike();
```

### create a tensor from array with gradient

`NDArray` doesn't hold gradient by default and you have to explicitly require grad.

```python
a = [
    [1, 2, 3],
    [4, 5, 6]
]
torch.tensor(a, requires_grad=True, dtype=float)
```

```
NDArray a = manager.create(new float[][] { {1, 2, 3}, {4, 5, 6} });
a.setRequiresGradient(true);
a
```

<details>
<summary>
show output
</summary>

```
ND: (2, 3) cpu() float32 hasGradient
[
  [1., 2., 3.],
  [4., 5., 6.],
]
```

</details>

### create a tensor filled with values

Python

```python
# fill by value
torch.full((2, 3), fill_value=42)
# fill by random value
torch.rand(1, 2, 3)  # [0,1) uniform distribution
torch.randn(1, 2, 3)  # (0,1) normal distribution
torch.randint(0, 5, (1, 2, 3))

# fill by sequential values
torch.arange(1, 5, 0.5)
torch.linspace(1, 4, 5)

# diag
torch.eye(3)
```

#### full

```
manager.full(new Shape(2, 3), 42);
```

<details>
<summary>
show output
</summary>

```
ND: (2, 3) cpu() int32
[
 [42, 42, 42],
 [42, 42, 42],
]
```

</details>

---

#### rand/randomUniform

Different from Python, you need to specify a range of uniform distribution.

```
var from = 0;
var to = 1;
manager.randomUniform(from, to, new Shape(1, 2, 3));
```

<details>
<summary>
show output
</summary>

```
ND: (1, 2, 3) cpu() float32
[
  [
    [0.1044, 0.1518, 0.8869],
    [0.8307, 0.4503, 0.0178],
  ],
]
```

</details>

---

#### random/randomNormal

```
manager.randomNormal(new Shape(1, 2, 3));
```

<details>
<summary>
show output
</summary>

```
ND: (1, 2, 3) cpu() float32
[
  [
    [-1.7142,  0.8033, -0.406 ],
    [-1.8686, -0.3713,  0.4713],
  ],
]
```

</details>

---

#### arange and linspace

```
manager.arange(1f, 4f, 0.5f);
```

<details>
<summary>
show output
</summary>

```
ND: (6) cpu() float32
[1. , 1.5, 2. , 2.5, 3. , 3.5]
```

</details>

```
manager.linspace(1f,4f,5);
```

<details>
<summary>
show output
</summary>

```
ND: (5) cpu() float32
[1.  , 1.75, 2.5 , 3.25, 4.  ]
```

</details>

---

#### eye

```
manager.eye(3);
```

<details>
<summary>
show output
</summary>

```
ND: (3, 3) cpu() float32
[
  [1., 0., 0.],
  [0., 1., 0.],
  [0., 0., 1.],
]

```

</details>

---

## size, shape and transform

### overview

| torch                           | djl                               | note                                   |
|---------------------------------|-----------------------------------|----------------------------------------|
| `tensor.size()`                 | `$ndarray.getShape()`             | 2x3x4 tensor/ndarray returns `(2,3,4)` |
| `tensor.ndim()`                 | `$ndarray.getShape().dimension()` | 2x3x4 tensor returns `3`               |
| ???                             | `$ndarray.size()`                 | 2x3x4 ndarray returns `24`             |
| `tensor.reshape(p,q)`           | `$ndarray.reshape(p,q)`           |
| `torch.flatten($tensor)`        | `$ndarray.flatten()`              |
| `torch.squeeze($tensor)`        | `$ndarray.squeeze()`              |                                        |
| `torch.unsqueeze(tensor,dim)`   | `$ndarray.expandDims(dim)`        |                                        |
| `tensor.T, torch.t($tensor)`    | `$ndarray.transpose()`            |                                        |
| `torch.transpose(tensor,d0,d1)` | `$ndarray.transpose(d0,d1)`       |                                        |

### Size and Shape

#### get size and shape

```
var a = manager.zeros(new Shape(2, 3, 4))

a.getShape().dimension(); // => 3
a.getShape(); // => (2, 3, 4)
a.size(); // => 24
```

#### reshape

```
var a = manager.zeros(new Shape(2, 3, 4));
```

<details>
<summary>
show output
</summary>

```
ND: (2, 3, 4) cpu() float32
[
  [
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
  ],
  [
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
  ],
]
```

</details>

```
a.reshape(new Shape(6, 4));
// is equal to a.reshape(new Shape(6, -1))
```

<details>
<summary>
show output
</summary>

```
ND: (6, 4) cpu() float32
[
  [0., 0., 0., 0.],
  [0., 0., 0., 0.],
  [0., 0., 0., 0.],
  [0., 0., 0., 0.],
  [0., 0., 0., 0.],
  [0., 0., 0., 0.],
]
```

</details>

#### flatten

```
a.flatten();
// is equal to a.reshape(-1)
```

<details>
<summary>
show output
</summary>

```
ND: (24) cpu() float32
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ... 4 more]

```

</details>

You can also specify flatten dimension.

```
a.flatten(1,2);
```

<details>
<summary>
show output
</summary>

```
ND: (2, 12) cpu() float32
[
  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
]

```

</details>

### Transform

#### squeeze

`squeeze` Removes all singleton dimensions from this NDArray Shape.

```
var a = manager.zeros(new Shape(2, 1, 2));
```

`[[0., 0.]]` has a redundant dimension. `squeeze` method drops this dimension.

```
ND: (2, 1, 2) cpu() float32
[
    [[0., 0.]],
    [[0., 0.]],
]
```

```
a.squeeze();
```

<details>
<summary>
show output
</summary>

```
ND: (2, 2) cpu() float32
[
  [0., 0.],
  [0., 0.],
]
```

</details>

You can also drop only specific singleton dimensions.

```
var a = manager.zeros(2, 1, 2, 1);
```

```
ND: (2, 1, 2, 1) cpu() float32
[
  [[[0.],[0.]]],
  [[[0.],[0.]]],
]
```

```
a.squeeze(1);
```

<details>
<summary>
show output
</summary>

```
ND: (2, 2, 1) cpu() float32
[
  [[0.],[0.],],
  [[0.],[0.],],
]
```

</details>

#### unsqueeze/expandDims

Python

```python
a = torch.zeros(2, 2)
torch.unsqueeze(a, 0)
```

Java

```
var a = manager.zeros(new Shape(2,2));
```

```
ND: (2, 2) cpu() float32
[
  [0., 0.],
  [0., 0.],
]
```

```
a.expandDims(0);
```

<details>
<summary>
show output
</summary>

```
ND: (1, 2, 2) cpu() float32
[[
    [0., 0.],
    [0., 0.],
]]
```

</details>

```
var a = manager.create(new int[] {10, 20, 30, 40});
```

```
ND: (4) cpu() int32
[10, 20, 30, 40]
```

```
a.expandDims(-1);
```

<details>
<summary>
show output
</summary>

```
ND: (4, 1) cpu() int32
[
  [10],
  [20],
  [30],
  [40],
]
```

</details>

### Transpose

Python

```python

m = [
    [1, 2],
    [3, 4],
]
a = torch.tensor(m)
a.T
```

Java

```
var a = manager.create(new float[][] { {1, 2}, {3, 4} });
a.transpose();
```

<details>
<summary>
show output
</summary>

```
ND: (2, 2) cpu() int32
[
  [ 1,  3],
  [ 2,  4],
]
```

</details>

## Index and Slice

### Overview

In general, you can replace PyTorch fancy index expression with String interpolation.

| torch                          | djl                                                            |                                                                                                                      |
|--------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `torch.flip(q,(n*))`           | `$ndarray.flip(n*)`                                            |                                                                                                                      |
| `torch.roll(q))`               | ???                                                            |                                                                                                                      |
| `$tensor[idx]`                 | `$ndarray.get(idx)`                                            |                                                                                                                      |
| `$tensor[n:]`                  | `$ndarray.get("n:")` ,`$ndarray.get("{}:",n)`                  |                                                                                                                      |
| `$tensor[[p,q,r]]`             | `$ndarray.get(manager.create(new int[] {p,q,r}))`              |                                                                                                                      |
| `$tensor[n,m]`                 | `$ndarray.get(n,m)`                                            |                                                                                                                      |
| `$tensor[$indices]`            | `$ndarray.get($indices)`                                       | `$indices` is int tensor/ndarray of shape 2 x 2                                                                      |
| `$tensor[:,n]`                 | `$ndarray.get(":,{}",n)`                                       |                                                                                                                      |
| `$tensor[:,n:m]`               | `$ndarray.get(":,{}:{}",n,m)`                                  |                                                                                                                      |
| `$tensor[:,[n,m]]`             | `$ndarray.get(":,{}",$colIndices)`                             | `$colIndices` is `NDArray {n, m}`.                                                                                   |
| `$tensor[[p,q,r],[s,t,u]]`     | `$ndarray.get("{},{}",$rowIndices,$colIndices)`                | `$tensor` and `$ndarray` are 2 dimension tensor/ndarray. `$rowIndices` and `$colIndices` are 1 dimension int ndarray |
| `$tensor[[[p],[q],[r]],[s,t]]` | `$ndarray.get("{},{}",$rowIndices.expandDims(-1),$colIndices)` | `rowIndices` is `NDArray {p, q, r}`                                                                                  |

### 1 dimension ndarray slice

```
var a = manager.arange(0, 100, 10);
```

```
a.get("5:");
```

<details>
<summary>
show output
</summary>

```
ND: (5) cpu() int32
[ 50,  60,  70,  80,  90]
```

</details>

```
a.get("{}:",5);
```

<details>
<summary>
show output
</summary>

```
ND: (5) cpu() int32
[ 50,  60,  70,  80,  90]
```

</details>

```
var indices = manager.create(new int[] {1, 3, 2});
a.get(indices);
```

<details>
<summary>
show output
</summary>

```
ND: (3) cpu() int32
[ 10,  30,  20]
```

</details>

```
var indices = manager.create(new int[][] { {2, 4}, {6, 8} });
a.get(indices);
```

<details>
<summary>
show output
</summary>

```
ND: (2, 2) cpu() int32
[
  [20, 40],
  [60, 80],
]
```

</details>

### multi-dimension ndarray slice

```
var a = manager.create(new int[][] { {5, 10, 20}, {30, 40, 50}, {60, 70, 80} });
```

---

#### `$tensor[n,m]`/`$ndarray.get(n,m)`

```
a.get(1, 2);
```

<details>
<summary>
show output
</summary>

```
ND: () cpu() int32
50
```

</details>


---

#### `tensor[:,n]` / `ndarray.get(":,{}",n)`

```
var n = 1;
a.get(":,{}", n);
```

<details>
<summary>
show output
</summary>

```
ND: (3) cpu() int32
[10, 40, 70]
```

</details>

---

#### `tensor[:,n:m]` / `ndarray.get(":,{}:{}",n,m)`

```
var n = 1;
var m = 2;
a.get(":,{}:{}", n, m);
```

<details>
<summary>
show output
</summary>

```
ND: (3, 1) cpu() int32
[
  [10],
  [40],
  [70],
]
```

</details>

---

#### `tensor[:,[n,m]]` / `ndarray.get(":,{}",$colIndices)`

```
var colIndices = manager.create(new int[] {2, 0});
a.get(":,{}", colIndices);
```

<details>
<summary>
show output
</summary>

```
ND: (3, 2) cpu() int32
[
  [20,  0],
  [50, 30],
  [80, 60],
]
```

</details>

---

#### `$tensor[[p,q,r],[s,t,u]]` / `$ndarray.get("{},{}",$rowIndices,$colIndices)`

```
var rowIndices = manager.create(new int[] {0, 1, 2});
var colIndices = manager.create(new int[] {2, 0, 1);
// select values at (0,2), (1,0) and (2,1) in 2d ndarray
a.get("{},{}", rowIndices,colIndices);
```

<details>
<summary>
show output
</summary>

```
ND: (3) cpu() int32
[20, 30, 70]
```

</details>

#### `$tensor[[[p],[q],[r]],[s,t]]`/ `$ndarray.get({},{},$rowIndices.expandDims(-1),$colIndices)`

```
var rowIndices = manager.create(new int[] {0, 1, 2}).expandDims(-1);
var colIndices = manager.create(new int[] {2, 0});
a.get("{},{}", rowIndices,colIndices);
```

<details>
<summary>
show output
</summary>

```
ND: (3, 2) cpu() int32
[
  [20,  0],
  [50, 30],
  [80, 60],
]
```

</details>


---

#### flip

```
var a = manager.create(new int[] {1,2,3,4,5,6,7,8}, new Shape(2,2,2));
```

```
ND: (2, 2, 2) cpu() int32
[
  [
    [ 1,  2],
    [ 3,  4],
  ],
  [
    [ 5,  6],
    [ 7,  8],
  ],
]
```

```
a.flip(0);
```

<details>
<summary>
show output
</summary>

```
[
  [
    [ 5,  6],
    [ 7,  8],
  ],
  [
    [ 1,  2],
    [ 3,  4],
  ],
]
```

</details>

```
a.flip(1);
```

<details>
<summary>
show output
</summary>

```
[
  [
    [ 3,  4],
    [ 1,  2],
  ],
  [
    [ 7,  8],
    [ 5,  6],
  ],
]
```

</details>

```
a.flip(2);
```

<details>
<summary>
show output
</summary>

```
ND: (2, 2, 2) cpu() int32
[
  [
    [ 2,  1],
    [ 4,  3],
  ],
  [
    [ 6,  5],
    [ 8,  7],
  ],
]
```

</details>

```
a.flip(0,1,2);
```

<details>
<summary>
show output
</summary>

```
ND: (2, 2, 2) cpu() int32
[
  [
    [ 8,  7],
    [ 6,  5],
  ],
  [
    [ 4,  3],
    [ 2,  1],
  ],
]
```

</details>

---

### concat and split

| torch                            | djl                           | note                                                                                                     |
|----------------------------------|-------------------------------|----------------------------------------------------------------------------------------------------------|
| `torch.cat(tensor0,tensor1,n)`   | `ndarray0.concat(ndarray1,n)` | vertically concat when `n` is `0` like `np.vstack`, horizontally concat when `n` is `1` like `np.hstack` |
| `torch.stack($tensor0,$tensor1)` | `$ndarray0.stack($ndarray1)`  |                                                                                                          |

#### concat

```
var zeros = manager.zeros(new Shape(2, 3));
var ones = manager.ones(new Shape(2, 3));
```

```
zeros.concat(ones, 0);
```

<details>
<summary>
show output
</summary>

```
ND: (4, 3) cpu() float32
[
  [0., 0., 0.],
  [0., 0., 0.],
  [1., 1., 1.],
  [1., 1., 1.],
]
```

</details>

```
zeros.concat(ones, 1);
```

<details>
<summary>
show output
</summary>

```
ND: (2, 6) cpu() float32
[
  [0., 0., 0., 1., 1., 1.],
  [0., 0., 0., 1., 1., 1.],
]
```

</details>

#### stack

```
var images0 = manager.create(new int[][] { {128, 0, 0}, {0, 128, 0} });
var images1 = manager.create(new int[][] { {0, 0, 128}, {127, 127, 127} });
```

```
images0.stack(images1);
```

<details>
<summary>
show output
</summary>

```
[
  [
    [128,   0,   0],
    [  0, 128,   0],
  ],
  [
    [  0,   0, 128],
    [127, 127, 127],
  ],
]
```

</details>

```
images0.stack(images1, 1);
```

<details>
<summary>
show output
</summary>

```
ND: (2, 2, 3) cpu() int32
[
  [
    [128,   0,   0],
    [  0,   0, 128],
  ],
  [
    [  0, 128,   0],
    [127, 127, 127],
  ],
]
```

</details>

```
images0.stack(images1, 2);
```

<details>
<summary>
show output
</summary>

```
[
  [
    [128,   0],
    [  0,   0],
    [  0, 128],
  ],
  [
    [  0, 127],
    [128, 127],
    [  0, 127],
  ],
]
```

</details>

### Arithmetic operations

| torch                             | djl                                       |
|-----------------------------------|-------------------------------------------|
| `$tensor + 1`                     | `$ndarray.add(1)`                         |
| `$tensor - 1`                     | `$ndarray.sub(1)`                         |
| `$tensor * 2`                     | `$ndarray.mul(2)`                         |
| `$tensor / 3 `                    | `$ndarray.div(3)`                         |
| `torch.mean($tensor)`             | `$ndarray.mean()`                         |
| `torch.median($tensor)`           | `$ndarray.median()`                       |
| `torch.sum($tensor)`              | `$ndarray.sum()`                          |
| `torch.prod($tensor)`             | `$ndarray.prod()`                         |
| `torch.cumsum($tensor)`           | `$ndarray.cumsum()`                       |
| `torch.topk($tensor,k,dim)`       | `$ndarray.topK(k,dim)`                    |
| `torch.kthvalue($tensor,k,dim)`   | ???                                       |
| `torch.mode($tensor)`             | ???                                       |
| `torch.std($tensor)`              | ???                                       |
| `torch.var($tensor)`              | ???                                       |
| `torch.std_mean($tensor)`         | ???                                       |
| `torch.abs($tensor)`              | `$ndarray.abs()`                          |
| `torch.ceil($tensor)`             | `$ndarray.ceil()`                         |
| `torch.round($tensor)`            | `$ndarray.round()`                        |
| `torch.trunc($tensor)`            | `$ndarray.trunc()`                        |
| `torch.flac($tensor)`             | ???                                       |
| `torch.clamp(tensor,min,max)`     | ???                                       |
| `torch.log($tensor)`              | `$ndarray.log()`                          |
| `torch.log2($tensor)`             | `$ndarray.log2()`                         |
| `torch.log10($tensor)`            | `$ndarray.log10()`                        |
| `torch.pow($tensor,n)`            | `$ndarray.power(n)`                       |
| `torch.pow(n,$tensor)`            | ???                                       |
| `torch.sigmoid($tensor)`          | `ai.djl.nn.Activation::sigmoid($ndarray)` |
| `torch.sign($tensor)`             | `$ndarray.sign()`                         |
| `torch.norm($tensor)`             | `$ndarray.norm()`                         |
| `torch.dist($tensor0,$tensor,p)`  | ???                                       |
| `torch.cdist($tensor0,$tensor,p)` | ???                                       |

#### mean

```python
torch.mean(tensor)
```

```
var a = manager.create(new float[] {0f,1f,2f,3f,4f,5f,6f,7f,8f,9f});
a.mean();
```

<details>
<summary>
show output
</summary>

```
ND: () cpu() float32
4.5
```

</details>

---

```
var a = manager.create(new float[] {0f,1f,2f,3f,4f,5f,6f,7f,8f,9f}).reshape(5,2);

a.mean(new int[] {0});
a.mean(new int[] {1});
```

<details>
<summary>
show output
</summary>

```
ND: (2) cpu() float32
[4., 5.]


ND: (5) cpu() float32
[0.5, 2.5, 4.5, 6.5, 8.5]
```

</details>

#### abs

```python
torch.abs(tensor)
```

```
var ndarray = manager.create(new int[][] { {1, 2}, {-1, -2} });
ndarray.abs();
```

<details>
<summary>
show output
</summary>

```
ND: (2, 2) cpu() int32
[
  [ 1,  2],
  [ 1,  2],
]
```

</details>

---

#### ceil

```python
torch.ceil(tensor)
```

```
var ndarray = manager.create(new double[][] { {1.0,1.1,1.2,1.3,1.4}, {1.5,1.6,1.7,1.8,1.9} });
ndarray.ceil();
```

<details>
<summary>
show output
</summary>

```
ND: (2, 5) cpu() float64
[
  [1., 2., 2., 2., 2.],
  [2., 2., 2., 2., 2.],
]
```

</details>

---

#### floor

```python
torch.floor(tensor)
```

```
var ndarray = manager.create(new double[][] { {1.0,1.1,1.2,1.3,1.4}, {1.5,1.6,1.7,1.8,1.9} });
ndarray.floor();
```

<details>
<summary>
show output
</summary>

```
ND: (2, 5) cpu() float64
[
  [1., 1., 1., 1., 1.],
  [1., 1., 1., 1., 1.],
]
```

</details>

---

#### round

```python
torch.round(tensor)
```

```
var ndarray = manager.create(new double[][] { {1.0,1.1,1.2,1.3,1.4}, {1.5,1.6,1.7,1.8,1.9} });
ndarray.round();
```

<details>
<summary>
show output
</summary>

```
ND: (2, 5) cpu() float64
[
  [1., 1., 1., 1., 1.],
  [2., 2., 2., 2., 2.],
]
```

</details>

---

#### sign

```python
torch.sign(tensor)
```

```
var ndarray = manager.create(new double[][] { {-0.1, -0.0}, {0.0, 0.1} });
ndarray.sign();
```

<details>
<summary>
show output
</summary>

```
ND: (2, 2) cpu() float64
[
  [-1.,  0.],
  [ 0.,  1.],
]
```

</details>

---

#### norm

```
var a = manager.create(new float[] {1f, 1f, 1f});
a.norm();
```

<details>
<summary>
show output
</summary>

```
ND: () cpu() float32
1.7321
```

</details>

### checking and filtering

| torch                                     | djl                                 |
|-------------------------------------------|-------------------------------------|
| `torch.isinf($tensor)`                    | `$ndarray.isInfinite()`             |
| `torch.isfinite($tensor)`                 | ???                                 |
| `torch.isnan($tensor)`                    | `$ndarray.isNaN()`                  |
| `torch.nonzero($tensor)`                  | `$ndarray.nonzero()`                |
| `torch.masked_select(tensor0,maskTensor)` | `$ndarray.booleanMask(maskndarray)` |
| `torch.where(a, cond)`                    | ???                                 |

---

#### isinf

```python
torch.isinf(tensor)
```

```
var ndarray = manager.create(
    new float[][] {
        {Float.NegativeInfinity, Float.MinValue, 0.0f},
        {Float.MaxValue, Float.PositiveInfinity, Float.NaN}
    }
);
ndarray.isInfinite();
```

<details>
<summary>
show output
</summary>

```
[
  [ true, false, false],
  [false,  true, false],
]
```

</details>

---

#### isNaN

```python
torch.isnan(tensor)
```

```
var ndarray = manager.create(
    new float[][] {
        {Float.NegativeInfinity, Float.MinValue, 0.0f},
        {Float.MaxValue, Float.PositiveInfinity, Float.NaN}
    }
);
ndarray.isNaN();
```

<details>
<summary>
show output
</summary>

```
ND: (2, 3) cpu() boolean
[
  [false, false, false],
  [false, false,  true],
]
```

</details>

---

#### nonzero

```python
a = torch.tensor(
    [
        [0.0, 0.1],
        [0.2, 0.3],
    ])
torch.nonzero()
```

```
var a = manager.create(new long[][] { {0.0,0.1}, {0.2,0.3} });
a.nonzero();
```

<details>
<summary>
show output
</summary>

```
ND: (3, 2) cpu() int64
[
  [ 0,  1],
  [ 1,  0],
  [ 1,  1],
]
```

</details>

---

#### masked_select/booleanMask

```python
t = torch.tensor(
    [
        [0.1, 0.2],
        [0.3, 0.4]
    ]
)
mask = torch.tensor(
    [
        [False, True],
        [True, False]
    ]
)

torch.masked_select(t, mask)
```

```
var ndarray = manager.create(new double[][] { {0.1, 0.2}, {0.3, 0.4} });
var mask = manager.create(new boolean[][] { {false, true}, {true,false} });
ndarray.booleanMask(mask);
```

<details>
<summary>
show output
</summary>

```
ND: (2) cpu() float64
[0.2, 0.3]
```

</details>


