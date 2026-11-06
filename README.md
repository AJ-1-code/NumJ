# NumJ

[![JitPack](https://jitpack.io/v/AJ-1-code/NumJ.svg)](https://jitpack.io/#AJ-1-code/NumJ)
[![License: Custom MIT-Based](https://img.shields.io/badge/License-Custom%20MIT--Based-blue.svg)](LICENSE.md)

**NumJ** is a high-performance, feature-rich multidimensional array and linear algebra library built in Java. Designed with artificial intelligence and deep learning workflows in mind, NumJ brings NumPy-inspired multidimensional array capabilities, stride-based views, broadcasting, and optimized mathematical operations directly to the Java Virtual Machine.

NumJ combines cache-conscious algorithms, SIMD acceleration through the Java Vector API, and parallel execution to provide efficient numerical computing and neural network operations without relying on external native numerical libraries.

With support for efficient array transformations, in-place arithmetic, and optimized matrix multiplication, NumJ provides the mathematical building blocks for numerical computing and neural network development in Java.

---

## Key Architectural Highlights

* **SIMD Acceleration:** Uses the Java Vector API (`jdk.incubator.vector`) to perform operations on multiple values simultaneously using CPU vector instructions. This enables SIMD acceleration for supported numerical operations, including optimized matrix multiplication kernels.

* **Parallel Matrix Operations:** Uses parallel execution for sufficiently large matrix multiplication workloads. Specialized kernels can divide output columns across multiple workers, improving performance for matrix-vector multiplication and other workloads where conventional matrix multiplication strategies are less effective.

* **Specialized Matrix Multiplication:** Includes cache-conscious blocked matrix multiplication for conventional matrices, alongside specialized skinny matrix multiplication kernels for workloads with very few input rows. These kernels are designed to take advantage of the memory layout and structure of different multiplication workloads.

* **Zero-Copy Virtual Views:** Operations like `transpose()`, `reshape()`, `expandDims()`, and `broadcastTo()` can represent compatible transformations through array metadata, including strides and offsets, without copying the underlying numerical data. This enables efficient manipulation of array layouts while avoiding unnecessary data duplication.

* **In-Place Mutation Control:** Includes an in-place API (`addi()`, `subi()`, `muli()`, `divi()`, `reluInPlace()`) that allows developers to modify existing arrays directly. These operations can reduce intermediate allocations and memory traffic, providing greater control over memory usage in numerical computing and neural network workflows.

* **NumPy Interoperability:** Supports NumPy's NPY binary array format through `saveNpy()` and `loadNpy()`, allowing numerical array data to be exchanged between Python and Java applications for tasks such as transferring neural network weights.

---

## Deep Neural Network Benchmark

NumJ is designed to support large neural network inference workloads through optimized matrix operations, SIMD acceleration, and in-place mutation capabilities.

The following benchmark evaluates a network containing **989,945,866 parameters**:

`16,384 -> 32,768 -> 12,288 -> 4,096 -> 10`

The benchmark uses single-sample inference with float32 parameters, one warmup iteration, and five timed inference iterations.

### NumJ Benchmark Results

```text
Initializing network...
Allocating Layer 1
After Layer 1 | Used: 2.01 GiB
Allocating Layer 2
After Layer 2 | Used: 3.52 GiB
Allocating Layer 3
After Layer 3 | Used: 3.71 GiB
Allocating Layer 4...
After Layer 4 | Used: 3.71 GiB

Network: 16384 -> 32768 -> 12288 -> 4096 -> 10
Parameters: 989945866
Raw parameter storage: 3776.34 MiB

Warmup iterations: 1
Timed iterations: 5

Heap used after initialization: 3803.89 MiB
Maximum JVM heap: 12288.00 MiB

Output shape: [1, 10]

===== NumJ Metrics Output =====
Total time: 1230.692 ms
Average inference: 246.138 ms
Inferences/second: 4.06

Checksum: 2.501251935959

Heap used after benchmark: 4246.47 MiB
GC collections during timing: 0
GC time during timing: 0 ms

Output:
0.50025040, 0.50025022, 0.50024956,
0.50024855, 0.50024712, 0.50024527,
0.50024301, 0.50024033, 0.50023723,
0.50023377
```

### Performance Comparison

| Implementation | Average inference |
|---|---:|
| Original NumJ (double precision) | 593.711 ms |
| NumJ v3 (float, no SIMD) | 774.697 ms |
| NumJ v1.0.5 (float + SIMD) | 1,109.556 ms |
| **NumJ (optimized skinny kernels)** | **246.138 ms** |
| NumPy (float32) | 202.889 ms |

Compared with the previous SIMD implementation, the optimized NumJ implementation achieved approximately **4.51x faster inference**, reducing average inference time by approximately **77.8%**.

In this particular benchmark, NumJ runs approximately **1.21x slower than NumPy**, putting its inference performance within roughly 21% of the measured NumPy result.

These figures represent individual benchmark runs and are not a guarantee of performance across different hardware, JVM configurations, or workloads. NumPy's performance also depends on its underlying BLAS implementation and threading configuration.

**Memory note:** No garbage collection activity was observed during the five timed inference iterations. This indicates that the benchmark did not trigger garbage collection during the measurement window, although it does not imply that every NumJ operation is allocation-free.

---

## Installation

### 1. Configure Repository Settings

Add the **JitPack** repository block inside your root `settings.gradle` file:

```groovy
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}
```

### 2. Add Project Dependency

Add the NumJ dependency inside the `dependencies` block of your `build.gradle` file:

```groovy
dependencies {
    implementation 'com.github.AJ-1-code:NumJ:1.1.0'
}
```

### 3. Configure the Java Vector API

NumJ's SIMD implementation uses the Java Vector API, which is an incubating module in the JDK.

Add the following JVM argument when running applications that use NumJ:

```text
--add-modules=jdk.incubator.vector
```

For example, with Gradle:

```groovy
application {
    applicationDefaultJvmArgs = [
        '--add-modules=jdk.incubator.vector'
    ]
}
```

Use a JDK version compatible with the Vector API version targeted by your NumJ release.

---

## AI Usage Example

```java
package test;

import numj.NumJArray;

public class Main {

    public static void main(String[] args) {

        // Create a batch containing 16,384 input features
        NumJArray input = NumJArray.arange(0, 16384)
                                  .reshape(1, 16384);

        // Initialize weights and biases for a dense layer
        NumJArray weights = NumJArray.normal(16384, 4096);
        NumJArray bias = NumJArray.zeros(1, 4096);

        // Compute a dense layer with ReLU activation
        NumJArray layerOutput = input.matMul(weights)
                                     .add(bias)
                                     .relu();

        System.out.println(
            "Layer Output Shape: " +
            java.util.Arrays.toString(layerOutput.shape())
        );
    }
}
```

---

## License & Attribution

This project is licensed under a **Custom MIT-Based License**. See [LICENSE.md](LICENSE.md) for details.

### Mandatory Attribution Requirement

Any distributed software that incorporates, uses, or includes code from this library must include the following acknowledgment in its documentation, credits, About section, or another reasonably accessible location:

> "This program includes code using NumJ library"

---

## Comprehensive API Inventory

NumJ provides a broad collection of multidimensional array operations, mathematical functions, reductions, and neural network activation functions for numerical computing in Java.

<details>
<summary><b>Click to expand full method index</b></summary>

### Array Initializations

* `zeros`, `ones`, `full`, `of`, `arange`, `linspace`, `identity`, `eye`, `diag`, `tril`, `triu`
* Random Distributions: `normal`, `uniform`

### Metadata & Structural Properties

* `shape()`, `strides()`, `ndim()`, `size()`, `isContiguous()`, `toContiguousData()`

### Transformations & Virtual Views

* `reshape()`, `transpose()`, `swapAxes()`, `ravel()`, `flatten()`, `expandDims()`, `squeeze()`, `slice()`, `concat()`, `stack()`, `split()`, `repeat()`, `tile()`, `pad()`, `flip()`, `roll()`

### Element-Wise Arithmetic (Standard & In-Place)

* `add` / `addi`, `sub` / `subi`, `mul` / `muli`, `div` / `divi`, `mod`, `abs`, `neg`, `sqrt`, `pow`, `exp`, `log`, `floor`, `ceil`, `round`, `sign`, `clip()`

### Matrix Primitives & Mathematical Activations

* `dot()`, `matMul()`, `outer()`, `trace()`, `diagonal()`, `inner()`
* AI Activations & Gradients: `relu()` / `reluD()` / `relui()`, `sigmoid()` / `sigmoidD()`, `tanh()` / `tanhD()`, `softmax()` / `softmaxAxis()`

### Array Reductions & Statistics

* Global & Axis-Specific: `sum()`, `prod()`, `mean()`, `min()`, `max()`, `var()`, `std()`, `norm()`, `median()`, `percentile()`, `quantile()`, `cumSum()`, `cumProd()`
* Index Tracking: `argMax()`, `argMin()`, `argMaxAxis()`, `argMinAxis()`, `argsort()`

### Conditional Search & Evaluation

* `where()`, `putWhere()`, `conditionalPut()`, `take()`, `put()`, `unique()`, `uniqueCounts()`, `nonzero()`, `argwhere()`, `countNonzero()`
* Logical Matching: `equal()`, `notEqual()`, `greater()`, `greaterEqual()`, `less()`, `lessEqual()`, `logicalAnd()`, `logicalOr()`, `logicalXor()`, `logicalNot()`, `all()`, `any()`, `isClose()`, `allClose()`, `isNaN()`, `isInf()`, `isFinite()`

</details>
