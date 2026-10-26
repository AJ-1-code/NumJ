# NumJ

[![](https://jitpack.io/v/AJ-1-code/NumJ.svg)](https://jitpack.io/#AJ-1-code/NumJ)
[![License: Custom MIT-Based](https://shields.io)](LICENSE)

**NumJ** is a high-performance, feature-rich multidimensional array and linear algebra library built entirely in pure Java. Designed with artificial intelligence and deep learning workflows in mind, NumJ brings NumPy-inspired multidimensional array capabilities, stride-based views, and broadcasting directly to the Java Virtual Machine.

With support for efficient array transformations, in-place arithmetic, and optimized matrix operations, NumJ provides the mathematical building blocks for numerical computing and neural network development in Java.

---

## Key Architectural Highlights

*   **Zero-Copy Virtual Views:** Operations like `transpose()`, `reshape()`, `expandDims()`, and `broadcastTo()` can represent compatible transformations through array metadata, including strides and offsets, without copying the underlying numerical data. This enables efficient manipulation of array layouts while avoiding unnecessary data duplication.

*   **In-Place Mutation Control:** Includes an in-place API (`addi()`, `subi()`, `muli()`, `divi()`, `reluInPlace()`) that allows developers to modify existing arrays directly. These operations can reduce intermediate allocations and memory traffic, providing greater control over memory usage in numerical computing and neural network workflows.

*   **Cache-Conscious Matrix Operations:** Uses an optimized `i -> kk -> j` loop ordering for matrix multiplication (`matMul`). For conventional row-major matrices, this improves memory access locality by traversing rows of the right-hand operand and output matrix sequentially, helping make better use of CPU caches.

*   **NumPy Interoperability:** Supports NumPy's NPY binary array format through `saveNpy()` and `loadNpy()`, allowing numerical array data to be exchanged between Python and Java applications for tasks such as transferring neural network weights.

---

## Deep Neural Network Benchmark

NumJ is designed to support memory-efficient neural network inference through its array operations and in-place mutation capabilities.

The following benchmark evaluates a 990 Million Parameter Deep Network (`16,384 -> 32768 -> 12288 -> 4096 -> 10`) across 20 timed inference iterations using single-threaded sequential execution.

```text
Initializing network...
Allocating Layer 1 (536M params)...
Allocating Layer 2 (402M params)...
Allocating Layer 3 (50M params)...
Allocating Layer 4...
Network: 16384 -> 32768 -> 12288 -> 4096 -> 10
Parameters: 989945866
Raw parameter storage: 3776.34 MiB
Warmup iterations: 1
Timed iterations: 5
Heap used before initialization: 4.02 MiB
Heap used after initialization: 3803.33 MiB
Maximum JVM heap: 12288.00 MiB
Output shape: [1, 10]

===== NumJ Metrics Output =====
Total time: 2968.553 ms
Average inference: 593.711 ms
Inferences/second: 1.68
Checksum: 2.501251935959
Heap used after benchmark: 3811.50 MiB
GC collections during timing: 0
GC time during timing: 0 ms
Output: 0.50025040, 0.50025022, 0.50024956, 0.50024855, 0.50024712, 0.50024527, 0.50024301, 0.50024033, 0.50023723, 0.50023377
```

*No garbage collection activity was observed during the timed inference iterations. This indicates that the benchmark workload did not trigger garbage collection during the measurement window, although it does not imply that every NumJ operation is allocation-free.*

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

Add the NumJ dependency directly inside the `dependencies` block of your `build.gradle` file:

```groovy
dependencies {
    implementation 'com.github.AJ-1-code:NumJ:1.0.1'
}
```

---

## AI Usage Example

```java
package test;

import numj.NumJArray;

public class Main {
    public static void main(String[] args) {
        // Create a batch containing 16,384 input features
        NumJArray input = NumJArray.arange(0, 16384).reshape(1, 16384);

        // Initialize weights and biases for a dense layer
        NumJArray weights = NumJArray.normal(16384, 4096);
        NumJArray bias = NumJArray.zeros(1, 4096);

        // Compute a dense layer with ReLU activation
        NumJArray layerOutput = input.matMul(weights).add(bias).relu();

        System.out.println(
            "Layer Output Shape: " +
            java.util.Arrays.toString(layerOutput.shape())
        );
    }
}
```

---

## License & Attribution

This project is licensed under a **Custom MIT-Based License** - see the [LICENSE](LICENSE) file for details. 

### Mandatory Attribution Requirement
Any distributed software that incorporates, uses, or includes code from this library must include the following acknowledgment in its documentation, credits, About section, or another reasonably accessible location:

> *"This program includes code using NumJ library"*

---

## Comprehensive API Inventory

NumJ provides a broad collection of multidimensional array operations, mathematical functions, reductions, and neural network activation functions for numerical computing in Java.

<details>
<summary><b>Click to expand full method index</b></summary>

### Array Initializations
*   `zeros`, `ones`, `full`, `of`, `arange`, `linspace`, `identity`, `eye`, `diag`, `tril`, `triu`
*   Random Distributions: `normal`, `uniform`

### Metadata & Structural Properties
*   `shape()`, `strides()`, `ndim()`, `size()`, `isContiguous()`, `toContiguousData()`

### Transformations & Virtual Views
*   `reshape()`, `transpose()`, `swapAxes()`, `ravel()`, `flatten()`, `expandDims()`, `squeeze()`, `slice()`, `concat()`, `stack()`, `split()`, `repeat()`, `tile()`, `pad()`, `flip()`, `roll()`

### Element-Wise Arithmetic (Standard & In-Place)
*   `add` / `addi`, `sub` / `subi`, `mul` / `muli`, `div` / `divi`, `mod`, `abs`, `neg`, `sqrt`, `pow`, `exp`, `log`, `floor`, `ceil`, `round`, `sign`, `clip()`

### Matrix Primitives & Mathematical Activations
*   `dot()`, `matMul()`, `outer()`, `trace()`, `diagonal()`, `inner()`
*   AI Activations & Gradients: `relu()` / `reluD()` / `relui()` , `sigmoid()` / `sigmoidD()`, `tanh()` / `tanhD()`, `softmax()` / `softmaxAxis()`

### Array Reductions & Statistics
*   Global & Axis-Specific: `sum()`, `prod()`, `mean()`, `min()`, `max()`, `var()`, `std()`, `norm()`, `median()`, `percentile()`, `quantile()`, `cumSum()`, `cumProd()`
*   Indices tracking: `argMax()`, `argMin()`, `argMaxAxis()`, `argMinAxis()`, `argsort()`

### Conditional Search & Evaluation
*   `where()`, `putWhere()`, `conditionalPut()`, `take()`, `put()`, `unique()`, `uniqueCounts()`, `nonzero()`, `argwhere()`, `countNonzero()`
*   Logical Matching: `equal()`, `notEqual()`, `greater()`, `greaterEqual()`, `less()`, `lessEqual()`, `logicalAnd()`, `logicalOr()`, `logicalXor()`, `logicalNot()`, `all()`, `any()`, `isClose()`, `allClose()`, `isNaN()`, `isInf()`, `isFinite()`

</details>
