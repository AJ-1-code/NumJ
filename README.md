# NumJ 

[![](https://jitpack.io/v/AJ-1-code/NumJ.svg)](https://jitpack.io/#AJ-1-code/NumJ)

**NumJ** is a high-performance, feature-complete, monolithic linear algebra and tensor engine built natively in pure Java. Designed specifically for artificial intelligence and deep learning workflows, NumJ brings Python's NumPy-style multi-dimensional array capabilities, stride manipulation, and broadcasting mechanics straight into the Java Virtual Machine ecosystem—with **zero overhead**.

---

## Key Architectural Highlights

*   **Zero-Allocation Virtual Views:** Core operations like `transpose()`, `reshape()`, `expandDims()`, and `broadcastTo()` mutate array metadata (strides and offsets) natively. They create lightweight virtual views over raw data without duplicating the underlying memory arrays.
*   **Surgical Mutation Control:** Includes a full in-place mutating API (`addi()`, `subi()`, `muli()`, `divi()`, `reluInPlace()`) allowing developers to tightly optimize complex layer transitions and eliminate intermediate memory thrashing.
*   **Cache-Conscious Matrix Operations:** Employs an optimized `i -> kk -> j` stride reordering pattern for matrix multiplication (`matMul`). This guarantees data is accessed in contiguous layout blocks, maximizing L1/L2 hardware CPU cache hits.
*   **NumPy Ecosystem Interoperability:** Features native binary parsing via `saveNpy()` and `loadNpy()`, allowing seamless weight file transfers directly between Python environments and your Java backend application.

---

## Deep Neural Network Benchmark

NumJ is engineered to eliminate the biggest flaw in Java math engines: Garbage Collector overhead. The following benchmark runs a massive **76.55 Million Parameter Deep Network** (`16,384 -> 4,096 -> 2,048 -> 512 -> 10`) across 20 iterations using single-threaded sequential execution:

```text
Initializing network...
Network: 16384 -> 4096 -> 2048 -> 512 -> 10
Parameters: 76557834
Raw parameter storage: 584.09 MiB
Warmup iterations: 5
Timed iterations: 20
Heap used before initialization: 3.14 MiB
Heap used after initialization: 592.31 MiB

===== NumJ Benchmark Metrics =====
Average Inference Time : 57.553 ms
Inferences Per Second  : 17.38
GC Collections Timing  : 0
GC Time Spent Timing   : 0 ms
```
*NumJ maintains a flawless **0ms Garbage Collection footprint** under heavy arithmetic load due to its strict zero-allocation memory footprint strategy.*

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
Add the core library line straight into the `dependencies` block of your `build.gradle` file:

```groovy
dependencies {
    implementation 'com.github.AJ-1-code:NumJ:1.0.0'
}
```

---

## AI Usage Exampel

```java
package test;

import numj.NumJArray;

public class Main {
    public static void main(String[] args) {
        // Fluent chain layer sequence mimicking modern tensor frameworks
        NumJArray input = NumJArray.arange(0, 16384).reshape(1, 16384);
        NumJArray weights = NumJArray.normal(16384, 4096);
        NumJArray bias = NumJArray.zeros(1, 4096);

        // Compute forward layer smoothly with zero-allocation mechanics
        NumJArray layerOutput = input.matMul(weights).add(bias).relu();
        
        System.out.println("Layer Output Shape: " + java.util.Arrays.toString(layerOutput.shape()));
    }
}
```

---

## Comprehensive API Inventory

NumJ includes hundreds of production-ready mathematical primitives mapping directly to Python's linear algebra footprint:

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
*   AI Activations & Gradients: `relu()` / `reluD()`, `sigmoid()` / `sigmoidD()`, `tanh()` / `tanhD()`, `softmax()` / `softmaxAxis()`

### Array Reductions & Statistics
*   Global & Axis-Specific: `sum()`, `prod()`, `mean()`, `min()`, `max()`, `var()`, `std()`, `norm()`, `median()`, `percentile()`, `quantile()`, `cumSum()`, `cumProd()`
*   Indices tracking: `argMax()`, `argMin()`, `argMaxAxis()`, `argMinAxis()`, `argsort()`

### Conditional Search & Evaluation
*   `where()`, `putWhere()`, `conditionalPut()`, `take()`, `put()`, `unique()`, `uniqueCounts()`, `nonzero()`, `argwhere()`, `countNonzero()`
*   Logical Matching: `equal()`, `notEqual()`, `greater()`, `greaterEqual()`, `less()`, `lessEqual()`, `logicalAnd()`, `logicalOr()`, `logicalXor()`, `logicalNot()`, `all()`, `any()`, `isClose()`, `allClose()`, `isNaN()`, `isInf()`, `isFinite()`

</details>

