package numj;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

public class NumJTest {

    public static void main(String[] args) {

        NumJArray a = new NumJArray(
            new double[]{1, 2, 3, 4},
            new int[]{2, 2}
        );

        NumJArray b = new NumJArray(
            new double[]{5, 6, 7, 8},
            new int[]{2, 2}
        );

        NumJArray c = new NumJArray(
            new double[]{1, 2, 3, 4, 5, 6},
            new int[]{2, 3}
        );

        NumJArray d = new NumJArray(
            new double[]{0, 1, 2, 3, 4, 5},
            new int[]{6}
        );

        // Shape
        check(a.shape()[0] == 2 && a.shape()[1] == 2, "shape");

        // Size
        check(a.size() == 4, "size");

        // Get / Set
        check(a.get(0, 1) == 2, "get");
        a.set(10, 0, 1);
        check(a.get(0, 1) == 10, "set");
        a.set(2, 0, 1);

        // Reductions
        check(a.sum() == 10, "sum");
        check(a.mean() == 2.5, "mean");
        check(a.min() == 1, "min");
        check(a.max() == 4, "max");

        // Math
        NumJArray negative = new NumJArray(
            new double[]{-1, -2, -3, -4},
            new int[]{2, 2}
        );

        check(negative.abs().get(0, 0) == 1, "abs");
        check(new NumJArray(new double[]{1, 4, 9, 16}, new int[]{2, 2}).sqrt().get(1, 1) == 4, "sqrt");
        check(a.pow(2).get(1, 1) == 16, "pow");
        check(Math.abs(a.exp().get(0, 0) - Math.exp(1)) < 0.000001, "exp");
        check(Math.abs(a.log().get(0, 0) - Math.log(1)) < 0.000001, "log");

        // Array operations
        NumJArray broadcast = new NumJArray(
            new double[]{9, 18, 18, 36},
            new int[]{2, 2}
        );

        NumJArray b1 = new NumJArray(
            new double[]{2, 2},
            new int[]{1, 2}
        );

        NumJArray result = broadcast.add(b1);

        check(
            result.get(0, 0) == 11 &&
            result.get(0, 1) == 20 &&
            result.get(1, 0) == 20 &&
            result.get(1, 1) == 38,
            "broadcast add"
        );

        check(b.sub(a).get(0, 0) == 4 && b.sub(a).get(1, 1) == 4, "sub");
        check(a.mul(b).get(0, 0) == 5 && a.mul(b).get(1, 1) == 32, "mul");
        check(b.div(a).get(0, 0) == 5 && b.div(a).get(1, 1) == 2, "div");

        // Scalar operations
        check(a.add(5).get(0, 0) == 6, "scalar add");
        check(a.sub(1).get(1, 1) == 3, "scalar sub");
        check(a.mul(2).get(1, 1) == 8, "scalar mul");
        check(a.div(2).get(1, 1) == 2, "scalar div");
        check(a.mod(3).get(1, 1) == 1, "mod");

        // Matrix multiplication
        NumJArray matmul = a.matMul(b);

        check(
            matmul.get(0, 0) == 19 &&
            matmul.get(0, 1) == 22 &&
            matmul.get(1, 0) == 43 &&
            matmul.get(1, 1) == 50,
            "matrix multiplication"
        );

        // Transpose
        NumJArray transpose = a.transpose();

        check(
            transpose.get(0, 0) == 1 &&
            transpose.get(0, 1) == 3 &&
            transpose.get(1, 0) == 2 &&
            transpose.get(1, 1) == 4,
            "transpose"
        );

        // Reshape
        NumJArray reshape = c.reshape(3, 2);

        check(
            reshape.get(0, 0) == 1 &&
            reshape.get(0, 1) == 2 &&
            reshape.get(1, 0) == 3 &&
            reshape.get(1, 1) == 4 &&
            reshape.get(2, 0) == 5 &&
            reshape.get(2, 1) == 6,
            "reshape"
        );

        // NumJ creation
        NumJArray zeros = NumJArray.zeros(2, 3);
        check(zeros.size() == 6 && zeros.get(1, 2) == 0, "zeros");

        NumJArray ones = NumJArray.ones(2, 3);
        check(ones.size() == 6 && ones.get(0, 0) == 1 && ones.get(1, 2) == 1, "ones");

        NumJArray full = NumJArray.full(7, 2, 3);
        check(full.size() == 6 && full.get(0, 0) == 7 && full.get(1, 2) == 7, "full");

        // Arange
        NumJArray arange1 = NumJArray.arange(5);
        check(
            arange1.size() == 5 &&
            arange1.get(0) == 0 &&
            arange1.get(4) == 4,
            "arange(stop)"
        );

        NumJArray arange2 = NumJArray.arange(1, 6);
        check(
            arange2.size() == 5 &&
            arange2.get(0) == 1 &&
            arange2.get(4) == 5,
            "arange(start, stop)"
        );

        NumJArray arange3 = NumJArray.arange(1, 6, 2);
        check(
            arange3.size() == 3 &&
            arange3.get(0) == 1 &&
            arange3.get(1) == 3 &&
            arange3.get(2) == 5,
            "arange(start, stop, step)"
        );

        // Linspace
        NumJArray linspace = NumJArray.linspace(1, 4, 4);

        check(
            linspace.size() == 4 &&
            linspace.get(0) == 1 &&
            linspace.get(1) == 2 &&
            linspace.get(2) == 3 &&
            linspace.get(3) == 4,
            "linspace"
        );

        // ReLU
        NumJArray input = new NumJArray(
            new double[]{-3, -1, 0, 2, 5},
            new int[]{5}
        );

        NumJArray relu = input.relu();

        check(
            relu.get(0) == 0 &&
            relu.get(1) == 0 &&
            relu.get(2) == 0 &&
            relu.get(3) == 2 &&
            relu.get(4) == 5,
            "ReLU"
        );

        // Slice
        NumJArray slice = d.slice(new int[]{1, 5, 1});

        check(
            slice.get(0) == 1 &&
            slice.get(1) == 2 &&
            slice.get(2) == 3 &&
            slice.get(3) == 4,
            "slice"
        );

        NumJArray e = new NumJArray(
            new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12
            },
            new int[]{3, 4}
        );

        NumJArray slice2D = e.slice(
            new int[]{0, 3, 1},
            new int[]{1, 4, 1}
        );

        check(
            slice2D.get(0, 0) == 2 &&
            slice2D.get(0, 2) == 4 &&
            slice2D.get(1, 0) == 6 &&
            slice2D.get(1, 2) == 8 &&
            slice2D.get(2, 0) == 10 &&
            slice2D.get(2, 2) == 12,
            "2D slice"
        );

        // Strides
        int[] strides = c.strides();

        check(
            strides[0] == 3 &&
            strides[1] == 1,
            "strides"
        );

        // Axis reductions
        NumJArray sumAxis0 = c.sumAxis(0);

        check(
            sumAxis0.get(0) == 5 &&
            sumAxis0.get(1) == 7 &&
            sumAxis0.get(2) == 9,
            "sumAxis"
        );

        NumJArray meanAxis0 = c.meanAxis(0);

        check(
            meanAxis0.get(0) == 2.5 &&
            meanAxis0.get(1) == 3.5 &&
            meanAxis0.get(2) == 4.5,
            "meanAxis"
        );

        // Concatenate
        NumJArray concat = a.concat(b, 0);

        check(
            concat.shape()[0] == 4 &&
            concat.shape()[1] == 2 &&
            concat.get(0, 0) == 1 &&
            concat.get(1, 1) == 4 &&
            concat.get(2, 0) == 5 &&
            concat.get(3, 1) == 8,
            "concat"
        );

        // Stack
        NumJArray stack = a.stack(b, 0);

        check(
            stack.shape()[0] == 2 &&
            stack.shape()[1] == 2 &&
            stack.shape()[2] == 2 &&
            stack.get(0, 0, 0) == 1 &&
            stack.get(0, 1, 1) == 4 &&
            stack.get(1, 0, 0) == 5 &&
            stack.get(1, 1, 1) == 8,
            "stack"
        );

        // Split
        NumJArray splitInput = new NumJArray(
            new double[]{
                1, 2,
                3, 4,
                5, 6,
                7, 8
            },
            new int[]{4, 2}
        );

        NumJArray[] split = splitInput.split(0, 2);

        check(
            split.length == 2 &&
            split[0].shape()[0] == 2 &&
            split[0].get(0, 0) == 1 &&
            split[0].get(1, 1) == 4 &&
            split[1].get(0, 0) == 5 &&
            split[1].get(1, 1) == 8,
            "split"
        );

        // Flatten
        NumJArray flatten = c.flatten();

        check(
            flatten.shape()[0] == 6 &&
            flatten.get(0) == 1 &&
            flatten.get(5) == 6,
            "flatten"
        );

        // Copy
        NumJArray copy = a.copy();

        copy.set(100, 0, 0);

        check(
            a.get(0, 0) == 1 &&
            copy.get(0, 0) == 100,
            "copy"
        );

        // Dot
        NumJArray dotA = new NumJArray(
            new double[]{1, 2, 3},
            new int[]{3}
        );

        NumJArray dotB = new NumJArray(
            new double[]{4, 5, 6},
            new int[]{3}
        );

        check(
            dotA.dot(dotB).item() == 32,
            "dot"
        );

        // ArgMax / ArgMin
        NumJArray values = new NumJArray(
            new double[]{4, 1, 9, 3, 2},
            new int[]{5}
        );

        check(values.argMax()[0] == 2, "argMax");
        check(values.argMin()[0] == 1, "argMin");

        // Clip
        NumJArray clip = values.clip(2, 5);

        check(
            clip.get(0) == 4 &&
            clip.get(1) == 2 &&
            clip.get(2) == 5 &&
            clip.get(3) == 3 &&
            clip.get(4) == 2,
            "clip"
        );

        // Floor
        NumJArray decimals = new NumJArray(
            new double[]{1.2, 2.8, -1.4},
            new int[]{3}
        );

        NumJArray floor = decimals.floor();

        check(
            floor.get(0) == 1 &&
            floor.get(1) == 2 &&
            floor.get(2) == -2,
            "floor"
        );

        // Ceil
        NumJArray ceil = decimals.ceil();

        check(
            ceil.get(0) == 2 &&
            ceil.get(1) == 3 &&
            ceil.get(2) == -1,
            "ceil"
        );

        // Round
        NumJArray round = decimals.round();

        check(
            round.get(0) == 1 &&
            round.get(1) == 3 &&
            round.get(2) == -1,
            "round"
        );

        // Sign
        NumJArray sign = new NumJArray(
            new double[]{-5, 0, 8},
            new int[]{3}
        ).sign();

        check(
            sign.get(0) == -1 &&
            sign.get(1) == 0 &&
            sign.get(2) == 1,
            "sign"
        );

        // Reusable fixtures (c = [[1,2,3],[4,5,6]], ct = c transposed view)
        NumJArray ct = c.transpose();

        // ---- Creation: arange / linspace edge cases ----
        check(arrayEquals(NumJArray.arange(0, 1, 0.25), 0, 0.25, 0.5, 0.75), "arange fractional step");
        check(arrayEquals(NumJArray.arange(5, 0, -2), 5, 3, 1), "arange negative step");
        check(NumJArray.arange(3, 3).size() == 0, "arange empty");
        check(NumJArray.arange(-2).size() == 0, "arange(stop) negative is empty");
        check(arrayEquals(NumJArray.linspace(0, 1, 5), 0, 0.25, 0.5, 0.75, 1), "linspace 0..1");
        check(arrayEquals(NumJArray.linspace(5, 9, 1), 5), "linspace num=1");
        check(NumJArray.linspace(0, 1, 0).size() == 0, "linspace num=0");
        check(NumJArray.linspace(0, 1, 7).get(6) == 1, "linspace ends exactly on stop");
        checkThrows(() -> NumJArray.arange(0, 5, 0), "arange step 0");
        checkThrows(() -> NumJArray.linspace(0, 1, -1), "linspace negative num");

        // ---- Creation: other factories ----
        check(arrayEquals(NumJArray.of(1, 2, 3), 1, 2, 3), "of(1D)");
        NumJArray of2d = NumJArray.of(new double[][]{{1, 2}, {3, 4}});
        check(of2d.get(1, 0) == 3 && Arrays.equals(of2d.shape(), new int[]{2, 2}), "of(2D)");
        checkThrows(() -> NumJArray.of(new double[][]{{1, 2}, {3}}), "of(2D) ragged");

        NumJArray eye = NumJArray.identity(3);
        check(eye.get(0, 0) == 1 && eye.get(1, 1) == 1 && eye.get(0, 1) == 0 && eye.trace() == 3, "identity");

        NumJArray rnd = NumJArray.normal(0, 1, new int[]{3, 4});
        check(Arrays.equals(rnd.shape(), new int[]{3, 4}), "normal shape");
        NumJArray uni = NumJArray.uniform(2, 3, 100);
        check(uni.min() >= 2 && uni.max() < 3, "uniform range");

        // ---- Constructor / get / set validation ----
        checkThrows(() -> new NumJArray(new double[5], new int[]{2, 3}), "constructor wrong data length");
        checkThrows(() -> new NumJArray(new double[4], new int[]{3}, new int[]{2}), "constructor strides reach outside data");
        checkThrows(() -> new NumJArray(new double[3], new int[]{-1, 3}), "constructor negative dimension");
        checkThrows(() -> c.get(2, 0), "get out of bounds");
        checkThrows(() -> c.get(1), "get wrong number of indices");
        check(c.get(-1, -1) == 6, "get negative indices");
        check(c.ndim() == 2 && d.ndim() == 1, "ndim");
        check(NumJArray.of(42).item() == 42, "item");
        checkThrows(() -> c.item(), "item on multi-element array");

        // ---- Views: everything must work on a transposed array ----
        check(ct.toString().equals("shape=[3, 2], data=[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]"), "toString on view");
        check(ct.sum() == 21 && ct.min() == 1 && ct.max() == 6 && ct.mean() == 3.5, "reductions on view");
        check(arrayEquals(ct.mul(2), 2, 8, 4, 10, 6, 12), "scalar op on view keeps logical order");
        check(arrayEquals(ct.flatten(), 1, 4, 2, 5, 3, 6), "flatten on view");
        check(arrayEquals(ct.reshape(6), 1, 4, 2, 5, 3, 6), "reshape on non-contiguous view");
        check(arrayEquals(ct.copy(), 1, 4, 2, 5, 3, 6) && ct.copy().strides()[0] == 2, "copy of view is contiguous");
        check(arrayEquals(ct.matMul(c), 17, 22, 27, 22, 29, 36, 27, 36, 45), "matMul with transposed left side");
        check(arrayEquals(c.matMul(ct), 14, 32, 32, 77), "matMul with transposed right side");
        check(ct.trace() == 6, "trace on view");
        check(Arrays.equals(ct.argMax(), new int[]{2, 1}), "argMax on view");
        check(arrayEquals(ct.add(NumJArray.of(new double[][]{{1}, {2}, {3}})), 2, 5, 4, 7, 6, 9), "broadcast on view");
        a.transpose().set(50, 0, 1);
        check(a.get(1, 0) == 50, "set through view writes to original");
        a.set(3, 1, 0);

        // ---- Views: share memory vs copy ----
        NumJArray reshapeView = c.reshape(3, 2);
        reshapeView.set(-1, 0, 0);
        check(c.get(0, 0) == -1, "reshape of contiguous data is a view");
        c.set(1, 0, 0);
        check(Arrays.equals(c.reshape(-1, 2).shape(), new int[]{3, 2}), "reshape with -1");
        checkThrows(() -> c.reshape(4, 2), "reshape wrong element count");
        checkThrows(() -> c.reshape(-1, -1), "reshape two -1");
        check(arrayEquals(c.ravel(), 1, 2, 3, 4, 5, 6), "ravel");

        // ---- Axis reductions ----
        check(arrayEquals(c.sumAxis(1), 6, 15), "sumAxis(1)");
        check(arrayEquals(c.sumAxis(-1), 6, 15), "sumAxis(-1)");
        check(arrayEquals(ct.sumAxis(0), 6, 15), "sumAxis on view");
        check(NumJArray.of(1, 2, 3).sumAxis(0).item() == 6, "sumAxis on 1D gives scalar");
        check(arrayEquals(c.meanAxis(1), 2, 5), "meanAxis(1)");
        check(arrayEquals(c.prodAxis(1), 6, 120), "prodAxis");
        check(arrayEquals(c.minAxis(0), 1, 2, 3), "minAxis");
        check(arrayEquals(c.maxAxis(1), 3, 6), "maxAxis");
        check(arrayEquals(ct.maxAxis(1), 4, 5, 6), "maxAxis on view");
        check(arrayEquals(c.argMaxAxis(1), 2, 2), "argMaxAxis");
        check(arrayEquals(c.argMinAxis(0), 0, 0, 0), "argMinAxis");
        check(arrayEquals(c.cumSumAxis(1), 1, 3, 6, 4, 9, 15), "cumSumAxis(1)");
        check(arrayEquals(c.cumSumAxis(0), 1, 2, 3, 5, 7, 9), "cumSumAxis(0)");
        check(Math.abs(c.varAxis(1).get(0) - 2.0 / 3) < 1e-12, "varAxis");
        check(Math.abs(c.stdAxis(0).get(0) - 1.5) < 1e-12, "stdAxis");
        check(Math.abs(c.normAxis(1).get(0) - Math.sqrt(14)) < 1e-12, "normAxis");
        checkThrows(() -> c.sumAxis(2), "axis out of range");
        checkThrows(() -> c.sumAxis(-3), "negative axis out of range");

        // 3D axis reduction
        NumJArray cube = NumJArray.arange(24).reshape(2, 3, 4);
        check(cube.sumAxis(0).get(1, 2) == 6 + 18, "sumAxis 3D axis 0");
        check(cube.sumAxis(1).get(1, 3) == 15 + 19 + 23, "sumAxis 3D axis 1");
        check(cube.sumAxis(2).get(1, 2) == 20 + 21 + 22 + 23, "sumAxis 3D axis 2");
        check(cube.transpose(2, 0, 1).get(3, 1, 2) == 23, "transpose with axis order");

        // ---- Whole-array statistics ----
        check(c.prod() == 720, "prod");
        check(Math.abs(c.var() - 35.0 / 12) < 1e-12, "var");
        check(Math.abs(c.std() - Math.sqrt(35.0 / 12)) < 1e-12, "std");
        check(Math.abs(c.norm() - Math.sqrt(91)) < 1e-12, "norm");
        check(c.trace() == 6, "trace");
        check(Arrays.equals(c.argMax(), new int[]{1, 2}) && Arrays.equals(c.argMin(), new int[]{0, 0}), "argMax/argMin 2D");
        checkThrows(() -> NumJArray.zeros(0).min(), "min on empty");
        checkThrows(() -> NumJArray.zeros(0).argMax(), "argMax on empty");
        checkThrows(() -> d.trace(), "trace on 1D array");

        // ---- Element-wise ----
        check(arrayEquals(NumJArray.of(1, -2).neg(), -1, 2), "neg");
        check(arrayEquals(NumJArray.of(1, 2, 3).map(x -> x * x), 1, 4, 9), "map");
        check(arrayEquals(NumJArray.of(-1, 5).mod(3), 2, 2), "mod negative follows sign of divisor");
        check(arrayEquals(NumJArray.of(0.5, 1.5, 2.5).round(), 0, 2, 2), "round half to even");
        check(arrayEquals(NumJArray.of(-5, 0.5, 9).clip(0, 1), 0, 0.5, 1), "clip floats");
        checkThrows(() -> values.clip(5, 2), "clip min > max");

        // ---- Broadcasting ----
        check(arrayEquals(c.add(NumJArray.of(10, 20, 30)), 11, 22, 33, 14, 25, 36), "broadcast row vector");
        check(arrayEquals(c.sub(NumJArray.of(new double[][]{{1}, {2}})), 0, 1, 2, 2, 3, 4), "broadcast column vector");
        check(arrayEquals(NumJArray.of(new double[][]{{1}, {2}}).mul(NumJArray.of(10, 20)), 10, 20, 20, 40), "broadcast outer product");
        checkThrows(() -> c.add(NumJArray.of(1, 2)), "broadcast incompatible shapes");
        checkThrows(() -> a.matMul(c.reshape(3, 2)), "matMul inner dimension mismatch");
        checkThrows(() -> c.matMul(c), "matMul 2x3 times 2x3");

        // ---- Activations ----
        check(Math.abs(NumJArray.of(0).sigmoid().get(0) - 0.5) < 1e-12, "sigmoid(0)");
        check(Math.abs(NumJArray.of(0, 2).sigmoid().get(1) - 0.8807970779778823) < 1e-12, "sigmoid is element-wise");
        check(NumJArray.of(-1000).sigmoid().get(0) == 0 && NumJArray.of(1000).sigmoid().get(0) == 1, "sigmoid stable for large |x|");
        check(Math.abs(NumJArray.of(0).sigmoidD().get(0) - 0.25) < 1e-12, "sigmoidD(0)");
        check(Math.abs(NumJArray.of(0, 1).tanh().get(1) - Math.tanh(1)) < 1e-12, "tanh element-wise");
        check(Math.abs(NumJArray.of(0).tanhD().get(0) - 1) < 1e-12, "tanhD(0)");
        check(arrayEquals(NumJArray.of(-1, 0, 2).reluD(), 0, 0, 1), "reluD");

        NumJArray softAll = c.softmax();
        check(Math.abs(softAll.sum() - 1) < 1e-12, "softmax sums to 1");
        check(!Double.isNaN(NumJArray.of(1000, 1001).softmax().get(0)), "softmax stable for large values");

        NumJArray softRows = c.softmaxAxis(1);
        check(Math.abs(softRows.sumAxis(1).get(0) - 1) < 1e-12 && Math.abs(softRows.sumAxis(1).get(1) - 1) < 1e-12, "softmaxAxis rows sum to 1");
        check(Math.abs(c.softmaxAxis(0).sumAxis(0).get(2) - 1) < 1e-12, "softmaxAxis columns sum to 1");

        // ---- Shape manipulation ----
        check(c.swapAxes(0, 1).get(2, 1) == 6, "swapAxes");
        check(Arrays.equals(c.expandDims(1).shape(), new int[]{2, 1, 3}) && c.expandDims(-1).get(1, 2, 0) == 6, "expandDims");
        check(Arrays.equals(c.expandDims(0).squeeze().shape(), new int[]{2, 3}), "squeeze");
        check(arrayEquals(c.select(0, 1), 4, 5, 6), "select row");
        check(arrayEquals(c.select(1, -1), 3, 6), "select last column");
        checkThrows(() -> c.select(0, 2), "select index out of bounds");

        // ---- Slicing ----
        NumJArray stepped = c.slice(new int[]{0, 2}, new int[]{0, 3, 2});
        check(arrayEquals(stepped, 1, 3, 4, 6), "slice with step");
        check(arrayEquals(c.slice(null, new int[]{2, -1, -1}), 3, 2, 1, 6, 5, 4), "slice reversed");
        check(arrayEquals(c.slice(new int[]{1, 99}), 4, 5, 6), "slice clamps stop");
        check(c.slice(new int[]{5, 9}).size() == 0, "slice outside array is empty");
        check(arrayEquals(c.slice(new int[]{1, 2}), 4, 5, 6) && c.slice(new int[]{1, 2}).shape().length == 2, "slice with fewer args keeps rest");
        check(arrayEquals(stepped.transpose().slice(new int[]{1, 2}), 3, 6), "slice of a transposed slice");
        NumJArray sliceView = c.slice(new int[]{0, 1}, new int[]{0, 1});
        sliceView.set(99, 0, 0);
        check(c.get(0, 0) == 99, "slice is a view");
        c.set(1, 0, 0);
        checkThrows(() -> c.slice(new int[]{0, 1, 0}), "slice step 0");
        checkThrows(() -> c.slice(null, null, null), "slice too many dimensions");
        checkThrows(() -> c.slice(new int[]{1}), "slice with wrong entry count");

        // ---- Concat / stack / split on views and other axes ----
        check(arrayEquals(c.concat(c, 1), 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6), "concat axis 1");
        check(arrayEquals(ct.concat(ct, 1), 1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6), "concat on views");
        check(arrayEquals(c.stack(c.mul(2), 2), 1, 2, 2, 4, 3, 6, 4, 8, 5, 10, 6, 12), "stack on last axis");
        NumJArray[] colParts = c.split(1, 3);
        check(colParts.length == 3 && arrayEquals(colParts[1], 2, 5), "split columns");
        colParts[0].set(77, 0, 0);
        check(c.get(0, 0) == 77, "split parts are views");
        c.set(1, 0, 0);
        checkThrows(() -> c.split(1, 2), "split not divisible");
        checkThrows(() -> c.split(1, 0), "split zero sections");
        checkThrows(() -> c.concat(ct, 0), "concat shape mismatch");
        checkThrows(() -> c.stack(ct, 0), "stack shape mismatch");

        // ---- Dot ----
        checkThrows(() -> dotA.dot(c), "dot different shapes");

        NumJArray row = NumJArray.of(1, 5, 3);
        check(arrayEquals(c.equal(row), 1, 0, 1, 0, 1, 0), "equal (broadcast)");
        check(arrayEquals(c.notEqual(row), 0, 1, 0, 1, 0, 1), "notEqual");
        check(arrayEquals(c.greater(3), 0, 0, 0, 1, 1, 1), "greater (scalar)");
        check(arrayEquals(c.greaterEqual(3), 0, 0, 1, 1, 1, 1), "greaterEqual");
        check(arrayEquals(c.less(3), 1, 1, 0, 0, 0, 0), "less");
        check(arrayEquals(c.lessEqual(3), 1, 1, 1, 0, 0, 0), "lessEqual");
        check(arrayEquals(ct.greater(3), 0, 1, 0, 1, 0, 1), "greater on view");
        check(arrayEquals(c.less(row), 0, 1, 0, 0, 0, 0), "less (broadcast)");
        check(arrayEquals(NumJArray.of(Double.NaN, 1).equal(NumJArray.of(Double.NaN, 1)), 0, 1), "NaN is never equal");

        NumJArray p = NumJArray.of(1, 1, 0, 0);
        NumJArray q = NumJArray.of(1, 0, 1, 0);
        check(arrayEquals(p.logicalAnd(q), 1, 0, 0, 0), "logicalAnd");
        check(arrayEquals(p.logicalOr(q), 1, 1, 1, 0), "logicalOr");
        check(arrayEquals(p.logicalXor(q), 0, 1, 1, 0), "logicalXor");
        check(arrayEquals(p.logicalNot(), 0, 0, 1, 1), "logicalNot");

        check(c.all() && !p.all() && p.any() && !NumJArray.zeros(3).any(), "all / any");
        check(NumJArray.zeros(0).all() && !NumJArray.zeros(0).any(), "all / any on empty array");
        NumJArray boolGrid = NumJArray.of(new double[][]{{1, 0}, {1, 1}});
        check(arrayEquals(boolGrid.allAxis(1), 0, 1) && arrayEquals(boolGrid.allAxis(0), 1, 0), "allAxis");
        check(arrayEquals(boolGrid.anyAxis(1), 1, 1) && arrayEquals(boolGrid.anyAxis(-2), 1, 1), "anyAxis");

        check(arrayEquals(NumJArray.of(1, 2).add(1e-9).isClose(NumJArray.of(1, 2)), 1, 1), "isClose (tiny difference)");
        check(arrayEquals(NumJArray.of(1, 2).add(0.1).isClose(NumJArray.of(1, 2)), 0, 0), "isClose (large difference)");
        check(arrayEquals(
            NumJArray.of(Double.NaN, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY)
                .isClose(NumJArray.of(Double.NaN, Double.POSITIVE_INFINITY, 1)), 0, 1, 0), "isClose NaN / infinity");
        check(arrayEquals(NumJArray.of(100, 102).isClose(100, 0.01, 0), 1, 0), "isClose relative tolerance");
        check(c.allClose(c.add(1e-10)) && !c.allClose(c.add(1)) && c.allClose(c.add(1), 0, 1.5), "allClose");
        checkThrows(() -> c.isClose(c, -1, 0), "isClose negative tolerance");
        checkThrows(() -> c.equal(NumJArray.of(1, 2)), "comparison with bad shape");

        check(arrayEquals(NumJArray.of(1, 2, 3).dot(NumJArray.of(4, 5, 6)), 32), "dot 1D . 1D");
        check(NumJArray.of(1, 2, 3).dot(NumJArray.of(4, 5, 6)).ndim() == 0, "dot 1D . 1D is 0-D");
        check(NumJArray.of(1, 2, 3).inner(NumJArray.of(4, 5, 6)) == 32 && c.inner(c) == 91, "inner");
        checkThrows(() -> c.inner(ct), "inner different shapes");
        check(arrayEquals(c.dot(NumJArray.of(1, 0, 1)), 4, 10), "dot 2D . 1D");
        check(arrayEquals(NumJArray.of(1, 1).dot(c), 5, 7, 9), "dot 1D . 2D");
        check(arrayEquals(c.dot(ct), 14, 32, 32, 77), "dot 2D . 2D (view)");
        check(arrayEquals(c.dot(NumJArray.of(2, 3).sumAxis(0)), 5, 10, 15, 20, 25, 30), "dot with scalar multiplies");
        checkThrows(() -> c.dot(c), "dot misaligned shapes");
        checkThrows(() -> c.dot(NumJArray.of(1, 2)), "dot misaligned with vector");

        // N-D . M-D: result[i,j,l] = sum_k A[i,k] * B[j,k,l]
        NumJArray dotMatA = NumJArray.arange(6).reshape(2, 3);
        NumJArray dotMatB = NumJArray.arange(60).reshape(4, 3, 5);
        NumJArray dotMatR = dotMatA.dot(dotMatB);
        boolean dotOk = Arrays.equals(dotMatR.shape(), new int[]{2, 4, 5});
        for(int i = 0; i < 2 && dotOk; i++) for(int j = 0; j < 4; j++) for(int l = 0; l < 5; l++) {
            double expected = 0;
            for(int k = 0; k < 3; k++) expected += dotMatA.get(i, k) * dotMatB.get(j, k, l);
            if(dotMatR.get(i, j, l) != expected) dotOk = false;
        }
        check(dotOk, "dot 2D . 3D");

        // 3D . 1D: sums over last axis
        check(arrayEquals(cube.dot(NumJArray.of(1, 0, 0, 1)), 3, 11, 19, 27, 35, 43), "dot 3D . 1D");

        check(NumJArray.of(1, 2, 3).matMul(NumJArray.of(4, 5, 6)).item() == 32, "matMul 1D . 1D");
        check(arrayEquals(c.matMul(NumJArray.of(1, 0, 1)), 4, 10), "matMul 2D . 1D");
        check(arrayEquals(NumJArray.of(1, 1).matMul(c), 5, 7, 9), "matMul 1D . 2D");

        // batched matMul with broadcasting: [2,2,3] x [3,2] and [2,2,3] x [2,3,2]
        NumJArray batchA = NumJArray.arange(12).reshape(2, 2, 3);
        NumJArray sharedB = NumJArray.arange(6).reshape(3, 2);
        NumJArray stackedB = NumJArray.arange(12).reshape(2, 3, 2);
        NumJArray bm1 = batchA.matMul(sharedB);
        NumJArray bm2 = batchA.matMul(stackedB);
        boolean bmOk = Arrays.equals(bm1.shape(), new int[]{2, 2, 2}) && Arrays.equals(bm2.shape(), new int[]{2, 2, 2});
        for(int n = 0; n < 2 && bmOk; n++) for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
            double e1 = 0, e2 = 0;
            for(int k = 0; k < 3; k++) {
                e1 += batchA.get(n, i, k) * sharedB.get(k, j);
                e2 += batchA.get(n, i, k) * stackedB.get(n, k, j);
            }
            if(bm1.get(n, i, j) != e1 || bm2.get(n, i, j) != e2) bmOk = false;
        }
        check(bmOk, "batched matMul (shared and stacked right side)");
        check(Arrays.equals(NumJArray.arange(6).reshape(1, 2, 3).matMul(NumJArray.arange(12).reshape(2, 3, 2)).shape(), new int[]{2, 2, 2}), "matMul broadcasts batch size 1");
        checkThrows(() -> batchA.matMul(batchA), "batched matMul inner mismatch");
        checkThrows(() -> NumJArray.arange(12).reshape(3, 2, 2).matMul(NumJArray.arange(12).reshape(2, 2, 3)), "batched matMul batch mismatch");
        checkThrows(() -> NumJArray.of(2, 3).sumAxis(0).matMul(c), "matMul with scalar");

        NumJArray cube3 = NumJArray.arange(18).reshape(2, 3, 3);
        check(arrayEquals(cube3.trace(1, 2), 12, 39), "trace(axis1, axis2) on 3D");
        check(arrayEquals(cube3.diagonal(1, 2), 0, 4, 8, 9, 13, 17) && Arrays.equals(cube3.diagonal(1, 2).shape(), new int[]{2, 3}), "diagonal on 3D");
        check(arrayEquals(c.diagonal(0, 1), 1, 5), "diagonal on rectangular matrix");
        checkThrows(() -> cube3.trace(), "trace() on 3D");
        checkThrows(() -> c.diagonal(0, 0), "diagonal with the same axis twice");

        check(cube.sum() == 276, "sum() whole array");
        check(arrayEquals(cube.sum(0, 2), 60, 92, 124), "sum over axes 0 and 2");
        check(arrayEquals(cube.sum(-1, 0), 60, 92, 124), "sum with negative and unordered axes");
        check(arrayEquals(cube.mean(0, 2), 7.5, 11.5, 15.5), "mean over axes 0 and 2");
        check(arrayEquals(cube.min(0, 1), 0, 1, 2, 3), "min over axes 0 and 1");
        check(arrayEquals(cube.max(1, 2), 11, 23), "max over axes 1 and 2");
        check(arrayEquals(c.prod(0), 4, 10, 18), "prod over axis 0");
        check(cube.sum(0, 1, 2).item() == 276 && cube.sum(0, 1, 2).ndim() == 0, "sum over every axis gives 0-D");
        check(cube.sum(new int[0]).equals(cube), "sum with no axes returns a copy");
        checkThrows(() -> cube.sum(0, 0), "sum with repeated axis");
        checkThrows(() -> cube.sum(0, -3), "sum with repeated axis (negative)");
        checkThrows(() -> cube.sum(3), "sum with axis out of range");

        check(c.reduce((x, y) -> x * y) == 720, "reduce");
        check(c.reduce((x, y) -> x + y, 100) == 121, "reduce with init");
        check(arrayEquals(c.reduceAxis(1, (x, y) -> x - y), -4, -7), "reduceAxis starts from first element");
        check(arrayEquals(c.reduceAxis(1, (x, y) -> x - y, 100), 94, 85), "reduceAxis with init");
        check(arrayEquals(c.reduceAxis(0, Math::max), 4, 5, 6), "reduceAxis max");
        checkThrows(() -> NumJArray.zeros(0).reduce((x, y) -> x + y), "reduce on empty array");

        check(arrayEquals(c.elementwise(NumJArray.of(1, 2, 3), (x, y) -> x * 10 + y), 11, 22, 33, 41, 52, 63), "elementwise (array, op)");
        check(arrayEquals(NumJArray.of(1, 2, 3).elementwise(2, Math::pow), 1, 4, 9), "elementwise (scalar, op)");
        check(arrayEquals(NumJArray.of(1, 2, 3).elementwise(x -> x + 1), 2, 3, 4), "elementwise (unary)");
        check(c.argMaxFlat() == 5 && c.argMinFlat() == 0 && ct.argMaxFlat() == 5 && ct.argMinFlat() == 0, "argMaxFlat / argMinFlat");

        NumJArray w = c.copy();
        w.addi(1).muli(2).subi(2).divi(2);
        check(arrayEquals(w, 1, 2, 3, 4, 5, 6), "scalar in-place chain returns to original");
        w.addi(NumJArray.of(10, 20, 30));
        check(arrayEquals(w, 11, 22, 33, 14, 25, 36), "addi broadcast row");
        w.subi(NumJArray.of(new double[][]{{10}, {10}}));
        check(arrayEquals(w, 1, 12, 23, 4, 15, 26), "subi broadcast column");
        w.muli(NumJArray.of(2)).divi(NumJArray.of(2));
        check(arrayEquals(w, 1, 12, 23, 4, 15, 26), "muli / divi with [1] array");
        checkThrows(() -> NumJArray.of(1, 2).addi(NumJArray.of(1, 2, 3)), "addi cannot grow the array");
        checkThrows(() -> c.copy().addi(NumJArray.of(1, 2)), "addi incompatible shape");

        NumJArray base = NumJArray.zeros(3, 4);
        base.slice(new int[]{1, 3}, new int[]{0, 2}).assign(NumJArray.of(new double[][]{{1, 2}, {3, 4}}));
        check(arrayEquals(base, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0), "assign into a slice");
        base.slice(null, new int[]{3, 4}).assign(NumJArray.of(9));
        check(base.get(0, 3) == 9 && base.get(2, 3) == 9 && base.get(1, 2) == 0, "assign broadcasts a [1] array");
        base.slice(new int[]{0, 1}).addi(5);
        check(base.get(0, 0) == 5 && base.get(1, 0) == 1, "in-place op on slice writes through");
        base.transpose().assign(NumJArray.zeros(4, 3));
        check(base.sum() == 0, "assign through a transposed view");

        NumJArray shifting = NumJArray.of(1, 2, 3, 4, 5);
        shifting.slice(new int[]{1, 5}).assign(shifting.slice(new int[]{0, 4}));
        check(arrayEquals(shifting, 1, 1, 2, 3, 4), "assign with overlapping source and destination");

        check(arrayEquals(NumJArray.zeros(2, 2).fill(5), 5, 5, 5, 5), "fill(value)");
        check(arrayEquals(NumJArray.zeros(2, 3).fill(co -> co[0] * 10 + co[1]), 0, 1, 2, 10, 11, 12), "fill(coordinates)");
        check(NumJArray.zeros(3, 3).fill(co -> co[0] == co[1] ? 1 : 0).equals(NumJArray.identity(3)), "fill builds identity");
        check(arrayEquals(NumJArray.of(1, 2, 3).elementwisei(x -> x * x), 1, 4, 9), "elementwisei (unary)");
        check(arrayEquals(NumJArray.of(1, 2, 3).elementwisei(10, (x, y) -> x + y), 11, 12, 13), "elementwisei (scalar, op)");
        NumJArray tFill = c.copy().transpose();
        tFill.fill(co -> co[0] * 10 + co[1]);
        check(tFill.get(2, 1) == 21 && tFill.transpose().get(1, 2) == 21, "fill on a view");

        NumJArray mixed = NumJArray.of(-2, 3, -1, 4);
        check(arrayEquals(mixed.copy().putWhere(mixed.less(0), 0), 0, 3, 0, 4), "putWhere(mask, value)");
        check(arrayEquals(mixed.copy().putWhere(mixed.less(0), mixed.abs()), 2, 3, 1, 4), "putWhere(mask, values)");
        check(arrayEquals(mixed.copy().conditionalPut(v -> v > 3, 100), -2, 3, -1, 100), "conditionalPut");
        check(arrayEquals(mixed.where(v -> v > 0), 0, 1, 0, 1), "where(predicate)");
        check(arrayEquals(NumJArray.where(mixed.greater(0), mixed, mixed.mul(0)), 0, 3, 0, 4), "where(cond, x, y)");
        check(arrayEquals(NumJArray.where(mixed.greater(0), 1, -1), -1, 1, -1, 1), "where(cond, scalar, scalar)");
        check(arrayEquals(NumJArray.where(NumJArray.of(1, 0), NumJArray.of(new double[][]{{1}, {2}}), NumJArray.of(0)), 1, 0, 2, 0), "where broadcasts all three");
        check(arrayEquals(c.copy().putWhere(NumJArray.of(1, 0, 1), 0), 0, 2, 0, 0, 5, 0), "putWhere broadcasts the mask");
        checkThrows(() -> mixed.copy().putWhere(NumJArray.of(1, 0), 0), "putWhere mask cannot be broadcast");

        check(ct.getFlat(1) == 4 && ct.getFlat(-1) == 6 && c.getFlat(4) == 5, "getFlat (including a view)");
        NumJArray flatSet = c.copy();
        flatSet.transpose().setFlat(50, 1);
        check(flatSet.get(1, 0) == 50, "setFlat through a view");
        check(c.pointer(3) == 3 && ct.pointer(1) == 3, "pointer");
        checkThrows(() -> c.getFlat(6), "getFlat out of bounds");
        checkThrows(() -> c.getFlat(-7), "getFlat negative out of bounds");

        check(c.canBroadcast(NumJArray.of(1, 2, 3)) && !c.canBroadcast(NumJArray.of(1, 2)), "canBroadcast");
        NumJArray stretched = NumJArray.of(1, 2, 3).broadcastTo(2, 3);
        check(arrayEquals(stretched, 1, 2, 3, 1, 2, 3) && stretched.strides()[0] == 0, "broadcastTo is a stride-0 view");
        check(arrayEquals(NumJArray.of(new double[][]{{1}, {2}}).broadcastTo(2, 2), 1, 1, 2, 2), "broadcastTo a column");
        checkThrows(() -> NumJArray.of(1, 2, 3).broadcastTo(2, 2), "broadcastTo incompatible");
        checkThrows(() -> c.broadcastTo(3), "broadcastTo fewer dimensions");
        check(NumJArray.of(1, 2, 3).broadcastGet(5, 2) == 3 && NumJArray.of(new double[][]{{1}, {2}}).broadcastGet(1, 7) == 2, "broadcastGet");
        checkThrows(() -> c.broadcastGet(1), "broadcastGet too few coordinates");
        checkThrows(() -> NumJArray.of(1, 2, 3).broadcastGet(3), "broadcastGet out of bounds");

        check(c.equals(c.copy()) && c.hashCode() == c.copy().hashCode(), "equals / hashCode on copies");
        check(ct.equals(c.transpose().copy()) && ct.hashCode() == ct.copy().hashCode(), "equals / hashCode on views");
        check(!c.equals(c.reshape(3, 2)) && !c.equals(c.add(1)) && !c.equals("text") && !c.equals(null), "not equal cases");
        check(NumJArray.of(Double.NaN).equals(NumJArray.of(Double.NaN)), "equals treats NaN as equal to itself");

        check(c.toPrettyString().equals("[[1.0, 2.0, 3.0],\n [4.0, 5.0, 6.0]]"), "toPrettyString 2D");
        check(NumJArray.of(1, 10).toPrettyString().equals("[ 1.0, 10.0]"), "toPrettyString aligns columns");
        check(NumJArray.arange(8).reshape(2, 2, 2).toPrettyString().equals("[[[0.0, 1.0],\n  [2.0, 3.0]],\n\n [[4.0, 5.0],\n  [6.0, 7.0]]]"), "toPrettyString 3D");
        check(NumJArray.of(2, 3).sumAxis(0).toPrettyString().equals("5.0") && NumJArray.zeros(0).toPrettyString().equals("[]"), "toPrettyString 0-D and empty");

        NumJArray sparse = NumJArray.of(new double[][]{{1, 0}, {1, 1}});
        check(NumJArray.of(0, 1, 0, 3).countNonzero() == 2, "countNonzero");
        check(arrayEquals(sparse.countNonzero(1), 1, 2) && arrayEquals(sparse.countNonzero(0), 2, 1), "countNonzero(axis)");
        check(NumJArray.of(Double.NaN, 0).countNonzero() == 1, "countNonzero counts NaN");

        int[][] nz = sparse.nonzero();
        check(Arrays.equals(nz[0], new int[]{0, 1, 1}) && Arrays.equals(nz[1], new int[]{0, 0, 1}), "nonzero");
        check(Arrays.equals(sparse.transpose().nonzero()[0], new int[]{0, 0, 1}), "nonzero on view");
        NumJArray aw = sparse.argwhere();
        check(Arrays.equals(aw.shape(), new int[]{3, 2}) && arrayEquals(aw, 0, 0, 1, 0, 1, 1), "argwhere");
        check(NumJArray.zeros(2, 2).argwhere().shape()[0] == 0, "argwhere with no matches");
        checkThrows(() -> NumJArray.of(2, 3).sumAxis(0).nonzero(), "nonzero on 0-D");

        check(arrayEquals(c.take(new int[]{5, 0, -1}), 6, 1, 6), "take (flat)");
        check(arrayEquals(ct.take(new int[]{1, 2}), 4, 2), "take (flat) on view");
        check(arrayEquals(c.take(1, 2, 0), 3, 1, 6, 4) && Arrays.equals(c.take(1, 2, 0).shape(), new int[]{2, 2}), "take along axis 1");
        check(Arrays.equals(c.take(0, 1, 1, 0).shape(), new int[]{3, 3}) && arrayEquals(c.take(0, 1, 1, 0), 4, 5, 6, 4, 5, 6, 1, 2, 3), "take along axis 0 with repeats");
        check(arrayEquals(c.take(-1, -1), 3, 6), "take with negative axis and index");
        check(arrayEquals(ct.take(0, 2), 3, 6), "take along axis on view");
        checkThrows(() -> c.take(1, 3), "take axis index out of bounds");
        checkThrows(() -> c.take(new int[]{6}), "take flat index out of bounds");

        NumJArray putTarget = c.copy();
        putTarget.put(new int[]{0, -1}, 9);
        check(arrayEquals(putTarget, 9, 2, 3, 4, 5, 9), "put(indices, value)");
        check(arrayEquals(NumJArray.zeros(5).put(new int[]{0, 1, 2, 3}, NumJArray.of(7, 8)), 7, 8, 7, 8, 0), "put(indices, values) repeats values");
        NumJArray putView = c.copy();
        putView.transpose().put(new int[]{1}, NumJArray.of(50));
        check(putView.get(1, 0) == 50, "put through a view");
        NumJArray putSafe = c.copy();
        checkThrows(() -> putSafe.put(new int[]{0, 99}, 1), "put with bad index");
        check(putSafe.get(0, 0) == 1, "put writes nothing when an index is bad");

        NumJArray unsorted = NumJArray.of(new double[][]{{5, 1}, {2, 7}});
        check(arrayEquals(NumJArray.of(3, 1, 2).sort(), 1, 2, 3), "sort 1D");
        check(arrayEquals(unsorted.sort(), 1, 5, 2, 7) && arrayEquals(unsorted.sort(1), 1, 5, 2, 7), "sort last axis");
        check(arrayEquals(unsorted.sort(0), 2, 1, 5, 7), "sort axis 0");
        check(arrayEquals(unsorted.transpose().sort(1), 2, 5, 1, 7), "sort on view");
        check(arrayEquals(unsorted, 5, 1, 2, 7), "sort does not change the original");
        check(arrayEquals(unsorted.sortFlat(), 1, 2, 5, 7), "sortFlat");
        NumJArray sortedNaN = NumJArray.of(3, Double.NaN, 1).sort();
        check(sortedNaN.get(0) == 1 && sortedNaN.get(1) == 3 && Double.isNaN(sortedNaN.get(2)), "sort puts NaN last");
        checkThrows(() -> NumJArray.of(2, 3).sumAxis(0).sort(), "sort on 0-D");

        check(arrayEquals(NumJArray.of(3, 1, 2).argsort(), 1, 2, 0), "argsort");
        check(arrayEquals(NumJArray.of(2, 1, 2, 1).argsort(), 1, 3, 0, 2), "argsort is stable for ties");
        check(arrayEquals(unsorted.argsort(0), 1, 0, 0, 1) && arrayEquals(unsorted.argsort(1), 1, 0, 0, 1), "argsort on axes");
        check(arrayEquals(unsorted.argsortFlat(), 1, 2, 0, 3), "argsortFlat");
        check(arrayEquals(NumJArray.of(Double.NaN, 1).argsort(), 1, 0), "argsort puts NaN last");
        check(arrayEquals(unsorted.transpose().argsort(0), 1, 0, 0, 1), "argsort on view");
        NumJArray order = NumJArray.of(30, 10, 20).argsort();
        check(arrayEquals(NumJArray.of(30, 10, 20).take(0, 1, 2, 0), 10, 20, 30), "take with argsort order");
        check(order.get(0) == 1, "argsort result usable as indices");

        check(arrayEquals(NumJArray.of(3, 1, 3, 2, 1).unique(), 1, 2, 3), "unique");
        NumJArray[] uc = NumJArray.of(3, 1, 3, 2, 1).uniqueCounts();
        check(arrayEquals(uc[0], 1, 2, 3) && arrayEquals(uc[1], 2, 1, 2), "uniqueCounts");
        check(arrayEquals(unsorted.unique(), 1, 2, 5, 7), "unique flattens N-D");
        check(NumJArray.of(Double.NaN, Double.NaN, 1).unique().size() == 2, "unique merges NaNs");
        check(NumJArray.of(0, -0.0).unique().size() == 1, "unique merges 0.0 and -0.0");
        check(NumJArray.zeros(0).unique().size() == 0, "unique on empty");

        check(arrayEquals(NumJArray.of(1, 2).repeat(2), 1, 1, 2, 2), "repeat flat");
        check(arrayEquals(c.repeat(2, 0), 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6) && Arrays.equals(c.repeat(2, 0).shape(), new int[]{4, 3}), "repeat axis 0");
        check(arrayEquals(c.repeat(2, 1), 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6), "repeat axis 1");
        check(arrayEquals(NumJArray.of(1, 2, 3).repeat(new int[]{1, 0, 2}, 0), 1, 3, 3), "repeat per element");
        check(c.repeat(2).size() == 12 && NumJArray.of(1, 2).repeat(0).size() == 0, "repeat sizes");
        check(arrayEquals(ct.repeat(2, 1), 1, 1, 4, 4, 2, 2, 5, 5, 3, 3, 6, 6), "repeat on view");
        checkThrows(() -> c.repeat(-1, 0), "repeat negative");
        checkThrows(() -> c.repeat(new int[]{1}, 0), "repeat wrong count length");

        check(arrayEquals(NumJArray.of(1, 2).tile(3), 1, 2, 1, 2, 1, 2), "tile 1D");
        NumJArray tiled = NumJArray.of(1, 2).tile(2, 2);
        check(Arrays.equals(tiled.shape(), new int[]{2, 4}) && arrayEquals(tiled, 1, 2, 1, 2, 1, 2, 1, 2), "tile adds leading dimension");
        check(Arrays.equals(c.tile(2).shape(), new int[]{2, 6}) && arrayEquals(c.tile(2), 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6), "tile pads reps in front");
        check(Arrays.equals(c.tile(2, 1).shape(), new int[]{4, 3}) && arrayEquals(c.tile(2, 1), 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6), "tile along rows");
        check(Arrays.equals(NumJArray.of(1, 2).tile(2, 1, 2).shape(), new int[]{2, 1, 4}), "tile with more reps than dimensions");
        check(arrayEquals(ct.tile(1, 2), 1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6), "tile on view");
        check(c.tile(0).size() == 0 && c.tile().equals(c), "tile(0) is empty, tile() copies");
        checkThrows(() -> c.tile(-1), "tile negative");

        check(arrayEquals(NumJArray.of(1, 2, 3).pad(new int[][]{{1, 2}}), 0, 1, 2, 3, 0, 0), "pad 1D");
        check(Arrays.equals(c.pad(new int[][]{{1, 0}, {2, 1}}).shape(), new int[]{3, 6}) && arrayEquals(c.pad(new int[][]{{1, 0}, {2, 1}}), 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 4, 5, 6, 0), "pad 2D asymmetric");
        check(Arrays.equals(ct.pad(new int[][]{{0, 0}, {1, 0}}).shape(), new int[]{3, 3}) && arrayEquals(ct.pad(new int[][]{{0, 0}, {1, 0}}), 0, 1, 4, 0, 2, 5, 0, 3, 6), "pad transposed view");
        check(arrayEquals(NumJArray.of(1, 2, 3).pad(new int[][]{{0, 0}}), 1, 2, 3), "pad zero padding");
        checkThrows(() -> NumJArray.of(1, 2, 3).pad(new int[][]{{1}}), "pad invalid padding pair");
        checkThrows(() -> NumJArray.of(1, 2, 3).pad(new int[][]{{-1, 0}}), "pad negative padding");
        checkThrows(() -> NumJArray.of(1, 2, 3).pad(new int[][]{{0, 0}, {0, 0}}), "pad dimension mismatch");

        check(arrayEquals(NumJArray.of(1, 2, 3).flip(), 3, 2, 1), "flip 1D");
        check(arrayEquals(c.flip(), 6, 5, 4, 3, 2, 1), "flip all axes");
        check(arrayEquals(c.flip(0), 4, 5, 6, 1, 2, 3) && arrayEquals(c.flip(1), 3, 2, 1, 6, 5, 4), "flip single axis");
        check(arrayEquals(c.flip(0, 1), 6, 5, 4, 3, 2, 1) && arrayEquals(c.flip(-1), 3, 2, 1, 6, 5, 4), "flip several / negative axes");
        NumJArray flipBase = c.copy();
        flipBase.flip(1).set(99, 0, 0);
        check(flipBase.get(0, 2) == 99, "flip is a view");
        NumJArray flipped = c.flip(1);
        check(flipped.sum() == 21 && arrayEquals(flipped.sumAxis(0), 9, 7, 5) && arrayEquals(flipped.sumAxis(1), 6, 15), "reductions on flipped view");
        check(arrayEquals(flipped.transpose().reshape(6), 3, 6, 2, 5, 1, 4), "reshape of flipped transposed view");
        check(arrayEquals(flipped.slice(new int[]{0, 2}, new int[]{0, 2}), 3, 2, 6, 5), "slice of flipped view");
        check(arrayEquals(flipped.flip(1), 1, 2, 3, 4, 5, 6), "flip twice restores");
        check(arrayEquals(flipped.matMul(NumJArray.of(1, 0, 0)), 3, 6), "matMul with flipped view");
        check(flipped.argMaxFlat() == 3 && flipped.getFlat(0) == 3, "flat access on flipped view");
        checkThrows(() -> c.flip(0, 0), "flip repeated axis");
        checkThrows(() -> c.flip(2), "flip axis out of range");

        NumJArray fiveVals = NumJArray.of(1, 2, 3, 4, 5);
        check(arrayEquals(fiveVals.roll(2), 4, 5, 1, 2, 3) && arrayEquals(fiveVals.roll(-1), 2, 3, 4, 5, 1), "roll flat");
        check(arrayEquals(fiveVals.roll(7), 4, 5, 1, 2, 3) && arrayEquals(fiveVals.roll(0), 1, 2, 3, 4, 5), "roll wraps / zero");
        check(arrayEquals(c.roll(1), 6, 1, 2, 3, 4, 5) && Arrays.equals(c.roll(1).shape(), new int[]{2, 3}), "roll keeps shape");
        check(arrayEquals(c.roll(1, 0), 4, 5, 6, 1, 2, 3) && arrayEquals(c.roll(1, 1), 3, 1, 2, 6, 4, 5), "roll along axis");
        check(arrayEquals(c.roll(-1, -1), 2, 3, 1, 5, 6, 4), "roll negative shift and axis");
        check(arrayEquals(ct.roll(1, 0), 3, 6, 1, 4, 2, 5), "roll on view");
        check(NumJArray.zeros(0).roll(3).size() == 0, "roll on empty");


        check(arrayEquals(c.cumSum(), 1, 3, 6, 10, 15, 21) && c.cumSum().ndim() == 1, "cumSum flat");
        check(arrayEquals(c.cumProd(), 1, 2, 6, 24, 120, 720), "cumProd flat");
        check(arrayEquals(c.cumProdAxis(1), 1, 2, 6, 4, 20, 120) && arrayEquals(c.cumProdAxis(0), 1, 2, 3, 4, 10, 18), "cumProdAxis");
        check(arrayEquals(ct.cumSum(), 1, 5, 7, 12, 15, 21), "cumSum on view");

        check(arrayEquals(NumJArray.of(1, 4, 9, 16).diff(), 3, 5, 7), "diff");
        check(arrayEquals(NumJArray.of(1, 4, 9, 16).diff(2, 0), 2, 2), "diff n=2");
        check(Arrays.equals(c.diff().shape(), new int[]{2, 2}) && arrayEquals(c.diff(), 1, 1, 1, 1), "diff last axis");
        check(Arrays.equals(c.diff(1, 0).shape(), new int[]{1, 3}) && arrayEquals(c.diff(1, 0), 3, 3, 3), "diff axis 0");
        check(arrayEquals(ct.diff(1, 0), 1, 1, 1, 1), "diff on view");
        check(NumJArray.of(5).diff().size() == 0 && NumJArray.of(1, 2, 3).diff(5, 0).size() == 0, "diff shorter than n is empty");
        check(arrayEquals(NumJArray.of(1, 2).diff(0, 0), 1, 2), "diff n=0 copies");
        checkThrows(() -> c.diff(-1, 0), "diff negative n");

        check(NumJArray.of(3, 1, 2).median() == 2 && NumJArray.of(1, 2, 3, 4).median() == 2.5, "median");
        check(arrayEquals(c.median(1), 2, 5) && arrayEquals(c.median(0), 2.5, 3.5, 4.5), "median(axis)");
        check(arrayEquals(ct.median(0), 2, 5), "median on view");
        NumJArray medianSrc = NumJArray.of(3, 1, 2);
        medianSrc.median();
        check(arrayEquals(medianSrc, 3, 1, 2), "median does not reorder the original");
        check(NumJArray.of(1, 2, 3, 4, 5).quantile(0.25) == 2 && NumJArray.of(0, 10).quantile(0.1) == 1, "quantile");
        check(NumJArray.of(1, 2, 3, 4, 5).quantile(0) == 1 && NumJArray.of(1, 2, 3, 4, 5).quantile(1) == 5, "quantile ends");
        check(NumJArray.of(1, 2, 3, 4, 5).percentile(50) == 3 && NumJArray.of(1, 2, 3, 4, 5).percentile(25) == 2, "percentile");
        check(arrayEquals(c.percentile(100, 1), 3, 6) && arrayEquals(c.quantile(0, 0), 1, 2, 3), "quantile / percentile along axis");
        check(Double.isNaN(NumJArray.of(1, Double.NaN, 3).median()), "median with NaN is NaN");
        check(NumJArray.of(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY).median() == Double.POSITIVE_INFINITY, "median of infinities");
        checkThrows(() -> c.quantile(1.5), "quantile out of range");
        checkThrows(() -> c.percentile(-1), "percentile out of range");
        checkThrows(() -> NumJArray.zeros(0).median(), "median of empty");

        check(NumJArray.eye(3).equals(NumJArray.identity(3)), "eye(n)");
        check(Arrays.equals(NumJArray.eye(2, 3).shape(), new int[]{2, 3}) && arrayEquals(NumJArray.eye(2, 3), 1, 0, 0, 0, 1, 0), "eye(n, m)");
        check(arrayEquals(NumJArray.eye(3, 3, 1), 0, 1, 0, 0, 0, 1, 0, 0, 0), "eye with k=1");
        check(arrayEquals(NumJArray.eye(3, 3, -1), 0, 0, 0, 1, 0, 0, 0, 1, 0), "eye with k=-1");
        check(arrayEquals(NumJArray.eye(2, 2, 5), 0, 0, 0, 0), "eye with k outside matrix");
        checkThrows(() -> NumJArray.eye(-1), "eye negative");

        check(arrayEquals(NumJArray.of(1, 2, 3).diag(), 1, 0, 0, 0, 2, 0, 0, 0, 3), "diag builds matrix");
        check(arrayEquals(NumJArray.of(1, 2).diag(1), 0, 1, 0, 0, 0, 2, 0, 0, 0) && arrayEquals(NumJArray.of(1, 2).diag(-1), 0, 0, 0, 1, 0, 0, 0, 2, 0), "diag builds off-diagonal matrix");
        NumJArray nine = NumJArray.arange(9).reshape(3, 3);
        check(arrayEquals(nine.diag(), 0, 4, 8) && arrayEquals(nine.diag(1), 1, 5) && arrayEquals(nine.diag(-1), 3, 7), "diag extracts diagonals");
        check(arrayEquals(c.diag(), 1, 5) && arrayEquals(c.diag(1), 2, 6) && arrayEquals(c.diag(2), 3) && arrayEquals(c.diag(-1), 4), "diag on rectangular matrix");
        check(c.diag(3).size() == 0 && c.diag(-2).size() == 0, "diag outside matrix is empty");
        check(arrayEquals(ct.diag(), 1, 5), "diag on view");
        NumJArray diagCopy = nine.copy();
        diagCopy.diag().set(100, 0);
        check(diagCopy.get(0, 0) == 0, "diag extraction is a copy");
        checkThrows(() -> cube.diag(), "diag on 3D");

        NumJArray oneToNine = NumJArray.arange(1, 10).reshape(3, 3);
        check(arrayEquals(oneToNine.tril(), 1, 0, 0, 4, 5, 0, 7, 8, 9), "tril");
        check(arrayEquals(oneToNine.triu(), 1, 2, 3, 0, 5, 6, 0, 0, 9), "triu");
        check(arrayEquals(oneToNine.tril(-1), 0, 0, 0, 4, 0, 0, 7, 8, 0), "tril k=-1");
        check(arrayEquals(oneToNine.triu(1), 0, 2, 3, 0, 0, 6, 0, 0, 0), "triu k=1");
        check(arrayEquals(c.tril(), 1, 0, 0, 4, 5, 0) && arrayEquals(c.triu(), 1, 2, 3, 0, 5, 6), "tril / triu on rectangular matrix");
        check(arrayEquals(NumJArray.arange(1, 9).reshape(2, 2, 2).tril(), 1, 0, 3, 4, 5, 0, 7, 8), "tril on a batch of matrices");
        check(arrayEquals(ct.tril(), 1, 0, 2, 5, 3, 6), "tril on view");
        check(oneToNine.tril().add(oneToNine.triu(1)).equals(oneToNine), "tril + triu(1) rebuilds the matrix");
        checkThrows(() -> NumJArray.of(1, 2, 3).tril(), "tril on 1D");


        NumJArray outerResult = NumJArray.of(1, 2).outer(NumJArray.of(3, 4, 5));
        check(Arrays.equals(outerResult.shape(), new int[]{2, 3}) && arrayEquals(outerResult, 3, 4, 5, 6, 8, 10), "outer");
        check(Arrays.equals(c.outer(NumJArray.of(1, 2)).shape(), new int[]{6, 2}) && ct.outer(NumJArray.of(1)).get(1, 0) == 4, "outer flattens inputs (including views)");

        NumJArray special = NumJArray.of(1, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        check(arrayEquals(special.isNaN(), 0, 1, 0, 0), "isNaN");
        check(arrayEquals(special.isInf(), 0, 0, 1, 1), "isInf");
        check(arrayEquals(special.isFinite(), 1, 0, 0, 0), "isFinite");

        check(arrayEquals(NumJArray.of(1, 5, 3).maximum(NumJArray.of(4, 2, 3)), 4, 5, 3), "maximum");
        check(arrayEquals(NumJArray.of(1, 5, 3).minimum(NumJArray.of(4, 2, 3)), 1, 2, 3), "minimum");
        check(arrayEquals(c.maximum(NumJArray.of(3, 3, 3)), 3, 3, 3, 4, 5, 6), "maximum broadcasts");
        check(arrayEquals(c.minimum(2.5), 1, 2, 2.5, 2.5, 2.5, 2.5) && arrayEquals(NumJArray.of(-1, 2).maximum(0), 0, 2), "maximum / minimum with scalar");
        check(Double.isNaN(NumJArray.of(Double.NaN, 1).maximum(NumJArray.of(0, 0)).get(0)), "maximum propagates NaN");
        check(arrayEquals(ct.maximum(3), 3, 4, 3, 5, 3, 6), "maximum on view");
        checkThrows(() -> c.maximum(NumJArray.of(1, 2)), "maximum with bad shape");

        try {
            Path njFile = Files.createTempFile("numj-test", ".nj");
            Path npyFile = Files.createTempFile("numj-test", ".npy");
        
            try {
                NumJArray original = NumJArray.of(1, 2, 3, 4, 5, 6).reshape(2, 3);
                original.save(njFile.toString());
                NumJArray loaded = NumJArray.load(njFile.toString());
                check(Arrays.equals(loaded.shape(), new int[]{2, 3}) && arrayEquals(loaded, 1, 2, 3, 4, 5, 6),"save/load NJ");
                NumJArray replacement = NumJArray.of(7, 8, 9);
                replacement.save(njFile.toString());
                NumJArray overwritten = NumJArray.load(njFile.toString());
                check(Arrays.equals(overwritten.shape(), new int[]{3}) && arrayEquals(overwritten, 7, 8, 9),"save overwrites NJ");
                // NPY — SHOULD WORK, BUT HAVEN'T FULLY TESTED
                original.saveNpy(npyFile.toString());
                NumJArray loadedNpy = NumJArray.loadNpy(npyFile.toString());
                check(Arrays.equals(loadedNpy.shape(), new int[]{2, 3}) && arrayEquals(loadedNpy, 1, 2, 3, 4, 5, 6),"saveNpy/loadNpy");
                NumJArray replacementNpy = NumJArray.of(9, 8, 7);
                replacementNpy.saveNpy(npyFile.toString());
                NumJArray overwrittenNpy = NumJArray.loadNpy(npyFile.toString());
                check(Arrays.equals(overwrittenNpy.shape(), new int[]{3}) && arrayEquals(overwrittenNpy, 9, 8, 7),"saveNpy overwrites NPY");
                checkThrows(() -> original.save("invalid.txt"),"save invalid extension");
                checkThrows(() -> NumJArray.load("invalid.txt"),"load invalid extension");
                checkThrows(() -> original.saveNpy("invalid.txt"),"saveNpy invalid extension");
                checkThrows(() -> NumJArray.loadNpy("invalid.txt"),"loadNpy invalid extension");
            } finally {Files.deleteIfExists(njFile); Files.deleteIfExists(npyFile);}
        } catch (java.io.IOException e1) {throw new RuntimeException("Save/load tests failed", e1);}

        System.out.println("All tests passed!");

    }

    public static void check(boolean condition, String name) {

        if(!condition) {
            throw new RuntimeException("Test failed: " + name);
        }

        System.out.println("Passed: " + name);
    }

    /** Passes only if running {@code action} throws a RuntimeException. */
    public static void checkThrows(Runnable action, String name) {

        try {
            action.run();
        }
        catch(RuntimeException expected) {
            System.out.println("Passed: " + name + " (threw " + expected.getClass().getSimpleName() + ")");
            return;
        }

        throw new RuntimeException("Test failed (expected an exception): " + name);
    }

    /** True if the array's logical values (in row-major order) equal {@code expected}. */
    public static boolean arrayEquals(NumJArray array, double... expected) {

        double[] actual = array.copy().data;

        if(actual.length != expected.length) {
            return false;
        }

        for(int i = 0; i < actual.length; i++) {
            if(Math.abs(actual[i] - expected[i]) > 1e-9) {
                return false;
            }
        }

        return true;
    }
}