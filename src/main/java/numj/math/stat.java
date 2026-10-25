package numj.math;

import java.util.Arrays;
import java.util.function.BiFunction;

import numj.core.NumJArray;

public class stat {

    /**
     * Sum of all elements.
     *
     * @return the sum ({@code 0.0} for an empty array)
     */
    public static float sum(NumJArray array) {
        float total = 0;
        for (float v : array.readOnlyData()) total += v;
        return total;
    }

    /**
     * Product of all elements.
     *
     * @return the product ({@code 1.0} for an empty array)
     */
    public static float prod(NumJArray array) {
        float total = 1;
        for (float v : array.readOnlyData()) total *= v;
        return total;
    }

    /**
     * Arithmetic mean of all elements.
     *
     * @return the mean ({@code NaN} for an empty array)
     */
    public static float mean(NumJArray array) {
        return sum(array) / array.size();
    }

    /**
     * Smallest element.
     *
     * @return the minimum value
     * @throws IllegalStateException if the array is empty
     */
    public static float min(NumJArray array) {
        array.requireNonEmpty("min");
        float[] d = array.readOnlyData();
        float best = d[0];
        for (int i = 1; i < d.length; i++) best = Math.min(best, d[i]);
        return best;
    }

    /**
     * Largest element.
     *
     * @return the maximum value
     * @throws IllegalStateException if the array is empty
     */
    public static float max(NumJArray array) {
        array.requireNonEmpty("max");
        float[] d = array.readOnlyData();
        float best = d[0];
        for (int i = 1; i < d.length; i++) best = Math.max(best, d[i]);
        return best;
    }

    /**
     * Population variance of all elements (divides by N, like NumPys default).
     *
     * @return the variance ({@code NaN} for an empty array)
     */
    public static float var(NumJArray array) {
        float mean = mean(array);
        float acc = 0;
        for (float v : array.readOnlyData()) acc += (v - mean) * (v - mean);
        return acc / array.size();
    }

    /**
     * Population standard deviation of all elements.
     *
     * @return the square root of {@link #var()}
     */
    public static float std(NumJArray array) {
        return (float) Math.sqrt(var(array));
    }

    /**
     * Finds the position of the largest element.
     *
     * @return the coordinates (one per dimension) of the first maximum
     * @throws IllegalStateException if the array is empty
     */
    public static int[] argMax(NumJArray array) {
        return array.unravel(array.argFlat(true));
    }

    /**
     * Finds the position of the smallest element.
     *
     * @return the coordinates (one per dimension) of the first minimum
     * @throws IllegalStateException if the array is empty
     */
    public static int[] argMin(NumJArray array) {
        return array.unravel(array.argFlat(false));
    }

    /**
     * Sums along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end ({@code -1} = last)
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray sumAxis(NumJArray array, int axis) {
        return array.reduceAxis(axis, Float::sum, 0.0f);
    }

    /**
     * Multiplies along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray prodAxis(NumJArray array, int axis) {
        return array.reduceAxis(axis, (x, y) -> x * y, 1.0f);
    }

    /**
     * Averages along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray meanAxis(NumJArray array, int axis) {
        int ax = array.normalizeAxis(axis);
        return sumAxis(array, ax).div(array.shape[ax]);
    }

    /**
     * Minimum along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public static NumJArray minAxis(NumJArray array, int axis) {
        return array.reduceAxis(axis, Math::min);
    }

    /**
     * Maximum along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public static NumJArray maxAxis(NumJArray array, int axis) {
        return array.reduceAxis(axis, Math::max);
    }

    /**
     * Population variance along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray varAxis(NumJArray array, int axis) {
        int ax = array.normalizeAxis(axis);
        NumJArray diff = array.sub(meanAxis(array, ax).expandDims(ax));
        return diff.mul(diff).meanAxis(ax);
    }

    /**
     * Population standard deviation along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray stdAxis(NumJArray array, int axis) {
        return varAxis(array, axis).sqrt();
    }

    /**
     * Index of the largest value along an axis (first one wins ties).
     *
     * @param axis the axis to search; negative values count from the end
     * @return a new array (same shape minus {@code axis}) holding indices stored as floats
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public static NumJArray argMaxAxis(NumJArray array, int axis) {
        return array.argReduceAxis(axis, true);
    }

    /**
     * Index of the smallest value along an axis (first one wins ties).
     *
     * @param axis the axis to search; negative values count from the end
     * @return a new array (same shape minus {@code axis}) holding indices stored as floats
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public static NumJArray argMinAxis(NumJArray array, int axis) {
        return array.argReduceAxis(axis, false);
    }

    /**
     * Running total along an axis.
     *
     * @param axis the axis to accumulate along; negative values count from the end
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray cumSumAxis(NumJArray array, int axis) {
        int ax = array.normalizeAxis(axis);
        NumJArray out = array.copy();
        int len = out.shape[ax];
        int st = out.strides[ax];
        for (int base : out.laneOffsets(ax)) {
            for (int k = 1; k < len; k++) {
                out.data[base + k * st] += out.data[base + (k - 1) * st];
            }
        }
        return out;
    }

    /**
     * Running total over the flattened array. (Use {@link #cumSumAxis(int)} to keep the shape.)
     *
     * @return a new 1-D array of {@link #size()} partial sums
     */
    public static NumJArray cumSum(NumJArray array) {
        float[] d = array.toContiguousData();
        for (int i = 1; i < d.length; i++) d[i] += d[i - 1];
        return new NumJArray(d, new int[]{d.length});
    }

    /**
     * Running product over the flattened array.
     *
     * @return a new 1-D array of {@link #size()} partial products
     */
    public static NumJArray cumProd(NumJArray array) {
        float[] d = array.toContiguousData();
        for (int i = 1; i < d.length; i++) d[i] *= d[i - 1];
        return new NumJArray(d, new int[]{d.length});
    }

    /**
     * Running product along an axis.
     *
     * @param axis the axis to accumulate along; negative values count from the end
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray cumProdAxis(NumJArray array, int axis) {
        int ax = array.normalizeAxis(axis);
        NumJArray out = array.copy();
        int len = out.shape[ax];
        int st = out.strides[ax];
        for (int base : out.laneOffsets(ax)) {
            for (int k = 1; k < len; k++) {
                out.data[base + k * st] *= out.data[base + (k - 1) * st];
            }
        }
        return out;
    }

    /**
     * Median of all elements.
     *
     * @return the middle value (the mean of the two middle values for an even count); NaN if any element is NaN
     * @throws IllegalStateException if the array is empty
     */
    public static float median(NumJArray array) {
        return quantile(array, 0.5f);
    }

    /**
     * Median along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public static NumJArray median(NumJArray array, int axis) {
        return quantile(array, 0.5f, axis);
    }

    /**
     * Percentile of all elements.
     *
     * @param p the percentile from 0 to 100
     * @return the interpolated value; NaN if any element is NaN
     * @throws IllegalArgumentException if {@code p} is outside {@code [0, 100]}
     * @throws IllegalStateException    if the array is empty
     */
    public static float percentile(NumJArray array, float p) {
        if (!(p >= 0 && p <= 100)) {
            throw new IllegalArgumentException("percentile must be between 0 and 100, got " + p);
        }
        return quantile(array, p / 100.0f);
    }

    /**
     * Percentile along an axis.
     *
     * @param p    the percentile from 0 to 100
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if {@code p} is outside {@code [0, 100]}, or the axis is out of range or has length 0
     */
    public static NumJArray percentile(NumJArray array, float p, int axis) {
        if (!(p >= 0 && p <= 100)) {
            throw new IllegalArgumentException("percentile must be between 0 and 100, got " + p);
        }
        return quantile(array, p / 100.0f, axis);
    }

    /**
     * Quantile of all elements. {@code 0} is the minimum, {@code 0.5} the median and {@code 1} the maximum;
     * values in between are interpolated linearly.
     *
     * @param q the quantile from 0 to 1
     * @return the interpolated value; NaN if any element is NaN
     * @throws IllegalArgumentException if {@code q} is outside {@code [0, 1]}
     * @throws IllegalStateException    if the array is empty
     */
    public static float quantile(NumJArray array, float q) {
        NumJArray.checkQuantile(q);
        array.requireNonEmpty("quantile");
        float[] d = array.toContiguousData();
        Arrays.sort(d);
        return NumJArray.interpolateSorted(d, q);
    }

    /**
     * Quantile along an axis.
     *
     * @param q    the quantile from 0 to 1
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if {@code q} is outside {@code [0, 1]}, or the axis is out of range or has length 0
     */
    public static NumJArray quantile(NumJArray array, float q, int axis) {
        NumJArray.checkQuantile(q);
        int ax = array.normalizeAxis(axis);
        int len = array.shape[ax];
        if (len == 0) {
            throw new IllegalArgumentException("Cannot take a quantile over axis " + ax + " because it has length 0");
        }

        int st = array.strides[ax];
        int[] lanes = array.laneOffsets(ax);
        float[] out = new float[lanes.length];
        float[] lane = new float[len];

        for (int l = 0; l < lanes.length; l++) {
            for (int k = 0; k < len; k++) lane[k] = array.data[lanes[l] + k * st];
            Arrays.sort(lane);
            out[l] = NumJArray.interpolateSorted(lane, q);
        }
        return new NumJArray(out, NumJArray.removeAt(array.shape, ax));
    }

     public static void checkQuantile(float q) {
        if (!(q >= 0 && q <= 1)) {
            throw new IllegalArgumentException("quantile must be between 0 and 1, got " + q);
        }
    }

    public static float interpolateSorted(float[] sorted, float q) {
        int n = sorted.length;
        if (Float.isNaN(sorted[n - 1])) return Float.NaN;

        float pos = q * (n - 1);
        int lo = (int) Math.floor(pos);
        int hi = Math.min(lo + 1, n - 1);
        float frac = pos - lo;
        if (frac == 0 || sorted[lo] == sorted[hi]) return sorted[lo];
        return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
    }

    /**
     * Folds every element into one number with your own operator, starting from the first element.
     * Example: {@code reduce((x, y) -> x * y)} is the product.
     *
     * @param op the operator combining the running result with the next element
     * @return the folded value
     * @throws IllegalArgumentException if {@code op} is null
     * @throws IllegalStateException    if the array is empty
     */
    public static float reduce(NumJArray array, FloatBinaryOperator op) {
        NumJArray.requireOperator(op);
        array.requireNonEmpty("reduce");
        float[] d = array.readOnlyData();
        float acc = d[0];
        for (int i = 1; i < d.length; i++) acc = op.applyAsFloat(acc, d[i]);
        return acc;
    }

    /**
     * Folds every element into one number with your own operator, starting from {@code init}.
     *
     * @param op   the operator combining the running result with the next element
     * @param init the starting value (also the result for an empty array)
     * @return the folded value
     * @throws IllegalArgumentException if {@code op} is null
     */
    public static float reduce(NumJArray array, FloatBinaryOperator op, float init) {
        NumJArray.requireOperator(op);
        float acc = init;
        for (float v : array.readOnlyData()) acc = op.applyAsFloat(acc, v);
        return acc;
    }

    /**
     * Folds along one axis with your own operator, starting from the first element of each lane.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @param op   the operator combining the running result with the next element
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or has length 0, or {@code op} is null
     */
    public static NumJArray reduceAxis(NumJArray array, int axis, FloatBinaryOperator op) {
        NumJArray.requireOperator(op);
        return array.reduceLanes(axis, op, null);
    }

    /**
     * Folds along one axis with your own operator, starting from {@code init} in each lane.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @param op   the operator combining the running result with the next element
     * @param init the starting value for every lane
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or {@code op} is null
     */
    public static NumJArray reduceAxis(NumJArray a, int axis, FloatBinaryOperator op, float init) {
        NumJArray.requireOperator(op);
        return a.reduceLanes(axis, op, init);
    }

    /** Applies a single-axis reduction once per listed axis, highest axis first so indices stay valid. */
    public static NumJArray reduceAxes(NumJArray a, int[] axes, BiFunction<NumJArray, Integer, NumJArray> single) {
        if (axes == null) {
            throw new IllegalArgumentException("axes must not be null");
        }
        int[] sorted = new int[axes.length];
        for (int i = 0; i < axes.length; i++) sorted[i] = a.normalizeAxis(axes[i]);
        Arrays.sort(sorted);
        for (int i = 1; i < sorted.length; i++) {
            if (sorted[i] == sorted[i - 1]) {
                throw new IllegalArgumentException("Axis " + sorted[i] + " is listed more than once");
            }
        }
        if (sorted.length == 0) return a.copy();

        NumJArray result = a;
        for (int i = sorted.length - 1; i >= 0; i--) {
            result = single.apply(result, sorted[i]);
        }
        return result;
    }

    /**
     * Sums over several axes at once, e.g. {@code sum(0, 2)} on shape {@code [2, 3, 4]} gives shape {@code [3]}.
     * (Call {@link #sum()} with no arguments for a plain {@code float} total.)
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public static NumJArray sum(NumJArray a, int... axes) {
        return reduceAxes(a, axes, NumJArray::sumAxis);
    }

    /**
     * Multiplies over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public static NumJArray prod(NumJArray a, int... axes) {
        return reduceAxes(a, axes, NumJArray::prodAxis);
    }

    /**
     * Averages over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public static NumJArray mean(NumJArray a, int... axes) {
        NumJArray total = sum(a, axes);
        long count = 1;
        for (int ax : axes) count *= a.shape[a.normalizeAxis(ax)];
        return total.div(count);
    }

    /**
     * Minimum over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range, listed twice, or has length 0
     */
    public static NumJArray min(NumJArray a, int... axes) {
        return reduceAxes(a, axes, NumJArray::minAxis);
    }

    /**
     * Maximum over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range, listed twice, or has length 0
     */
    public static NumJArray max(NumJArray a, int... axes) {
        return reduceAxes(a, axes, NumJArray::maxAxis);
    }

    /**
     * Position of the largest element in the flattened (row-major) array.
     *
     * @return the flat index of the first maximum
     * @throws IllegalStateException if the array is empty
     */
    public static int argMaxFlat(NumJArray a) {
        return argFlat(a, true);
    }

    /**
     * Position of the smallest element in the flattened (row-major) array.
     *
     * @return the flat index of the first minimum
     * @throws IllegalStateException if the array is empty
     */
    public static int argMinFlat(NumJArray a) {
        return argFlat(a, false);
    }

    public static int argFlat(NumJArray array, boolean findMax) {
        array.requireNonEmpty(findMax ? "argMax" : "argMin");
        float[] d = array.readOnlyData();
        int best = 0;
        for (int i = 1; i < d.length; i++) {
            if (findMax ? d[i] > d[best] : d[i] < d[best]) best = i;
        }
        return best;
    }

    public static NumJArray argReduceAxis(NumJArray array, int axis, boolean findMax) {
        int ax = array.normalizeAxis(axis);
        int len = array.shape[ax];
        int st = array.strides[ax];

        if (len == 0) {
            throw new IllegalArgumentException("Cannot search axis " + ax + " because it has length 0");
        }

        int[] lanes = array.laneOffsets(ax);
        float[] out = new float[lanes.length];

        for (int l = 0; l < lanes.length; l++) {
            int base = lanes[l];
            int best = 0;
            for (int k = 1; k < len; k++) {
                float cur = array.data[base + k * st];
                float bestVal = array.data[base + best * st];
                if (findMax ? cur > bestVal : cur < bestVal) best = k;
            }
            out[l] = best;
        }
        return new NumJArray(out, NumJArray.removeAt(array.shape, ax));
    }

}
