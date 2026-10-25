package numj.math;

import java.util.Arrays;

import numj.core.Creation;
import numj.core.NumJArray;

public class sorts {

    /**
     * Sorts along the last axis (ascending). The original is not changed.
     *
     * @return a new, sorted array of the same shape
     * @throws IllegalArgumentException if the array is 0-D
     */
    public static NumJArray sort(NumJArray array) {
        return sort(array, -1);
    }

    /**
     * Sorts along an axis (ascending). The original is not changed.
     *
     * @param axis the axis to sort along; negative values count from the end
     * @return a new, sorted array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray sort(NumJArray array, int axis) {
        int ax = array.normalizeAxis(axis);
        NumJArray out = array.copy();
        int len = out.shape[ax];
        int st = out.strides[ax];
        float[] lane = new float[len];

        for (int base : out.laneOffsets(ax)) {
            for (int k = 0; k < len; k++) lane[k] = out.data[base + k * st];
            Arrays.sort(lane);
            for (int k = 0; k < len; k++) out.data[base + k * st] = lane[k];
        }
        return out;
    }

    /**
     * Sorts all elements as one flat list (ascending). The original is not changed.
     *
     * @return a new 1-D array with {@link #size()} sorted values
     */
    public static NumJArray sortFlat(NumJArray array) {
        float[] d = array.toContiguousData();
        Arrays.sort(d);
        return new NumJArray(d, new int[]{d.length});
    }

    /**
     * Finds the order that would sort the array along the last axis. Ties keep their original order.
     *
     * @return a new array of the same shape holding indices (stored as floats)
     * @throws IllegalArgumentException if the array is 0-D
     */
    public static NumJArray argsort(NumJArray array) {
        return argsort(array, -1);
    }

    /**
     * Finds the order that would sort the array along an axis. Ties keep their original order.
     * Use {@code take(axis, order)} or {@code flip} on the result for other orderings.
     *
     * @param axis the axis to sort along; negative values count from the end
     * @return a new array of the same shape holding indices along {@code axis} (stored as floats)
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray argsort(NumJArray array, int axis) {
        int ax = array.normalizeAxis(axis);
        NumJArray out = creation.zeros(array.shape);
        int len = array.shape[ax];
        int srcStride = array.strides[ax];
        int dstStride = out.strides[ax];

        int[] srcLanes = array.laneOffsets(ax);
        int[] dstLanes = out.laneOffsets(ax);
        final float[] vals = new float[len];
        Integer[] order = new Integer[len];

        for (int l = 0; l < srcLanes.length; l++) {
            for (int k = 0; k < len; k++) {
                vals[k] = array.data[srcLanes[l] + k * srcStride];
                order[k] = k;
            }
            Arrays.sort(order, (i, j) -> NumJArray.compareNaNLast(vals[i], vals[j]));
            for (int k = 0; k < len; k++) out.data[dstLanes[l] + k * dstStride] = order[k];
        }
        return out;
    }

    /**
     * Finds the order that would sort all elements as one flat list. Ties keep their original order.
     *
     * @return a new 1-D array of flat positions (stored as floats)
     */
    public static NumJArray argsortFlat(NumJArray array) {
        return array.flatten().argsort(0);
    }

    /**
     * Finds the distinct values, sorted ascending. All NaNs count as one value, and 0.0 equals -0.0.
     *
     * @return a new 1-D array of the unique values
     */
    public static NumJArray unique(NumJArray array) {
        return uniqueCounts(array)[0];
    }

    /**
     * Finds the distinct values together with how often each occurs.
     *
     * @return an array of two 1-D arrays: {@code [0]} the sorted unique values, {@code [1]} their counts
     *         (stored as floats)
     */
    public static NumJArray[] uniqueCounts(NumJArray array) {
        float[] d = array.toContiguousData();
        Arrays.sort(d);

        float[] values = new float[d.length];
        float[] counts = new float[d.length];
        int n = 0;

        for (float v : d) {
            if (n > 0 && (values[n - 1] == v || (Float.isNaN(values[n - 1]) && Float.isNaN(v)))) {
                counts[n - 1]++;
            } else {
                values[n] = v;
                counts[n] = 1;
                n++;
            }
        }
        return new NumJArray[]{
            new NumJArray(Arrays.copyOf(values, n), new int[]{n}),
            new NumJArray(Arrays.copyOf(counts, n), new int[]{n})
        };
    }

    /** Ordering for sorting: normal numeric order, NaN last, and 0.0 equal to -0.0. */
    public static int compareNaNLast(float x, float y) {
        if (x < y) return -1;
        if (x > y) return 1;
        boolean nx = Float.isNaN(x);
        boolean ny = Float.isNaN(y);
        return (nx == ny) ? 0 : (nx ? 1 : -1);
    }
}
