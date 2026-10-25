package numj.core;

import java.util.Arrays;
import numj.array.Creation;
import numj.core.NumJArray.CoordinateFunction;


public class manipulation {

    /**
     * Reverses the order of all axes. Returns a view; no data is copied.
     *
     * @return a transposed view sharing data with this array
     */
    public static  NumJArray transpose(NumJArray array) {
        int rank = array.shape.length;
        int[] newShape = new int[rank];
        int[] newStrides = new int[rank];

        for (int i = 0; i < rank; i++) {
            newShape[i] = array.shape[rank - 1 - i];
            newStrides[i] = array.strides[rank - 1 - i];
        }
        return new NumJArray(array.data, newShape, newStrides, array.offset);
    }

    /**
     * Reorders the axes. Returns a view.
     *
     * @param axes a permutation of {@code 0..ndim-1}; {@code axes[i]} is the old axis that becomes axis {@code i}
     * @return a permuted view sharing data with this array
     * @throws IllegalArgumentException if {@code axes} is not a valid permutation
     */
    public static  NumJArray transpose(NumJArray array, int... axes) {
        if (axes == null || axes.length != array.shape.length) {
            throw new IllegalArgumentException("axes must list every dimension exactly once");
        }
        boolean[] seen = new boolean[array.shape.length];
        int[] newShape = new int[array.shape.length];
        int[] newStrides = new int[array.shape.length];

        for (int i = 0; i < axes.length; i++) {
            int ax = array.normalizeAxis(axes[i]);
            if (seen[ax]) {
                throw new IllegalArgumentException("Axis " + ax + " appears more than once");
            }
            seen[ax] = true;
            newShape[i] = array.shape[ax];
            newStrides[i] = array.strides[ax];
        }
        return new NumJArray(array.data, newShape, newStrides, array.offset);
    }

    /**
     * Swaps two axes. Returns a view.
     *
     * @param axis1 the first axis; negative values count from the end
     * @param axis2 the second axis; negative values count from the end
     * @return a view with the two axes exchanged
     * @throws IllegalArgumentException if either axis is out of range
     */
    public static  NumJArray swapAxes(NumJArray array, int axis1, int axis2) {
        int a = array.normalizeAxis(axis1);
        int b = array.normalizeAxis(axis2);
        int[] newShape = array.shape.clone();
        int[] newStrides = array.strides.clone();

        newShape[a] = array.shape[b];
        newShape[b] = array.shape[a];
        newStrides[a] = array.strides[b];
        newStrides[b] = array.strides[a];
        return new NumJArray(array.data, newShape, newStrides, array.offset);
    }

    /**
     * Gives the array a new shape with the same elements in the same logical order.
     * Returns a view if the data is contiguous, otherwise a copy.
     *
     * @param newShape the new dimensions; at most one may be {@code -1} and is then inferred
     * @return an array with the requested shape
     * @throws IllegalArgumentException if the shape is invalid or holds a different number of elements
     */
    public static  NumJArray reshape(NumJArray array, int... newShape) {
        if (newShape == null) {
            throw new IllegalArgumentException("newShape must not be null");
        }

        int[] resolved = newShape.clone();
        int unknown = -1;
        long known = 1;

        for (int i = 0; i < resolved.length; i++) {
            if (resolved[i] == -1) {
                if (unknown != -1) {
                    throw new IllegalArgumentException("Only one dimension can be -1");
                }
                unknown = i;
            } else if (resolved[i] < 0) {
                throw new IllegalArgumentException("Invalid dimension " + resolved[i] + " in reshape");
            } else {
                known *= resolved[i];
            }
        }

        if (unknown != -1) {
            if (known == 0 || array.size % known != 0) {
                throw new IllegalArgumentException(
                    "Cannot reshape " + array.size + " elements into " + Arrays.toString(newShape));
            }
            resolved[unknown] = (int) (array.size / known);
        }

        if (Validator.sizeOf(resolved) != array.size) {
            throw new IllegalArgumentException(
                "New shape must have the same number of elements: " + array.size + " vs " + Arrays.toString(newShape));
        }

        if (array.isContiguous()) {
            return new NumJArray(array.data, resolved, NumJArray.createStrides(resolved), array.offset);
        }
        return new NumJArray(array.toContiguousData(), resolved);
    }

    /**
     * Flattens to 1-D. Returns a view when possible, otherwise a copy (like NumPys {@code ravel}).
     *
     * @return a 1-D array with {@link #size()} elements
     */
    public static  NumJArray ravel(NumJArray array) {
        return reshape(array, array.size);
    }

    /**
     * Flattens to 1-D, always returning an independent copy.
     *
     * @return a new 1-D array with {@link #size()} elements
     */
    public static  NumJArray flatten(NumJArray array) {
        return new NumJArray(array.toContiguousData(), new int[]{array.size});
    }

    /**
     * Makes an independent, contiguous copy (views become normal arrays).
     *
     * @return a new array with the same shape and values
     */
    public static  NumJArray copy(NumJArray array) {
        return new NumJArray(array.toContiguousData(), array.shape);
    }

    /**
     * Inserts a new dimension of size 1. Returns a view.
     *
     * @param axis the position of the new axis in the result, from {@code 0} to {@code ndim}
     *             (negative values count from the end of the result)
     * @return a view with one more dimension
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static  NumJArray expandDims(NumJArray array, int axis) {
        int ax = NumJArray.normalizeAxis(axis, array.shape.length + 1);
        int[] newShape = new int[array.shape.length + 1];
        int[] newStrides = new int[array.shape.length + 1];

        for (int i = 0, src = 0; i < newShape.length; i++) {
            if (i == ax) {
                newShape[i] = 1;
                newStrides[i] = (src < array.shape.length) ? array.shape[src] * array.strides[src] : 1;
            } else {
                newShape[i] = array.shape[src];
                newStrides[i] = array.strides[src];
                src++;
            }
        }
        return new NumJArray(array.data, newShape, newStrides, array.offset);
    }

    /**
     * Removes every dimension of size 1. Returns a view.
     *
     * @return a view without singleton dimensions
     */
    public static  NumJArray squeeze(NumJArray array) {
        int kept = 0;
        for (int dim : array.shape) if (dim != 1) kept++;

        int[] newShape = new int[kept];
        int[] newStrides = new int[kept];
        for (int i = 0, j = 0; i < array.shape.length; i++) {
            if (array.shape[i] != 1) {
                newShape[j] = array.shape[i];
                newStrides[j++] = array.strides[i];
            }
        }
        return new NumJArray(array.data, newShape, newStrides, array.offset);
    }

    /**
     * Picks one index along an axis, removing that axis (e.g. one row or one column). Returns a view.
     *
     * @param axis  the axis to index; negative values count from the end
     * @param index the position along that axis; negative values count from the end
     * @return a view with one fewer dimension
     * @throws IllegalArgumentException  if the axis is out of range
     * @throws IndexOutOfBoundsException if the index is outside the axis
     */
    public static  NumJArray select(NumJArray array, int axis, int index) {
        int ax = NumJArray.normalizeAxis(axis, array.shape.length);
        int idx = index < 0 ? index + array.shape[ax] : index;
        if (idx < 0 || idx >= array.shape[ax]) {
            throw new IndexOutOfBoundsException(
                "Index " + index + " is out of bounds for axis " + ax + " with size " + array.shape[ax]);
        }
        return new NumJArray(array.data, NumJArray.removeAt(array.shape, ax), NumJArray.removeAt(array.strides, ax), array.offset + idx * array.strides[ax]);
    }

    /**
     * Takes a rectangular, optionally stepped, sub-array. Returns a view.
     *
     * <p>Each argument is {@code {start, stop}} or {@code {start, stop, step}} for one dimension
     * (stop is exclusive; step defaults to 1). Pass {@code null} to keep a whole dimension, and leave out
     * trailing dimensions to keep them whole. Out-of-range values are clamped rather than rejected.
     * With a negative step, walk backwards from {@code start} down to (but not including) {@code stop};
     * use {@code stop = -1} to include index 0. Indices are not wrapped from the end.
     *
     * <p>Example: {@code a.slice(new int[]{0, 2}, null)} is the first two rows of a 2-D array.
     *
     * @param slices one {@code {start, stop[, step]}} (or {@code null}) per leading dimension
     * @return a view over the selected elements
     * @throws IllegalArgumentException if there are more slices than dimensions, a slice does not have
     *                                  2 or 3 entries, or a step is 0
     */
    public static  NumJArray slice(NumJArray array, int[]... slices) {
        int nd = array.shape.length;
        if (slices == null) slices = new int[0][];
        if (slices.length > nd) {
            throw new IllegalArgumentException("Got " + slices.length + " slices for a " + nd + "D array");
        }

        int[] newShape = new int[nd];
        int[] newStrides = new int[nd];
        int newOffset = array.offset;

        for (int i = 0; i < nd; i++) {
            int dim = array.shape[i];
            int[] s = i < slices.length ? slices[i] : null;
            long start, stop, step;

            if (s == null) {
                start = 0;
                stop = dim;
                step = 1;
            } else {
                if (s.length != 2 && s.length != 3) {
                    throw new IllegalArgumentException("Each slice must be {start, stop} or {start, stop, step}");
                }
                start = s[0];
                stop = s[1];
                step = (s.length == 3) ? s[2] : 1;
                if (step == 0) {
                    throw new IllegalArgumentException("Slice step cannot be 0");
                }
            }

            long count;
            if (step > 0) {
                start = NumJArray.clamp(start, 0, dim);
                stop = NumJArray.clamp(stop, 0, dim);
                count = stop > start ? (stop - start + step - 1) / step : 0;
            } else {
                start = NumJArray.clamp(start, -1, dim - 1);
                stop = NumJArray.clamp(stop, -1, dim - 1);
                count = start > stop ? (start - stop + (-step) - 1) / (-step) : 0;
            }

            newShape[i] = (int) count;
            newStrides[i] = (int) (array.strides[i] * step);
            if (count > 0) newOffset += (int) start * array.strides[i];
        }

        return new NumJArray(array.data, newShape, newStrides, newOffset);
    }

    /**
     * Joins two arrays end to end along an existing axis.
     *
     * @param b    the array to append; same number of dimensions, and same size on every axis except {@code axis}
     * @param axis the axis to join along; negative values count from the end
     * @return a new array whose size along {@code axis} is the sum of both
     * @throws IllegalArgumentException if {@code b} is null, the axis is out of range, or the shapes do not fit
     */
    public static  NumJArray concat(NumJArray a, NumJArray b, int axis) {
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }
        if (a.shape.length != b.shape.length) {
            throw new IllegalArgumentException("Arrays must have the same number of dimensions.");
        }
        int ax = a.normalizeAxis(axis);

        for (int i = 0; i < a.shape.length; i++) {
            if (i != ax && a.shape[i] != b.shape[i]) {
                throw new IllegalArgumentException(
                    "Shapes must match except on the concatenation axis: "
                        + Arrays.toString(a.shape) + " vs " + Arrays.toString(b.shape));
            }
        }

        int[] newShape = a.shape.clone();
        newShape[ax] = a.shape[ax] + b.shape[ax];
        NumJArray out = Creation.zeros(newShape);

        out.sliceAxis(ax, 0, a.shape[ax]).assign(a);
        out.sliceAxis(ax, a.shape[ax], newShape[ax]).assign(b);
        return out;
    }

    /**
     * Joins two arrays of identical shape along a new axis.
     *
     * @param b    the array to stack; must have exactly the same shape as this one
     * @param axis where the new axis goes, from {@code 0} to {@code ndim}; negative values count from the end
     * @return a new array with one more dimension, of size 2 along {@code axis}
     * @throws IllegalArgumentException if {@code b} is null, the shapes differ, or the axis is out of range
     */
    public static  NumJArray stack(NumJArray a, NumJArray b, int axis) {
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }
        if (!Arrays.equals(a.shape, b.shape)) {
            throw new IllegalArgumentException(
                "Arrays must have the same shape: " + Arrays.toString(a.shape) + " vs " + Arrays.toString(b.shape));
        }
        int ax = NumJArray.normalizeAxis(axis, a.shape.length + 1);

        int[] newShape = new int[a.shape.length + 1];
        for (int i = 0, src = 0; i < newShape.length; i++) {
            newShape[i] = (i == ax) ? 2 : a.shape[src++];
        }

        NumJArray out = Creation.zeros(newShape);
        out.select(ax, 0).assign(a);
        out.select(ax, 1).assign(b);
        return out;
    }

    /**
     * Splits into equal parts along an axis. The parts are views of this array;
     * call {@link #copy()} on a part if you need it independent.
     *
     * @param axis     the axis to split; negative values count from the end
     * @param sections the number of equal parts; must divide the axis length exactly
     * @return an array of {@code sections} views
     * @throws IllegalArgumentException if the axis is out of range, {@code sections <= 0},
     *                                  or the axis cannot be divided evenly
     */
    public static  NumJArray[] split(NumJArray array, int axis, int sections) {
        int ax = array.normalizeAxis(axis);
        if (sections <= 0) {
            throw new IllegalArgumentException("Sections must be greater than 0.");
        }
        if (array.shape[ax] % sections != 0) {
            throw new IllegalArgumentException(
                "Axis " + ax + " of size " + array.shape[ax] + " cannot be split into " + sections + " equal parts");
        }

        int part = array.shape[ax] / sections;
        NumJArray[] result = new NumJArray[sections];
        for (int s = 0; s < sections; s++) {
            result[s] = array.sliceAxis(ax, s * part, (s + 1) * part);
        }
        return result;
    }

    /**
     * Keeps the lower triangle (main diagonal and below) of every matrix in the last two axes, zeroing the rest.
     *
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public static  NumJArray tril(NumJArray a) {
        return tril(a, 0);
    }

    /**
     * Keeps everything on or below a chosen diagonal of every matrix in the last two axes, zeroing the rest.
     *
     * @param k which diagonal: 0 is the main one, positive values include more above it, negative values exclude more
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public static  NumJArray tril(NumJArray a, int k) {
        return triangle(a, k, true);
    }

    /**
     * Keeps the upper triangle (main diagonal and above) of every matrix in the last two axes, zeroing the rest.
     *
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public static  NumJArray triu(NumJArray a) {
        return triu(a, 0);
    }

    /**
     * Keeps everything on or above a chosen diagonal of every matrix in the last two axes, zeroing the rest.
     *
     * @param k which diagonal: 0 is the main one, positive values exclude more below, negative values include more
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public static  NumJArray triu(NumJArray a, int k) {
        return triangle(a, k, false);
    }

    public static  NumJArray triangle(NumJArray a, int k, boolean lower) {
        int[] shape = a.shape;
        int nd = shape.length;

        if (nd < 2) {
            throw new IllegalArgumentException("tril/triu need at least 2 dimensions, got " + nd + "D");
        }

        float[] d = a.toContiguousData();
        if (d.length == 0) return new NumJArray(d, shape);

        int rows = shape[nd - 2];
        int cols = shape[nd - 1];

        for (int i = 0; i < d.length; i++) {
            int r = (i / cols) % rows;
            int c = i % cols;
            boolean keep = lower ? (c - r <= k) : (c - r >= k);

            if (!keep) d[i] = 0;
        }

        return new NumJArray(d, shape);
    }

    /**
     * Finds the coordinates of every non-zero element, in row-major order.
     *
     * @return one {@code int[]} per dimension: {@code result[d][i]} is the index along axis {@code d}
     *         of the i-th non-zero element (so {@code get(result[0][i], result[1][i])} reads it on a 2-D array)
     * @throws IllegalArgumentException if the array is 0-D
     */
    public static  int[][] nonzero(NumJArray array) {
        if (array.shape.length == 0) {
            throw new IllegalArgumentException("nonzero() needs at least 1 dimension");
        }
        float[] d = array.readOnlyData();
        int count = 0;
        for (float v : d) if (v != 0) count++;

        int nd = array.shape.length;
        int[][] out = new int[nd][count];
        int[] idx = new int[nd];
        int w = 0;

        for (int i = 0; i < d.length; i++) {
            if (d[i] != 0) {
                for (int k = 0; k < nd; k++) out[k][w] = idx[k];
                w++;
            }
            for (int dim = nd - 1; dim >= 0; dim--) {
                if (++idx[dim] < array.shape[dim]) break;
                idx[dim] = 0;
            }
        }
        return out;
    }

    /**
     * Finds the coordinates of every non-zero element as rows of a table.
     *
     * @return a new array of shape {@code [count, ndim]}; row i holds the coordinates of the i-th
     *         non-zero element (stored as floats)
     * @throws IllegalArgumentException if the array is 0-D
     */
    public static  NumJArray argwhere(NumJArray array) {
        int[][] nz = array.nonzero();
        int nd = nz.length;
        int count = nz[0].length;

        float[] out = new float[count * nd];
        for (int i = 0; i < count; i++) {
            for (int k = 0; k < nd; k++) out[i * nd + k] = nz[k][i];
        }
        return new NumJArray(out, new int[]{count, nd});
    }

    /**
     * Picks elements by their position in the flattened (row-major) array.
     *
     * @param indices flat positions; negative values count from the end
     * @return a new 1-D array with one value per index, in the order given
     * @throws IllegalArgumentException  if {@code indices} is null
     * @throws IndexOutOfBoundsException if any index is outside the array
     */
    public static  NumJArray take(NumJArray array, int[] indices) {
        if (indices == null) {
            throw new IllegalArgumentException("indices must not be null");
        }
        float[] out = new float[indices.length];
        for (int i = 0; i < indices.length; i++) {
            out[i] = array.data[array.pointer(indices[i])];
        }
        return new NumJArray(out, new int[]{indices.length});
    }

    /**
     * Picks whole slices along an axis, e.g. {@code take(0, 2, 0)} gives rows 2 and 0, and
     * {@code take(-1, 3)} gives the fourth column. Indices may repeat.
     *
     * @param axis    the axis to pick from; negative values count from the end
     * @param indices positions along that axis; negative values count from the end
     * @return a new array with {@code indices.length} entries along {@code axis}
     * @throws IllegalArgumentException  if the axis is out of range or {@code indices} is null
     * @throws IndexOutOfBoundsException if any index is outside the axis
     */
    public static  NumJArray take(NumJArray array, int axis, int... indices) {
        if (indices == null) {
            throw new IllegalArgumentException("indices must not be null");
        }
        int ax = array.normalizeAxis(axis);
        int len = array.shape[ax];

        int[] picked = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i] < 0 ? indices[i] + len : indices[i];
            if (idx < 0 || idx >= len) {
                throw new IndexOutOfBoundsException(
                    "Index " + indices[i] + " is out of bounds for axis " + ax + " with size " + len);
            }
            picked[i] = idx;
        }

        int[] newShape = array.shape.clone();
        newShape[ax] = indices.length;
        NumJArray out = Creation.zeros(newShape);
        for (int i = 0; i < picked.length; i++) {
            out.select(ax, i).assign(array.select(ax, picked[i]));
        }
        return out;
    }

    /**
     * Writes one value into the elements at the given flat positions (in place).
     * Nothing is written if any index is invalid.
     *
     * @param flatIndices positions in the flattened row-major order; negative values count from the end
     * @param value       the value to store
     * @return this array
     * @throws IllegalArgumentException  if {@code flatIndices} is null
     * @throws IndexOutOfBoundsException if any index is outside the array
     */
    public static  NumJArray put(NumJArray array, int[] flatIndices, float value) {
        int[] slots = array.resolveFlat(flatIndices);
        for (int s : slots) array.data[s] = value;
        return array;
    }

    /**
     * Writes values into the elements at the given flat positions (in place). If there are fewer values
     * than indices, the values are repeated from the start, like NumPy's {@code put}.
     * Nothing is written if any index is invalid.
     *
     * @param flatIndices positions in the flattened row-major order; negative values count from the end
     * @param values      the values to store (read in row-major order)
     * @return this array
     * @throws IllegalArgumentException  if an argument is null, or {@code values} is empty but indices were given
     * @throws IndexOutOfBoundsException if any index is outside the array
     */
    public static  NumJArray put(NumJArray array, int[] flatIndices, NumJArray values) {
        if (values == null) {
            throw new IllegalArgumentException("values must not be null");
        }
        int[] slots = array.resolveFlat(flatIndices);
        float[] v = values.toContiguousData();
        if (v.length == 0) {
            if (slots.length == 0) return array;
            throw new IllegalArgumentException("values must not be empty");
        }
        for (int i = 0; i < slots.length; i++) array.data[slots[i]] = v[i % v.length];
        return array;
    }

    /**
     * Repeats every element of the flattened array.
     * Example: {@code [1, 2].repeat(2)} gives {@code [1, 1, 2, 2]}.
     *
     * @param repeats how many times to repeat each element; must be non-negative
     * @return a new 1-D array of length {@code size * repeats}
     * @throws IllegalArgumentException if {@code repeats} is negative
     */
    public static NumJArray repeat(NumJArray array, int repeats) {
        return array.flatten().repeat(repeats, 0);
    }

    /**
     * Repeats every slice along an axis, e.g. {@code repeat(2, 0)} floats each row in place.
     *
     * @param repeats how many times to repeat each slice; must be non-negative
     * @param axis    the axis to repeat along; negative values count from the end
     * @return a new array that is {@code repeats} times longer along {@code axis}
     * @throws IllegalArgumentException if {@code repeats} is negative or the axis is out of range
     */
    public static NumJArray repeat(NumJArray array, int repeats, int axis) {
        if (repeats < 0) {
            throw new IllegalArgumentException("repeats must be non-negative, got " + repeats);
        }
        int ax = array.normalizeAxis(axis);
        int len = array.shape[ax];
        if ((long) len * repeats > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Result is too large");
        }

        int[] idx = new int[len * repeats];
        for (int i = 0; i < len; i++) {
            for (int r = 0; r < repeats; r++) idx[i * repeats + r] = i;
        }
        return array.take(ax, idx);
    }

    /**
     * Repeats each slice along an axis a different number of times.
     * Example: {@code [1, 2, 3].repeat(new int[]{1, 0, 2}, 0)} gives {@code [1, 3, 3]}.
     *
     * @param repeats one count per slice along {@code axis}; each must be non-negative
     * @param axis    the axis to repeat along; negative values count from the end
     * @return a new array whose length along {@code axis} is the sum of {@code repeats}
     * @throws IllegalArgumentException if {@code repeats} has the wrong length, holds a negative value,
     *                                  or the axis is out of range
     */
    public static NumJArray repeat(NumJArray array, int[] repeats, int axis) {
        if (repeats == null) {
            throw new IllegalArgumentException("repeats must not be null");
        }
        int ax = array.normalizeAxis(axis);
        if (repeats.length != array.shape[ax]) {
            throw new IllegalArgumentException(
                "repeats needs " + array.shape[ax] + " entries for axis " + ax + ", got " + repeats.length);
        }

        long total = 0;
        for (int r : repeats) {
            if (r < 0) {
                throw new IllegalArgumentException("repeats must be non-negative, got " + r);
            }
            total += r;
        }
        if (total > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Result is too large");
        }

        int[] idx = new int[(int) total];
        int w = 0;
        for (int i = 0; i < repeats.length; i++) {
            for (int r = 0; r < repeats[i]; r++) idx[w++] = i;
        }
        return array.take(ax, idx);
    }

    /**
     * Builds a bigger array by laying copies of this one side by side (NumPy's {@code tile}).
     * If {@code reps} has more entries than the array has dimensions, size-1 dimensions are added in
     * front; if it has fewer, it is padded with 1s in front.
     * Example: a {@code [2, 3]} array tiled with {@code (2, 2)} has shape {@code [4, 6]}.
     *
     * @param reps how many copies along each dimension (counted from the last); each must be non-negative
     * @return a new array with {@code shape[i] * reps[i]} entries on each dimension
     * @throws IllegalArgumentException if {@code reps} is null or holds a negative value
     */
    public static NumJArray tile(NumJArray array, int... reps) {
        if (reps == null) {
            throw new IllegalArgumentException("reps must not be null");
        }
        for (int r : reps) {
            if (r < 0) {
                throw new IllegalArgumentException("reps must be non-negative, got " + r);
            }
        }
        if (reps.length == 0) return array.copy();

        int nd = Math.max(array.shape.length, reps.length);
        int leadS = nd - array.shape.length;
        int leadR = nd - reps.length;

        int[] baseShape = new int[nd];
        int[] baseStrides = new int[nd];
        int[] outShape = new int[nd];
        for (int i = 0; i < nd; i++) {
            baseShape[i] = (i < leadS) ? 1 : array.shape[i - leadS];
            baseStrides[i] = (i < leadS) ? 0 : array.strides[i - leadS];
            outShape[i] = baseShape[i] * ((i < leadR) ? 1 : reps[i - leadR]);
        }

        float[] out = new float[Validator.sizeOf(outShape)];
        if (out.length == 0) return new NumJArray(out, outShape);

        int[] idx = new int[nd];
        for (int i = 0; i < out.length; i++) {
            int pos = array.offset;
            for (int d = 0; d < nd; d++) {
                pos += (idx[d] % baseShape[d]) * baseStrides[d];
            }
            out[i] = array.data[pos];
            for (int d = nd - 1; d >= 0; d--) {
                if (++idx[d] < outShape[d]) break;
                idx[d] = 0;
            }
        }
        return new NumJArray(out, outShape);
    }

    /**
     * Pads an N-dimensional array with zeros.
     *
     * @param padding An Nx2 array where padding[d][0] is the
     *                padding before dimension d and padding[d][1]
     *                is the padding after dimension d.
     * @return A new array containing the original values surrounded by zeros.
     */
    public static NumJArray pad(NumJArray array, int[][] padding) {

        if (padding == null || padding.length != array.shape.length) {
            throw new IllegalArgumentException("Padding array length must match array dimensions");
        }

        int nd = array.shape.length;
        int[] newShape = new int[nd];

        for (int d = 0; d < nd; d++) {
            if (padding[d] == null || padding[d].length != 2) {
                throw new IllegalArgumentException("Each padding dimension must contain exactly two values");
            }

            int before = padding[d][0];
            int after = padding[d][1];

            if (before < 0 || after < 0) {
                throw new IllegalArgumentException("Padding values cannot be negative");
            }

            newShape[d] = array.shape[d] + before + after;
        }

        NumJArray result = Creation.zeros(newShape);
        int totalElements = Validator.sizeOf(array.shape);
        int[] srcIdx = new int[nd];
        int[] destIdx = new int[nd];

        for (int i = 0; i < totalElements; i++) {

            int srcFlat = array.offset;

            for (int d = 0; d < nd; d++) {
                srcFlat += srcIdx[d] * array.strides[d];
                destIdx[d] = srcIdx[d] + padding[d][0];
            }

            int destFlat = result.getFlatIndex(destIdx);

            result.data[destFlat] = array.data[srcFlat];

            for (int d = nd - 1; d >= 0; d--) {

                srcIdx[d]++;

                if (srcIdx[d] < array.shape[d]) {
                    break;
                }

                srcIdx[d] = 0;
            }
        }

        return result;
    }

    /**
     * Reverses the order of elements along evry axis. Returns a view (negative strides).
     *
     * @return a flipped view sharing data with this array
     */
    public static NumJArray flip(NumJArray array) {
        int[] all = new int[array.shape.length];
        for (int i = 0; i < all.length; i++) all[i] = i;
        return flip(array, all);
    }

    /**
     * Reverses the order of elements along the given axes. Returns a view (negative strides).
     *
     * @param axes the axes to reverse; negative values count from the end
     * @return a flipped view sharing data with this array
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public static NumJArray flip(NumJArray array, int... axes) {
        if (axes == null) {
            throw new IllegalArgumentException("axes must not be null");
        }
        boolean[] seen = new boolean[array.shape.length];
        int[] newStrides = array.strides.clone();
        int newOffset = array.offset;

        for (int axis : axes) {
            int ax = array.normalizeAxis(axis);
            if (seen[ax]) {
                throw new IllegalArgumentException("Axis " + ax + " is listed more than once");
            }
            seen[ax] = true;
            if (array.shape[ax] > 0) {
                newOffset += (array.shape[ax] - 1) * array.strides[ax];
                newStrides[ax] = -array.strides[ax];
            }
        }
        return new NumJArray(array.data, array.shape, newStrides, newOffset);
    }

    /**
     * Shifts the elements of the flattened array, wrapping around, and keeps the shape.
     * Example: {@code [1, 2, 3, 4, 5].roll(2)} gives {@code [4, 5, 1, 2, 3]}.
     *
     * @param shift how far to move elements forward; negative values move them backward
     * @return a new array of the same shape
     */
    public static NumJArray roll(NumJArray array, int shift) {
        if (array.size == 0) return array.copy();

        int s = Math.floorMod(shift, array.size);
        int[] idx = new int[array.size];
        for (int i = 0; i < array.size; i++) idx[i] = Math.floorMod(i - s, array.size);
        return array.take(idx).reshape(array.shape);
    }

    /**
     * Shifts the slices along one axis, wrapping around (e.g. {@code roll(1, 0)} moves each row down one).
     *
     * @param shift how far to move slices forward; negative values move them backward
     * @param axis  the axis to shift along; negative values count from the end
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray roll(NumJArray array, int shift, int axis) {
        int ax = array.normalizeAxis(axis);
        int len = array.shape[ax];
        if (len == 0) return array.copy();

        int s = Math.floorMod(shift, len);
        int[] idx = new int[len];
        for (int i = 0; i < len; i++) idx[i] = Math.floorMod(i - s, len);
        return array.take(ax, idx);
    }

    public static NumJArray diff(NumJArray array) {
        return diff(array, 1, -1);
    }

    /**
     * Differences between neighbouring elements along an axis, applied {@code n} times.
     * Each pass makes the axis one shorter (never below 0).
     *
     * @param n    how many times to take differences; must be non-negative ({@code 0} returns a copy)
     * @param axis the axis to difference along; negative values count from the end
     * @return a new array that is {@code n} shorter along {@code axis}
     * @throws IllegalArgumentException if {@code n} is negative or the axis is out of range
     */
    public static NumJArray diff(NumJArray array, int n, int axis) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative, got " + n);
        }
        int ax = array.normalizeAxis(axis);

        NumJArray cur = array;
        for (int step = 0; step < n; step++) {
            int len = cur.shape[ax];
            if (len == 0) break;
            cur = cur.sliceAxis(ax, 1, len).sub(cur.sliceAxis(ax, 0, len - 1));
        }
        return (cur == array) ? array.copy() : cur;
    }

    /** Turns flat indices into slots in {@code data}, failing before anything is written. */
    public static  int[] resolveFlat(NumJArray array, int[] flatIndices) {
        if (flatIndices == null) {
            throw new IllegalArgumentException("flatIndices must not be null");
        }
        int[] slots = new int[flatIndices.length];
        for (int i = 0; i < slots.length; i++) slots[i] = array.pointer(flatIndices[i]);
        return slots;
    }

    /**
     * Copies another arrays values into this one. Because slices are views, this is how you write
     * into part of an array: {@code a.slice(rows, cols).assign(values)}. Overlapping source and
     * destination are handled safely.
     *
     * @param src an array broadcastable to this arrays shape (a scalar-like {@code [1]} fills everything)
     * @return this array
     * @throws IllegalArgumentException if {@code src} is null or cannot be broadcast to this shape
     */
    public static NumJArray assign(NumJArray a, NumJArray src) {
        return a.elementwisei(src, (x, y) -> y);
    }

    /**
     * Sets every element to one value.
     *
     * @param value the value to store
     * @return this array
     */
    public static NumJArray fill(NumJArray a, float value) {
        for (int p : a.positions()) a.data[p] = value;
        return a;
    }

    /**
     * Sets every element from a function of its coordinates, e.g.
     * {@code zeros(3, 3).fill(c -> c[0] == c[1] ? 1 : 0)} builds an identity matrix.
     *
     * @param f computes the value for the element at the given coordinates
     * @return this array
     * @throws IllegalArgumentException if {@code f} is null
     */
    public static NumJArray fill(NumJArray a, CoordinateFunction f) {
        if (f == null) {
            throw new IllegalArgumentException("f must not be null");
        }
        if (a.size == 0) return a;

        int nd = a.shape.length;
        int[] idx = new int[nd];
        int[] pos = a.positions();

        for (int i = 0; i < pos.length; i++) {
            a.data[pos[i]] = f.apply(idx);
            for (int d = nd - 1; d >= 0; d--) {
                if (++idx[d] < a.shape[d]) break;
                idx[d] = 0;
            }
        }
        return a;
    }

    /** View of {@code [start, stop)} along one axis, keeping all other axes whole. */
    public static NumJArray sliceAxis(NumJArray array, int ax, int start, int stop) {
        int[] newShape = array.shape.clone();
        newShape[ax] = stop - start;
        int newOffset = (stop > start) ? array.offset + start * array.strides[ax] : array.offset;
        return new NumJArray(array.data, newShape, array.strides, newOffset);
    }
}
