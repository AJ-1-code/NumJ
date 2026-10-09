package numj;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A small, NumPy-style N-dimensional array of {@code double}s.
 * 
 *   {@code data}    - the flat backing array
 *   {@code shape}   - the size of each dimension
 *   {@code strides} - how many slots in {@code data} to jump to move one step along each dimension
 *   {@code offset}  - the slot in {@code data} where element {@code [0, 0, ...]} lives
 * 
 *   {@link #transpose()}, {@link #slice(int[][])},
 *   {@link #select(int, int)}, {@link #expandDims(int)} and {@link #reshape(int...)}
 *
 *
 * 
 */
public class NumJArray {

    /** Flat backing storage. Shared between this array and any views made from it. */
    public final double[] data;

    /** Size of each dimension. Do not modify; use {@link #shape()} for a safe copy. */
    public final int[] shape;

    /** Step (in {@code data} slots) to move one index along each dimension. */
    public final int[] strides;

    /** Index in {@code data} of the element at coordinates {@code [0, 0, ...]}. */
    public final int offset;

    private final int size;

    /**
     * Creates a contiguous (row-major) array that wraps {@code data} without copying it.
     *
     * @param data  the values in row-major order; its length must equal the product of {@code shape}
     * @param shape the size of each dimension (an empty shape creates a 0-D scalar array)
     * @throws IllegalArgumentException if {@code data} or {@code shape} is null, a dimension is
     *                                  negative, or {@code data.length} does not match the shape
     */
    public NumJArray(double[] data, int[] shape) {
        this(requireLength(data, shape), shape, createStrides(shape), 0);
    }

    /**
     * Creates an array over {@code data} with custom strides and no offset.
     *
     * @param data    the backing array
     * @param shape   the size of each dimension
     * @param strides the stride of each dimension; must have the same length as {@code shape}
     * @throws IllegalArgumentException if the arguments are inconsistent or would read outside {@code data}
     */
    public NumJArray(double[] data, int[] shape, int[] strides) {
        this(data, shape, strides, 0);
    }

    /**
     * Creates an array over {@code data} with custom strides and offset. This is the constructor
     * views are built with.
     *
     * @param data    the backing array
     * @param shape   the size of each dimension
     * @param strides the stride of each dimension; must have the same length as {@code shape}
     * @param offset  index in {@code data} of the first element
     * @throws IllegalArgumentException if the arguments are inconsistent or any element
     *                                  would lie outside {@code data}
     */
    public NumJArray(double[] data, int[] shape, int[] strides, int offset) {
        if (data == null) {
            throw new IllegalArgumentException("data must not be null");
        }
        checkShape(shape);
        if (strides == null || strides.length != shape.length) {
            throw new IllegalArgumentException("strides must have the same length as shape");
        }

        this.data = data;
        this.shape = shape.clone();
        this.strides = strides.clone();
        this.offset = offset;
        this.size = sizeOf(shape);

        if (size > 0) {
            long lowest = offset;
            long highest = offset;
            for (int i = 0; i < shape.length; i++) {
                long extent = (long) (shape[i] - 1) * strides[i];
                if (extent > 0) highest += extent; else lowest += extent;
            }
            if (lowest < 0 || highest >= data.length) {
                throw new IllegalArgumentException(
                    "Shape, strides and offset reach outside the data array (data length " + data.length + ")");
            }
        }
    }

    /**
     * Creates an array of zeros.
     *
     * @param shape the size of each dimension
     * @return a new contiguous array filled with {@code 0.0}
     * @throws IllegalArgumentException if the shape is invalid
     */
    public static NumJArray zeros(int... shape) {
        return new NumJArray(new double[sizeOf(checkShape(shape))], shape);
    }

    /**
     * Creates an array of ones.
     *
     * @param shape the size of each dimension
     * @return a new contiguous array filled with {@code 1.0}
     * @throws IllegalArgumentException if the shape is invalid
     */
    public static NumJArray ones(int... shape) {
        return full(1.0, shape);
    }

    /**
     * Creates an array filled with one value.
     *
     * @param value the value for every element
     * @param shape the size of each dimension
     * @return a new contiguous array filled with {@code value}
     * @throws IllegalArgumentException if the shape is invalid
     */
    public static NumJArray full(double value, int... shape) {
        double[] d = new double[sizeOf(checkShape(shape))];
        Arrays.fill(d, value);
        return new NumJArray(d, shape);
    }

    /**
     * Creates a 1-D array from the given values (the values are copied).
     *
     * @param values the elements
     * @return a new 1-D array
     * @throws IllegalArgumentException if {@code values} is null
     */
    public static NumJArray of(double... values) {
        if (values == null) {
            throw new IllegalArgumentException("values must not be null");
        }
        return new NumJArray(values.clone(), new int[]{values.length});
    }

    /**
     * Creates a 2-D array from a rectangular {@code double[][]} (the values are copied).
     *
     * @param rows the rows; every row must have the same length
     * @return a new 2-D array of shape {@code [rows.length, rows[0].length]}
     * @throws IllegalArgumentException if {@code rows} is null, or the rows are ragged
     */
    public static NumJArray of(double[][] rows) {
        if (rows == null) {
            throw new IllegalArgumentException("rows must not be null");
        }
        int cols = rows.length == 0 ? 0 : rows[0].length;
        double[] d = new double[rows.length * cols];
        for (int i = 0; i < rows.length; i++) {
            if (rows[i] == null || rows[i].length != cols) {
                throw new IllegalArgumentException("All rows must have the same length");
            }
            System.arraycopy(rows[i], 0, d, i * cols, cols);
        }
        return new NumJArray(d, new int[]{rows.length, cols});
    }

    /**
     * Creates a 1-D array counting up from 0 to {@code stop} in steps of 1, like Pythons {@code range(stop)}.
     * For example {@code arange(5)} gives {@code [0, 1, 2, 3, 4]}.
     *
     * @param stop the end of the range (exclusive)
     * @return a new 1-D array (empty if {@code stop <= 0})
     */
    public static NumJArray arange(double stop) {
        return arange(0, stop, 1);
    }

    /**
     * Creates a 1-D array counting up from {@code start} to {@code stop} in steps of 1.
     * For example {@code arange(1, 6)} gives {@code [1, 2, 3, 4, 5]}.
     *
     * @param start the first value
     * @param stop  the end of the range (exclusive)
     * @return a new 1-D array (empty if {@code stop <= start})
     */
    public static NumJArray arange(double start, double stop) {
        return arange(start, stop, 1);
    }

    /**
     * Creates a 1-D array of evenly spaced values in {@code [start, stop)}.
     * For example {@code arange(1, 6, 2)} gives {@code [1, 3, 5]}; a negative step counts down.
     *
     * @param start the first value
     * @param stop  the end of the range (exclusive)
     * @param step  the spacing between values; must not be 0
     * @return a new 1-D array (empty if the range contains no values)
     * @throws IllegalArgumentException if {@code step} is 0 or not finite
     */
    public static NumJArray arange(double start, double stop, double step) {
        if (step == 0 || Double.isNaN(step) || Double.isInfinite(step)) {
            throw new IllegalArgumentException("step must be a non-zero finite number");
        }
        int n = (int) Math.max(0, Math.ceil((stop - start) / step));
        double[] d = new double[n];
        for (int i = 0; i < n; i++) {
            d[i] = start + i * step;
        }
        return new NumJArray(d, new int[]{n});
    }

    /**
     * Creates a 1-D array of {@code num} evenly spaced values from {@code start} to {@code stop},
     * including {@code stop} (unlike {@link #arange(double, double, double)}).
     * For example {@code linspace(1, 4, 4)} gives {@code [1, 2, 3, 4]}.
     *
     * @param start the first value
     * @param stop  the last value (inclusive)
     * @param num   how many values to generate; must be non-negative ({@code 1} gives just {@code [start]})
     * @return a new 1-D array of length {@code num}
     * @throws IllegalArgumentException if {@code num} is negative
     */
    public static NumJArray linspace(double start, double stop, int num) {
        if (num < 0) {
            throw new IllegalArgumentException("num must be non-negative, got " + num);
        }
        double[] d = new double[num];
        if (num == 0) {
            return new NumJArray(d, new int[]{0});
        }
        if (num == 1) {
            d[0] = start;
        } else {
            double step = (stop - start) / (num - 1);
            for (int i = 0; i < num; i++) {
                d[i] = start + i * step;
            }
            d[num - 1] = stop;
        }
        return new NumJArray(d, new int[]{num});
    }

    /**
     * Creates an {@code n x n} identity matrix.
     *
     * @param n the number of rows and columns; must be non-negative
     * @return a new contiguous 2-D array with ones on the diagonal
     * @throws IllegalArgumentException if {@code n} is negative
     */
    public static NumJArray identity(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative, got " + n);
        }
        double[] d = new double[n * n];
        for (int i = 0; i < n; i++) {
            d[i * n + i] = 1.0;
        }
        return new NumJArray(d, new int[]{n, n});
    }

    /**
     * Creates an array of normally distributed random numbers.
     *
     * @param mean   the mean of the distribution
     * @param stdDev the standard deviation; must be non-negative
     * @param shape  the size of each dimension
     * @return a new contiguous array of random samples
     * @throws IllegalArgumentException if {@code stdDev} is negative or the shape is invalid
     */
    public static NumJArray normal(double mean, double stdDev, int... shape) {
        if (stdDev < 0) {
            throw new IllegalArgumentException("stdDev must be non-negative, got " + stdDev);
        }
        double[] d = new double[sizeOf(checkShape(shape))];
        Random rand = new Random();
        for (int i = 0; i < d.length; i++) {
            d[i] = mean + rand.nextGaussian() * stdDev;
        }
        return new NumJArray(d, shape);
    }

    /**
     * Creates an array of uniformly distributed random numbers in {@code [low, high)}.
     *
     * @param low   the lower bound (inclusive)
     * @param high  the upper bound (exclusive); must be {@code >= low}
     * @param shape the size of each dimension
     * @return a new contiguous array of random samples
     * @throws IllegalArgumentException if {@code high < low} or the shape is invalid
     */
    public static NumJArray uniform(double low, double high, int... shape) {
        if (high < low) {
            throw new IllegalArgumentException("high must be >= low");
        }
        double[] d = new double[sizeOf(checkShape(shape))];
        Random rand = new Random();
        for (int i = 0; i < d.length; i++) {
            d[i] = low + rand.nextDouble() * (high - low);
        }
        return new NumJArray(d, shape);
    }

    /**
     * Returns the shape.
     *
     * @return a copy of the shape array
     */
    public int[] shape() {
        return shape.clone();
    }

    /**
     * Returns the strides.
     *
     * @return a copy of the strides array
     */
    public int[] strides() {
        return strides.clone();
    }

    /**
     * Returns the number of dimensions.
     *
     * @return the rank of the array ({@code 0} for a scalar array)
     */
    public int ndim() {
        return shape.length;
    }

    /**
     * Returns the total number of elements.
     *
     * @return the product of all dimensions
     */
    public int size() {
        return size;
    }

    /**
     * Returns the only element of a single-element array.
     *
     * @return the value of the one element
     * @throws IllegalStateException if the array does not contain exactly one element
     */
    public double item() {
        if (size != 1) {
            throw new IllegalStateException("item() requires exactly 1 element, array has " + size);
        }
        return data[offset];
    }

    /**
     * Reads one element.
     *
     * @param indices one index per dimension; negative indices count from the end
     * @return the value at those coordinates
     * @throws IllegalArgumentException  if the number of indices differs from {@link #ndim()}
     * @throws IndexOutOfBoundsException if any index is outside its dimension
     */
    public double get(int... indices) {
        return data[getFlatIndex(indices)];
    }

    /**
     * Writes one element. If this array is a view, the original array changes too.
     *
     * @param value   the value to store
     * @param indices one index per dimension; negative indices count from the end
     * @throws IllegalArgumentException  if the number of indices differs from {@link #ndim()}
     * @throws IndexOutOfBoundsException if any index is outside its dimension
     */
    public void set(double value, int... indices) {
        data[getFlatIndex(indices)] = value;
    }

    /**
     * Applies a function to every element.
     *
     * @param f the function to apply
     * @return a new array of the same shape
     * @throws IllegalArgumentException if {@code f} is null
     */
    public NumJArray map(DoubleUnaryOperator f) {
        if (f == null) {
            throw new IllegalArgumentException("f must not be null");
        }
        double[] out = toContiguousData();
        for (int i = 0; i < out.length; i++) {
            out[i] = f.applyAsDouble(out[i]);
        }
        return new NumJArray(out, shape);
    }

    /**
     * Element-wise absolute value.
     *
     * @return a new array of {@code |x|}
     */
    public NumJArray abs() {
        return map(Math::abs);
    }

    /**
     * Element-wise negation.
     *
     * @return a new array of {@code -x}
     */
    public NumJArray neg() {
        return map(x -> -x);
    }

    /**
     * Element-wise square root. Negative inputs give NaN.
     *
     * @return a new array of {@code sqrt(x)}
     */
    public NumJArray sqrt() {
        return map(Math::sqrt);
    }

    /**
     * Element-wise power.
     *
     * @param power the exponent
     * @return a new array of {@code x^power}
     */
    public NumJArray pow(double power) {
        return map(x -> Math.pow(x, power));
    }

    /**
     * Element-wise {@code e^x}.
     *
     * @return a new array of {@code exp(x)}
     */
    public NumJArray exp() {
        return map(Math::exp);
    }

    /**
     * Element-wise natural logarithm. Zero gives -Infinity, negatives give NaN.
     *
     * @return a new array of {@code ln(x)}
     */
    public NumJArray log() {
        return map(Math::log);
    }

    /**
     * Element-wise floor.
     *
     * @return a new array with each value rounded down
     */
    public NumJArray floor() {
        return map(Math::floor);
    }

    /**
     * Element-wise ceiling.
     *
     * @return a new array with each value rounded up
     */
    public NumJArray ceil() {
        return map(Math::ceil);
    }

    /**
     * Element-wise rounding to the nearest integer, ties to even (same as NumPys {@code round}).
     *
     * @return a new array of rounded values
     */
    public NumJArray round() {
        return map(Math::rint);
    }

    /**
     * Element-wise sign.
     *
     * @return a new array containing -1, 0 or 1 (NaN stays NaN)
     */
    public NumJArray sign() {
        return map(Math::signum);
    }

    /**
     * Limits every element to the range {@code [min, max]}.
     *
     * @param min the lower bound
     * @param max the upper bound; must be {@code >= min}
     * @return a new array with values clamped into the range
     * @throws IllegalArgumentException if {@code min > max}
     */
    public NumJArray clip(double min, double max) {
        if (min > max) {
            throw new IllegalArgumentException("min must be <= max");
        }
        return map(x -> Math.max(min, Math.min(max, x)));
    }

    /**
     * Adds a scalar to every element.
     *
     * @param b the scalar
     * @return a new array of {@code x + b}
     */
    public NumJArray add(double b) {
        return map(x -> x + b);
    }

    /**
     * Subtracts a scalar from every element.
     *
     * @param b the scalar
     * @return a new array of {@code x - b}
     */
    public NumJArray sub(double b) {
        return map(x -> x - b);
    }

    /**
     * Multiplies every element by a scalar.
     *
     * @param b the scalar
     * @return a new array of {@code x * b}
     */
    public NumJArray mul(double b) {
        return map(x -> x * b);
    }

    /**
     * Divides every element by a scalar.
     *
     * @param b the scalar (dividing by 0 gives Infinity/NaN, as with plain doubles)
     * @return a new array of {@code x / b}
     */
    public NumJArray div(double b) {
        return map(x -> x / b);
    }

    /**
     * Element-wise modulo. The result takes the sign of {@code b}, like NumPy and Python
     * (Javas {@code %} would take the sign of {@code x}).
     *
     * @param b the divisor
     * @return a new array of {@code x mod b}
     */
    public NumJArray mod(double b) {
        return map(x -> {
            double r = x % b;
            return (r != 0 && (r < 0) != (b < 0)) ? r + b : r;
        });
    }

    /**
     * Element-wise addition with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code this + b}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray add(NumJArray b) {
        return broadcastOp(b, (x, y) -> x + y);
    }

    /**
     * Element-wise subtraction with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code this - b}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray sub(NumJArray b) {
        return broadcastOp(b, (x, y) -> x - y);
    }

    /**
     * Element-wise multiplication with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code this * b}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray mul(NumJArray b) {
        return broadcastOp(b, (x, y) -> x * y);
    }

    /**
     * Element-wise division with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code this / b}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray div(NumJArray b) {
        return broadcastOp(b, (x, y) -> x / y);
    }

    /**
     * Sum of all elements.
     *
     * @return the sum ({@code 0.0} for an empty array)
     */
    public double sum() {
        double total = 0;
        for (double v : readOnlyData()) total += v;
        return total;
    }

    /**
     * Product of all elements.
     *
     * @return the product ({@code 1.0} for an empty array)
     */
    public double prod() {
        double total = 1;
        for (double v : readOnlyData()) total *= v;
        return total;
    }

    /**
     * Arithmetic mean of all elements.
     *
     * @return the mean ({@code NaN} for an empty array)
     */
    public double mean() {
        return sum() / size;
    }

    /**
     * Smallest element.
     *
     * @return the minimum value
     * @throws IllegalStateException if the array is empty
     */
    public double min() {
        requireNonEmpty("min");
        double[] d = readOnlyData();
        double best = d[0];
        for (int i = 1; i < d.length; i++) best = Math.min(best, d[i]);
        return best;
    }

    /**
     * Largest element.
     *
     * @return the maximum value
     * @throws IllegalStateException if the array is empty
     */
    public double max() {
        requireNonEmpty("max");
        double[] d = readOnlyData();
        double best = d[0];
        for (int i = 1; i < d.length; i++) best = Math.max(best, d[i]);
        return best;
    }

    /**
     * Population variance of all elements (divides by N, like NumPys default).
     *
     * @return the variance ({@code NaN} for an empty array)
     */
    public double var() {
        double mean = mean();
        double acc = 0;
        for (double v : readOnlyData()) acc += (v - mean) * (v - mean);
        return acc / size;
    }

    /**
     * Population standard deviation of all elements.
     *
     * @return the square root of {@link #var()}
     */
    public double std() {
        return Math.sqrt(var());
    }

    /**
     * Euclidean (L2) norm of all elements, i.e. {@code sqrt(sum(x^2))}.
     *
     * @return the norm
     */
    public double norm() {
        double acc = 0;
        for (double v : readOnlyData()) acc += v * v;
        return Math.sqrt(acc);
    }

    /**
     * Sum of the main diagonal of a 2-D array. For N-D arrays use {@link #trace(int, int)}.
     *
     * @return the trace
     * @throws IllegalArgumentException if the array is not 2-D
     */
    public double trace() {
        if (shape.length != 2) {
            throw new IllegalArgumentException(
                "trace() needs a 2D array, got " + shape.length + "D; use trace(axis1, axis2) for N-D arrays");
        }
        return diagonal(0, 1).sum();
    }

    /**
     * Sum along the diagonal formed by two axes, for arrays of any rank
     * (e.g. {@code trace(1, 2)} on shape {@code [batch, n, n]} gives one trace per batch entry).
     *
     * @param axis1 the first axis of the diagonal; negative values count from the end
     * @param axis2 the second axis of the diagonal; must differ from {@code axis1}
     * @return a new array with both axes removed
     * @throws IllegalArgumentException if an axis is out of range or both axes are the same
     */
    public NumJArray trace(int axis1, int axis2) {
        return diagonal(axis1, axis2).sumAxis(-1);
    }

    /**
     * Takes the diagonal formed by two axes. Returns a view.
     *
     * @param axis1 the first axis of the diagonal; negative values count from the end
     * @param axis2 the second axis of the diagonal; must differ from {@code axis1}
     * @return a view where both axes are replaced by one new last axis of length
     *         {@code min(shape[axis1], shape[axis2])}
     * @throws IllegalArgumentException if an axis is out of range or both axes are the same
     */
    public NumJArray diagonal(int axis1, int axis2) {
        int a1 = normalizeAxis(axis1);
        int a2 = normalizeAxis(axis2);
        if (a1 == a2) {
            throw new IllegalArgumentException("diagonal axes must be different, got " + axis1 + " and " + axis2);
        }

        int nd = shape.length;
        int[] newShape = new int[nd - 1];
        int[] newStrides = new int[nd - 1];
        int j = 0;
        for (int i = 0; i < nd; i++) {
            if (i != a1 && i != a2) {
                newShape[j] = shape[i];
                newStrides[j++] = strides[i];
            }
        }
        newShape[j] = Math.min(shape[a1], shape[a2]);
        newStrides[j] = strides[a1] + strides[a2];
        return new NumJArray(data, newShape, newStrides, offset);
    }

    /**
     * Inner product of two arrays of the same shape: {@code sum(this * other)}.
     * (For NumPy-style {@code dot} on vectors, matrices and N-D arrays use {@link #dot(NumJArray)}.)
     *
     * @param other an array with exactly the same shape
     * @return the scalar inner product
     * @throws IllegalArgumentException if {@code other} is null or the shapes differ
     */
    public double inner(NumJArray other) {
        if (other == null) {
            throw new IllegalArgumentException("other must not be null");
        }
        if (!Arrays.equals(shape, other.shape)) {
            throw new IllegalArgumentException(
                "Shapes must match for inner: " + Arrays.toString(shape) + " vs " + Arrays.toString(other.shape));
        }
        double[] a = readOnlyData();
        double[] b = other.readOnlyData();
        double acc = 0;
        for (int i = 0; i < a.length; i++) acc += a[i] * b[i];
        return acc;
    }

    /**
     * Finds the position of the largest element.
     *
     * @return the coordinates (one per dimension) of the first maximum
     * @throws IllegalStateException if the array is empty
     */
    public int[] argMax() {
        return unravel(argFlat(true));
    }

    /**
     * Finds the position of the smallest element.
     *
     * @return the coordinates (one per dimension) of the first minimum
     * @throws IllegalStateException if the array is empty
     */
    public int[] argMin() {
        return unravel(argFlat(false));
    }

    /**
     * Sums along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end ({@code -1} = last)
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray sumAxis(int axis) {
        return reduceAxis(axis, Double::sum, 0.0);
    }

    /**
     * Multiplies along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray prodAxis(int axis) {
        return reduceAxis(axis, (x, y) -> x * y, 1.0);
    }

    /**
     * Averages along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray meanAxis(int axis) {
        int ax = normalizeAxis(axis);
        return sumAxis(ax).div(shape[ax]);
    }

    /**
     * Minimum along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public NumJArray minAxis(int axis) {
        return reduceAxis(axis, Math::min);
    }

    /**
     * Maximum along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public NumJArray maxAxis(int axis) {
        return reduceAxis(axis, Math::max);
    }

    /**
     * Population variance along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray varAxis(int axis) {
        int ax = normalizeAxis(axis);
        NumJArray diff = this.sub(meanAxis(ax).expandDims(ax));
        return diff.mul(diff).meanAxis(ax);
    }

    /**
     * Population standard deviation along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray stdAxis(int axis) {
        return varAxis(axis).sqrt();
    }

    /**
     * Euclidean (L2) norm along an axis, e.g. the length of each row vector.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray normAxis(int axis) {
        return this.mul(this).sumAxis(axis).sqrt();
    }

    /**
     * Index of the largest value along an axis (first one wins ties).
     *
     * @param axis the axis to search; negative values count from the end
     * @return a new array (same shape minus {@code axis}) holding indices stored as doubles
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public NumJArray argMaxAxis(int axis) {
        return argReduceAxis(axis, true);
    }

    /**
     * Index of the smallest value along an axis (first one wins ties).
     *
     * @param axis the axis to search; negative values count from the end
     * @return a new array (same shape minus {@code axis}) holding indices stored as doubles
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public NumJArray argMinAxis(int axis) {
        return argReduceAxis(axis, false);
    }

    /**
     * Running total along an axis.
     *
     * @param axis the axis to accumulate along; negative values count from the end
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray cumSumAxis(int axis) {
        int ax = normalizeAxis(axis);
        NumJArray out = copy();
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
     * Logistic sigmoid of a single number, {@code 1 / (1 + e^-x)}, computed in a numerically stable way.
     *
     * @param x the input
     * @return a value in {@code (0, 1)}
     */
    public static double sig(double x) {
        if (x >= 0) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
        double e = Math.exp(x);
        return e / (1.0 + e);
    }

    /**
     * Hyperbolic tangent of a single number.
     *
     * @param x the input
     * @return a value in {@code (-1, 1)}
     */
    public static double tn(double x) {
        return Math.tanh(x);
    }

    /**
     * ReLU activation: {@code max(0, x)} for every element.
     *
     * @return a new array of the same shape
     */
    public NumJArray relu() {
        return map(x -> Math.max(0, x));
    }

    /**
     * Derivative of ReLU: 1 where {@code x > 0}, otherwise 0.
     *
     * @return a new array of 0s and 1s (input is the pre-activation value)
     */
    public NumJArray reluD() {
        return map(x -> x > 0 ? 1.0 : 0.0);
    }

    /**
     * Sigmoid activation applied to every element.
     *
     * @return a new array of the same shape with values in {@code (0, 1)}
     */
    public NumJArray sigmoid() {
        return map(NumJArray::sig);
    }

    /**
     * Derivative of sigmoid, {@code s * (1 - s)} where {@code s = sigmoid(x)}.
     *
     * @return a new array (input is the pre-activation value, not the sigmoid output)
     */
    public NumJArray sigmoidD() {
        return map(x -> {
            double s = sig(x);
            return s * (1.0 - s);
        });
    }

    /**
     * Tanh activation applied to every element.
     *
     * @return a new array of the same shape with values in {@code (-1, 1)}
     */
    public NumJArray tanh() {
        return map(Math::tanh);
    }

    /**
     * Derivative of tanh, {@code 1 - tanh(x)^2}.
     *
     * @return a new array (input is the pre-activation value, not the tanh output)
     */
    public NumJArray tanhD() {
        return map(x -> {
            double t = Math.tanh(x);
            return 1.0 - t * t;
        });
    }

    /**
     * Softmax over all elements (they sum to 1 afterwards). Stable against overflow.
     *
     * @return a new array of the same shape
     */
    public NumJArray softmax() {
        double[] out = toContiguousData();
        if (out.length == 0) return new NumJArray(out, shape);

        double max = Double.NEGATIVE_INFINITY;
        for (double v : out) max = Math.max(max, v);

        double total = 0;
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.exp(out[i] - max);
            total += out[i];
        }
        for (int i = 0; i < out.length; i++) out[i] /= total;

        return new NumJArray(out, shape);
    }

    /**
     * Softmax along one axis, e.g. {@code softmaxAxis(-1)} on a batch of scores gives
     * per-row probabilities. Stable against overflow.
     *
     * @param axis the axis that should sum to 1; negative values count from the end
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray softmaxAxis(int axis) {
        int ax = normalizeAxis(axis);
        NumJArray out = copy();
        int len = out.shape[ax];
        int st = out.strides[ax];

        for (int base : out.laneOffsets(ax)) {
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < len; k++) max = Math.max(max, out.data[base + k * st]);

            double total = 0;
            for (int k = 0; k < len; k++) {
                int p = base + k * st;
                out.data[p] = Math.exp(out.data[p] - max);
                total += out.data[p];
            }
            for (int k = 0; k < len; k++) out.data[base + k * st] /= total;
        }
        return out;
    }

    /**
     * NumPy-style dot product, for any rank:
     *
     *   scalar (0-D) with anything: element-wise multiply
     *   1-D with 1-D: inner product (a 0-D result; read it with {@link #item()})
     *   N-D with 1-D: sum over the last axis of this array
     *   N-D with M-D (M &gt;= 2): sum over the last axis of this array and the
     *   second-to-last axis of {@code b}; result shape is
     *   {@code this.shape[:-1] + b.shape[:-2] + b.shape[-1:]}
     * 
     *
     * @param b the right-hand array
     * @return a new array with the shape described above
     * @throws IllegalArgumentException if {@code b} is null or the contracted dimensions differ
     */
    public NumJArray dot(NumJArray b) {
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }
        if (shape.length == 0 || b.shape.length == 0) {
            return mul(b);
        }

        int nd = shape.length;
        int nb = b.shape.length;
        int k = shape[nd - 1];
        int kb = (nb == 1) ? b.shape[0] : b.shape[nb - 2];
        if (k != kb) {
            throw new IllegalArgumentException(
                "Shapes not aligned for dot: " + Arrays.toString(shape) + " and " + Arrays.toString(b.shape)
                    + " (" + k + " != " + kb + ")");
        }

        int[] leadShape = Arrays.copyOfRange(shape, 0, nd - 1);
        NumJArray a2 = reshape(sizeOf(leadShape), k);

        if (nb == 1) {
            return a2.matMul(b.reshape(k, 1)).reshape(leadShape);
        }

        int[] perm = new int[nb];
        perm[0] = nb - 2;
        for (int i = 1; i < nb - 1; i++) perm[i] = i - 1;
        perm[nb - 1] = nb - 1;

        int cols = 1;
        for (int i = 0; i < nb; i++) if (i != nb - 2) cols *= b.shape[i];

        NumJArray b2 = (nb > 2 ? b.transpose(perm) : b).reshape(k, cols);

        int[] outShape = new int[nd - 1 + nb - 1];
        System.arraycopy(shape, 0, outShape, 0, nd - 1);
        System.arraycopy(b.shape, 0, outShape, nd - 1, nb - 2);
        outShape[outShape.length - 1] = b.shape[nb - 1];

        return a2.matMul(b2).reshape(outShape);
    }

    /**
     * Matrix multiplication with NumPys {@code matmul} rules. Works on views.
     * 
     *   2-D with 2-D: ordinary matrix product
     *   1-D arguments are treated as a row (left) or column (right) vector and that
     *   dimension is removed from the result
     *   more than 2 dimensions: the array is a stack of matrices in the last two axes, and the
     *   leading "batch" axes are broadcast against each othee
     *
     *
     * @param b the right-hand array, with at least 1 dimension
     * @return a new array of shape {@code batch + [rows] + [cols]} (rows/cols dropped for 1-D inputs)
     * @throws IllegalArgumentException if either array is a scalar, the inner dimensions differ,
     *                                  or the batch dimensions cannot be broadcast
     */
    public NumJArray matMul(NumJArray b) {
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }
        if (shape.length == 0 || b.shape.length == 0) {
            throw new IllegalArgumentException("matMul does not accept scalars; use mul instead");
        }

        boolean aVec = shape.length == 1;
        boolean bVec = b.shape.length == 1;
        NumJArray a2 = aVec ? expandDims(0) : this;
        NumJArray b2 = bVec ? b.expandDims(1) : b;

        int na = a2.shape.length;
        int nb = b2.shape.length;
        int m = a2.shape[na - 2];
        int k = a2.shape[na - 1];
        int p = b2.shape[nb - 1];

        if (k != b2.shape[nb - 2]) {
            throw new IllegalArgumentException(
                "Inner dimensions do not match for matMul: " + Arrays.toString(shape) + " x " + Arrays.toString(b.shape));
        }

        int[] batch = broadcastShape(
            Arrays.copyOfRange(a2.shape, 0, na - 2),
            Arrays.copyOfRange(b2.shape, 0, nb - 2));
        int nBatch = batch.length;

        NumJArray aB = a2.broadcastTo(concatInts(batch, m, k));
        NumJArray bB = b2.broadcastTo(concatInts(batch, k, p));

        int count = sizeOf(batch);
        double[] out = new double[count * m * p];

        int[] idx = new int[nBatch];
        int baseA = aB.offset;
        int baseB = bB.offset;
        int aRow = aB.strides[nBatch];
        int aCol = aB.strides[nBatch + 1];
        int bRow = bB.strides[nBatch];
        int bCol = bB.strides[nBatch + 1];

        for (int n = 0; n < count; n++) {
            int outBase = n * m * p;

            for (int i = 0; i < m; i++) {
                int rowA = baseA + i * aRow;
                int rowOut = outBase + i * p;

                for (int kk = 0; kk < k; kk++) {
                    double valA = aB.data[rowA + kk * aCol];
                    int rowB = baseB + kk * bRow;

                    for (int j = 0; j < p; j++) {
                        out[rowOut + j] += valA * bB.data[rowB + j * bCol];
                    }
                }
            }

            for (int d = nBatch - 1; d >= 0; d--) {
                idx[d]++;
                baseA += aB.strides[d];
                baseB += bB.strides[d];
                if (idx[d] < batch[d]) break;
                baseA -= aB.strides[d] * batch[d];
                baseB -= bB.strides[d] * batch[d];
                idx[d] = 0;
            }
        }

        int[] outShape = new int[nBatch + (aVec ? 0 : 1) + (bVec ? 0 : 1)];
        System.arraycopy(batch, 0, outShape, 0, nBatch);
        int pos = nBatch;
        if (!aVec) outShape[pos++] = m;
        if (!bVec) outShape[pos] = p;

        return new NumJArray(out, outShape);
    }

    /**
     * Reverses the order of all axes. Returns a view; no data is copied.
     *
     * @return a transposed view sharing data with this array
     */
    public NumJArray transpose() {
        int rank = shape.length;
        int[] newShape = new int[rank];
        int[] newStrides = new int[rank];

        for (int i = 0; i < rank; i++) {
            newShape[i] = shape[rank - 1 - i];
            newStrides[i] = strides[rank - 1 - i];
        }
        return new NumJArray(data, newShape, newStrides, offset);
    }

    /**
     * Reorders the axes. Returns a view.
     *
     * @param axes a permutation of {@code 0..ndim-1}; {@code axes[i]} is the old axis that becomes axis {@code i}
     * @return a permuted view sharing data with this array
     * @throws IllegalArgumentException if {@code axes} is not a valid permutation
     */
    public NumJArray transpose(int... axes) {
        if (axes == null || axes.length != shape.length) {
            throw new IllegalArgumentException("axes must list every dimension exactly once");
        }
        boolean[] seen = new boolean[shape.length];
        int[] newShape = new int[shape.length];
        int[] newStrides = new int[shape.length];

        for (int i = 0; i < axes.length; i++) {
            int ax = normalizeAxis(axes[i]);
            if (seen[ax]) {
                throw new IllegalArgumentException("Axis " + ax + " appears more than once");
            }
            seen[ax] = true;
            newShape[i] = shape[ax];
            newStrides[i] = strides[ax];
        }
        return new NumJArray(data, newShape, newStrides, offset);
    }

    /**
     * Swaps two axes. Returns a view.
     *
     * @param axis1 the first axis; negative values count from the end
     * @param axis2 the second axis; negative values count from the end
     * @return a view with the two axes exchanged
     * @throws IllegalArgumentException if either axis is out of range
     */
    public NumJArray swapAxes(int axis1, int axis2) {
        int a = normalizeAxis(axis1);
        int b = normalizeAxis(axis2);
        int[] newShape = shape.clone();
        int[] newStrides = strides.clone();

        newShape[a] = shape[b];
        newShape[b] = shape[a];
        newStrides[a] = strides[b];
        newStrides[b] = strides[a];
        return new NumJArray(data, newShape, newStrides, offset);
    }

    /**
     * Gives the array a new shape with the same elements in the same logical order.
     * Returns a view if the data is contiguous, otherwise a copy.
     *
     * @param newShape the new dimensions; at most one may be {@code -1} and is then inferred
     * @return an array with the requested shape
     * @throws IllegalArgumentException if the shape is invalid or holds a different number of elements
     */
    public NumJArray reshape(int... newShape) {
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
            if (known == 0 || size % known != 0) {
                throw new IllegalArgumentException(
                    "Cannot reshape " + size + " elements into " + Arrays.toString(newShape));
            }
            resolved[unknown] = (int) (size / known);
        }

        if (sizeOf(resolved) != size) {
            throw new IllegalArgumentException(
                "New shape must have the same number of elements: " + size + " vs " + Arrays.toString(newShape));
        }

        if (isContiguous()) {
            return new NumJArray(data, resolved, createStrides(resolved), offset);
        }
        return new NumJArray(toContiguousData(), resolved);
    }

    /**
     * Flattens to 1-D. Returns a view when possible, otherwise a copy (like NumPys {@code ravel}).
     *
     * @return a 1-D array with {@link #size()} elements
     */
    public NumJArray ravel() {
        return reshape(size);
    }

    /**
     * Flattens to 1-D, always returning an independent copy.
     *
     * @return a new 1-D array with {@link #size()} elements
     */
    public NumJArray flatten() {
        return new NumJArray(toContiguousData(), new int[]{size});
    }

    /**
     * Makes an independent, contiguous copy (views become normal arrays).
     *
     * @return a new array with the same shape and values
     */
    public NumJArray copy() {
        return new NumJArray(toContiguousData(), shape);
    }

    /**
     * Inserts a new dimension of size 1. Returns a view.
     *
     * @param axis the position of the new axis in the result, from {@code 0} to {@code ndim}
     *             (negative values count from the end of the result)
     * @return a view with one more dimension
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray expandDims(int axis) {
        int ax = normalizeAxis(axis, shape.length + 1);
        int[] newShape = new int[shape.length + 1];
        int[] newStrides = new int[shape.length + 1];

        for (int i = 0, src = 0; i < newShape.length; i++) {
            if (i == ax) {
                newShape[i] = 1;
                newStrides[i] = (src < shape.length) ? shape[src] * strides[src] : 1;
            } else {
                newShape[i] = shape[src];
                newStrides[i] = strides[src];
                src++;
            }
        }
        return new NumJArray(data, newShape, newStrides, offset);
    }

    /**
     * Removes every dimension of size 1. Returns a view.
     *
     * @return a view without singleton dimensions
     */
    public NumJArray squeeze() {
        int kept = 0;
        for (int dim : shape) if (dim != 1) kept++;

        int[] newShape = new int[kept];
        int[] newStrides = new int[kept];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (shape[i] != 1) {
                newShape[j] = shape[i];
                newStrides[j++] = strides[i];
            }
        }
        return new NumJArray(data, newShape, newStrides, offset);
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
    public NumJArray select(int axis, int index) {
        int ax = normalizeAxis(axis);
        int idx = index < 0 ? index + shape[ax] : index;
        if (idx < 0 || idx >= shape[ax]) {
            throw new IndexOutOfBoundsException(
                "Index " + index + " is out of bounds for axis " + ax + " with size " + shape[ax]);
        }
        return new NumJArray(data, removeAt(shape, ax), removeAt(strides, ax), offset + idx * strides[ax]);
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
    public NumJArray slice(int[]... slices) {
        int nd = shape.length;
        if (slices == null) slices = new int[0][];
        if (slices.length > nd) {
            throw new IllegalArgumentException("Got " + slices.length + " slices for a " + nd + "D array");
        }

        int[] newShape = new int[nd];
        int[] newStrides = new int[nd];
        int newOffset = offset;

        for (int i = 0; i < nd; i++) {
            int dim = shape[i];
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
                start = clamp(start, 0, dim);
                stop = clamp(stop, 0, dim);
                count = stop > start ? (stop - start + step - 1) / step : 0;
            } else {
                start = clamp(start, -1, dim - 1);
                stop = clamp(stop, -1, dim - 1);
                count = start > stop ? (start - stop + (-step) - 1) / (-step) : 0;
            }

            newShape[i] = (int) count;
            newStrides[i] = (int) (strides[i] * step);
            if (count > 0) newOffset += (int) start * strides[i];
        }

        return new NumJArray(data, newShape, newStrides, newOffset);
    }

    /**
     * Joins two arrays end to end along an existing axis.
     *
     * @param b    the array to append; same number of dimensions, and same size on every axis except {@code axis}
     * @param axis the axis to join along; negative values count from the end
     * @return a new array whose size along {@code axis} is the sum of both
     * @throws IllegalArgumentException if {@code b} is null, the axis is out of range, or the shapes do not fit
     */
    public NumJArray concat(NumJArray b, int axis) {
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }
        if (shape.length != b.shape.length) {
            throw new IllegalArgumentException("Arrays must have the same number of dimensions.");
        }
        int ax = normalizeAxis(axis);

        for (int i = 0; i < shape.length; i++) {
            if (i != ax && shape[i] != b.shape[i]) {
                throw new IllegalArgumentException(
                    "Shapes must match except on the concatenation axis: "
                        + Arrays.toString(shape) + " vs " + Arrays.toString(b.shape));
            }
        }

        int[] newShape = shape.clone();
        newShape[ax] = shape[ax] + b.shape[ax];
        NumJArray out = zeros(newShape);

        out.sliceAxis(ax, 0, shape[ax]).assign(this);
        out.sliceAxis(ax, shape[ax], newShape[ax]).assign(b);
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
    public NumJArray stack(NumJArray b, int axis) {
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }
        if (!Arrays.equals(shape, b.shape)) {
            throw new IllegalArgumentException(
                "Arrays must have the same shape: " + Arrays.toString(shape) + " vs " + Arrays.toString(b.shape));
        }
        int ax = normalizeAxis(axis, shape.length + 1);

        int[] newShape = new int[shape.length + 1];
        for (int i = 0, src = 0; i < newShape.length; i++) {
            newShape[i] = (i == ax) ? 2 : shape[src++];
        }

        NumJArray out = zeros(newShape);
        out.select(ax, 0).assign(this);
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
    public NumJArray[] split(int axis, int sections) {
        int ax = normalizeAxis(axis);
        if (sections <= 0) {
            throw new IllegalArgumentException("Sections must be greater than 0.");
        }
        if (shape[ax] % sections != 0) {
            throw new IllegalArgumentException(
                "Axis " + ax + " of size " + shape[ax] + " cannot be split into " + sections + " equal parts");
        }

        int part = shape[ax] / sections;
        NumJArray[] result = new NumJArray[sections];
        for (int s = 0; s < sections; s++) {
            result[s] = sliceAxis(ax, s * part, (s + 1) * part);
        }
        return result;
    }

    /**
     * Element-wise {@code ==} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 (equal) and 0.0 (not equal); NaN is never equal to anything
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray equal(NumJArray b) {
        return broadcastOp(b, (x, y) -> flag(x == y));
    }

    /**
     * Element-wise {@code ==} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x == b}, otherwise 0.0
     */
    public NumJArray equal(double b) {
        return map(x -> flag(x == b));
    }

    /**
     * Element-wise {@code !=} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 (different) and 0.0 (equal); NaN is always "not equal"
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray notEqual(NumJArray b) {
        return broadcastOp(b, (x, y) -> flag(x != y));
    }

    /**
     * Element-wise {@code !=} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x != b}, otherwise 0.0
     */
    public NumJArray notEqual(double b) {
        return map(x -> flag(x != b));
    }

    /**
     * Element-wise {@code >} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where {@code this > b}, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray greater(NumJArray b) {
        return broadcastOp(b, (x, y) -> flag(x > y));
    }

    /**
     * Element-wise {@code >} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x > b}, otherwise 0.0
     */
    public NumJArray greater(double b) {
        return map(x -> flag(x > b));
    }

    /**
     * Element-wise {@code >=} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where {@code this >= b}, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray greaterEqual(NumJArray b) {
        return broadcastOp(b, (x, y) -> flag(x >= y));
    }

    /**
     * Element-wise {@code >=} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x >= b}, otherwise 0.0
     */
    public NumJArray greaterEqual(double b) {
        return map(x -> flag(x >= b));
    }

    /**
     * Element-wise {@code <} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where {@code this < b}, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray less(NumJArray b) {
        return broadcastOp(b, (x, y) -> flag(x < y));
    }

    /**
     * Element-wise {@code <} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x < b}, otherwise 0.0
     */
    public NumJArray less(double b) {
        return map(x -> flag(x < b));
    }

    /**
     * Element-wise {@code <=} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where {@code this <= b}, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray lessEqual(NumJArray b) {
        return broadcastOp(b, (x, y) -> flag(x <= y));
    }

    /**
     * Element-wise {@code <=} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x <= b}, otherwise 0.0
     */
    public NumJArray lessEqual(double b) {
        return map(x -> flag(x <= b));
    }

    /**
     * Element-wise logical AND with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where both values are non-zero, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray logicalAnd(NumJArray b) {
        return broadcastOp(b, (x, y) -> flag(x != 0 && y != 0));
    }

    /**
     * Element-wise logical OR with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where at least one value is non-zero, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray logicalOr(NumJArray b) {
        return broadcastOp(b, (x, y) -> flag(x != 0 || y != 0));
    }

    /**
     * Element-wise logical XOR with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where exactly one of the two values is non-zero, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray logicalXor(NumJArray b) {
        return broadcastOp(b, (x, y) -> flag((x != 0) != (y != 0)));
    }

    /**
     * Element-wise logical NOT.
     *
     * @return a new array of 1.0 where the value is 0, otherwise 0.0
     */
    public NumJArray logicalNot() {
        return map(x -> flag(x == 0));
    }

    /**
     * Checks whether every element is non-zero.
     *
     * @return {@code true} if all elements are non-zero (also {@code true} for an empty array)
     */
    public boolean all() {
        for (double v : readOnlyData()) {
            if (v == 0) return false;
        }
        return true;
    }

    /**
     * Checks whether at least one element is non-zero.
     *
     * @return {@code true} if any element is non-zero ({@code false} for an empty array)
     */
    public boolean any() {
        for (double v : readOnlyData()) {
            if (v != 0) return true;
        }
        return false;
    }

    /**
     * {@link #all()} along one axis, e.g. "is every value in this row non-zero?".
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed, holding 1.0 (all non-zero) or 0.0
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray allAxis(int axis) {
        return reduceAxis(axis, (acc, x) -> flag(acc != 0 && x != 0), 1.0);
    }

    /**
     * {@link #any()} along one axis, e.g. "is any value in this row non-zero?".
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed, holding 1.0 (any non-zero) or 0.0
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray anyAxis(int axis) {
        return reduceAxis(axis, (acc, x) -> flag(acc != 0 || x != 0), 0.0);
    }

    /** Default relative tolerance for {@code isClose} / {@code allClose} (same as NumPy). */
    public static final double DEFAULT_RTOL = 1e-5;

    /** Default absolute tolerance for {@code isClose} / {@code allClose} (same as NumPy). */
    public static final double DEFAULT_ATOL = 1e-8;

    /**
     * Element-wise "nearly equal" using the default tolerances ({@code rtol = 1e-5}, {@code atol = 1e-8}).
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where the values are close, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray isClose(NumJArray b) {
        return isClose(b, DEFAULT_RTOL, DEFAULT_ATOL);
    }

    /**
     * Element-wise "nearly equal": true where {@code |x - y| <= atol + rtol * |y|}.
     * Identical values (including two infinities of the same sign) are close; NaN is never close.
     *
     * @param b    the other array; its shape must be broadcastable with this one
     * @param rtol relative tolerance, scaled by {@code |y|}; must be non-negative
     * @param atol absolute tolerance; must be non-negative
     * @return a new array of 1.0 where the values are close, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null, a tolerance is negative, or the shapes cannot be broadcast
     */
    public NumJArray isClose(NumJArray b, double rtol, double atol) {
        checkTolerances(rtol, atol);
        return broadcastOp(b, (x, y) -> flag(closeEnough(x, y, rtol, atol)));
    }

    /**
     * Element-wise "nearly equal" against a scalar, using the default tolerances.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where the values are close, otherwise 0.0
     */
    public NumJArray isClose(double b) {
        return isClose(b, DEFAULT_RTOL, DEFAULT_ATOL);
    }

    /**
     * Element-wise "nearly equal" against a scalar: true where {@code |x - b| <= atol + rtol * |b|}.
     *
     * @param b    the value to compare with
     * @param rtol relative tolerance; must be non-negative
     * @param atol absolute tolerance; must be non-negative
     * @return a new array of 1.0 where the values are close, otherwise 0.0
     * @throws IllegalArgumentException if a tolerance is negative
     */
    public NumJArray isClose(double b, double rtol, double atol) {
        checkTolerances(rtol, atol);
        return map(x -> flag(closeEnough(x, b, rtol, atol)));
    }

    /**
     * Checks that every element is close to the matching element of {@code b} (default tolerances).
     * Handy in tests: {@code assertTrue(result.allClose(expected))}.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return {@code true} if every pair of elements is close
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public boolean allClose(NumJArray b) {
        return isClose(b).all();
    }

    /**
     * Checks that every element is close to the matching element of {@code b}.
     *
     * @param b    the other array; its shape must be broadcastable with this one
     * @param rtol relative tolerance; must be non-negative
     * @param atol absolute tolerance; must be non-negative
     * @return {@code true} if every pair of elements is close
     * @throws IllegalArgumentException if {@code b} is null, a tolerance is negative, or the shapes cannot be broadcast
     */
    public boolean allClose(NumJArray b, double rtol, double atol) {
        return isClose(b, rtol, atol).all();
    }

    /**
     * Counts the non-zero elements (NaN counts as non-zero).
     *
     * @return the number of elements that are not 0
     */
    public int countNonzero() {
        int count = 0;
        for (double v : readOnlyData()) {
            if (v != 0) count++;
        }
        return count;
    }

    /**
     * Counts the non-zero elements along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed, holding the counts (stored as doubles)
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray countNonzero(int axis) {
        return map(x -> flag(x != 0)).sumAxis(axis);
    }

    /**
     * Finds the coordinates of every non-zero element, in row-major order.
     *
     * @return one {@code int[]} per dimension: {@code result[d][i]} is the index along axis {@code d}
     *         of the i-th non-zero element (so {@code get(result[0][i], result[1][i])} reads it on a 2-D array)
     * @throws IllegalArgumentException if the array is 0-D
     */
    public int[][] nonzero() {
        if (shape.length == 0) {
            throw new IllegalArgumentException("nonzero() needs at least 1 dimension");
        }
        double[] d = readOnlyData();
        int count = 0;
        for (double v : d) if (v != 0) count++;

        int nd = shape.length;
        int[][] out = new int[nd][count];
        int[] idx = new int[nd];
        int w = 0;

        for (int i = 0; i < d.length; i++) {
            if (d[i] != 0) {
                for (int k = 0; k < nd; k++) out[k][w] = idx[k];
                w++;
            }
            for (int dim = nd - 1; dim >= 0; dim--) {
                if (++idx[dim] < shape[dim]) break;
                idx[dim] = 0;
            }
        }
        return out;
    }

    /**
     * Finds the coordinates of every non-zero element as rows of a table.
     *
     * @return a new array of shape {@code [count, ndim]}; row i holds the coordinates of the i-th
     *         non-zero element (stored as doubles)
     * @throws IllegalArgumentException if the array is 0-D
     */
    public NumJArray argwhere() {
        int[][] nz = nonzero();
        int nd = nz.length;
        int count = nz[0].length;

        double[] out = new double[count * nd];
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
    public NumJArray take(int[] indices) {
        if (indices == null) {
            throw new IllegalArgumentException("indices must not be null");
        }
        double[] out = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            out[i] = data[pointer(indices[i])];
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
    public NumJArray take(int axis, int... indices) {
        if (indices == null) {
            throw new IllegalArgumentException("indices must not be null");
        }
        int ax = normalizeAxis(axis);
        int len = shape[ax];

        int[] picked = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i] < 0 ? indices[i] + len : indices[i];
            if (idx < 0 || idx >= len) {
                throw new IndexOutOfBoundsException(
                    "Index " + indices[i] + " is out of bounds for axis " + ax + " with size " + len);
            }
            picked[i] = idx;
        }

        int[] newShape = shape.clone();
        newShape[ax] = indices.length;
        NumJArray out = zeros(newShape);
        for (int i = 0; i < picked.length; i++) {
            out.select(ax, i).assign(this.select(ax, picked[i]));
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
    public NumJArray put(int[] flatIndices, double value) {
        int[] slots = resolveFlat(flatIndices);
        for (int s : slots) data[s] = value;
        return this;
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
    public NumJArray put(int[] flatIndices, NumJArray values) {
        if (values == null) {
            throw new IllegalArgumentException("values must not be null");
        }
        int[] slots = resolveFlat(flatIndices);
        double[] v = values.toContiguousData();
        if (v.length == 0) {
            if (slots.length == 0) return this;
            throw new IllegalArgumentException("values must not be empty");
        }
        for (int i = 0; i < slots.length; i++) data[slots[i]] = v[i % v.length];
        return this;
    }

    /**
     * Sorts along the last axis (ascending). The original is not changed.
     *
     * @return a new, sorted array of the same shape
     * @throws IllegalArgumentException if the array is 0-D
     */
    public NumJArray sort() {
        return sort(-1);
    }

    /**
     * Sorts along an axis (ascending). The original is not changed.
     *
     * @param axis the axis to sort along; negative values count from the end
     * @return a new, sorted array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray sort(int axis) {
        int ax = normalizeAxis(axis);
        NumJArray out = copy();
        int len = out.shape[ax];
        int st = out.strides[ax];
        double[] lane = new double[len];

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
    public NumJArray sortFlat() {
        double[] d = toContiguousData();
        Arrays.sort(d);
        return new NumJArray(d, new int[]{d.length});
    }

    /**
     * Finds the order that would sort the array along the last axis. Ties keep their original order.
     *
     * @return a new array of the same shape holding indices (stored as doubles)
     * @throws IllegalArgumentException if the array is 0-D
     */
    public NumJArray argsort() {
        return argsort(-1);
    }

    /**
     * Finds the order that would sort the array along an axis. Ties keep their original order.
     * Use {@code take(axis, order)} or {@code flip} on the result for other orderings.
     *
     * @param axis the axis to sort along; negative values count from the end
     * @return a new array of the same shape holding indices along {@code axis} (stored as doubles)
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray argsort(int axis) {
        int ax = normalizeAxis(axis);
        NumJArray out = zeros(shape);
        int len = shape[ax];
        int srcStride = strides[ax];
        int dstStride = out.strides[ax];

        int[] srcLanes = laneOffsets(ax);
        int[] dstLanes = out.laneOffsets(ax);
        final double[] vals = new double[len];
        Integer[] order = new Integer[len];

        for (int l = 0; l < srcLanes.length; l++) {
            for (int k = 0; k < len; k++) {
                vals[k] = data[srcLanes[l] + k * srcStride];
                order[k] = k;
            }
            Arrays.sort(order, (i, j) -> compareNaNLast(vals[i], vals[j]));
            for (int k = 0; k < len; k++) out.data[dstLanes[l] + k * dstStride] = order[k];
        }
        return out;
    }

    /**
     * Finds the order that would sort all elements as one flat list. Ties keep their original order.
     *
     * @return a new 1-D array of flat positions (stored as doubles)
     */
    public NumJArray argsortFlat() {
        return flatten().argsort(0);
    }

    /**
     * Finds the distinct values, sorted ascending. All NaNs count as one value, and 0.0 equals -0.0.
     *
     * @return a new 1-D array of the unique values
     */
    public NumJArray unique() {
        return uniqueCounts()[0];
    }

    /**
     * Finds the distinct values together with how often each occurs.
     *
     * @return an array of two 1-D arrays: {@code [0]} the sorted unique values, {@code [1]} their counts
     *         (stored as doubles)
     */
    public NumJArray[] uniqueCounts() {
        double[] d = toContiguousData();
        Arrays.sort(d);

        double[] values = new double[d.length];
        double[] counts = new double[d.length];
        int n = 0;

        for (double v : d) {
            if (n > 0 && (values[n - 1] == v || (Double.isNaN(values[n - 1]) && Double.isNaN(v)))) {
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

    /**
     * Repeats every element of the flattened array.
     * Example: {@code [1, 2].repeat(2)} gives {@code [1, 1, 2, 2]}.
     *
     * @param repeats how many times to repeat each element; must be non-negative
     * @return a new 1-D array of length {@code size * repeats}
     * @throws IllegalArgumentException if {@code repeats} is negative
     */
    public NumJArray repeat(int repeats) {
        return flatten().repeat(repeats, 0);
    }

    /**
     * Repeats every slice along an axis, e.g. {@code repeat(2, 0)} doubles each row in place.
     *
     * @param repeats how many times to repeat each slice; must be non-negative
     * @param axis    the axis to repeat along; negative values count from the end
     * @return a new array that is {@code repeats} times longer along {@code axis}
     * @throws IllegalArgumentException if {@code repeats} is negative or the axis is out of range
     */
    public NumJArray repeat(int repeats, int axis) {
        if (repeats < 0) {
            throw new IllegalArgumentException("repeats must be non-negative, got " + repeats);
        }
        int ax = normalizeAxis(axis);
        int len = shape[ax];
        if ((long) len * repeats > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Result is too large");
        }

        int[] idx = new int[len * repeats];
        for (int i = 0; i < len; i++) {
            for (int r = 0; r < repeats; r++) idx[i * repeats + r] = i;
        }
        return take(ax, idx);
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
    public NumJArray repeat(int[] repeats, int axis) {
        if (repeats == null) {
            throw new IllegalArgumentException("repeats must not be null");
        }
        int ax = normalizeAxis(axis);
        if (repeats.length != shape[ax]) {
            throw new IllegalArgumentException(
                "repeats needs " + shape[ax] + " entries for axis " + ax + ", got " + repeats.length);
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
        return take(ax, idx);
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
    public NumJArray tile(int... reps) {
        if (reps == null) {
            throw new IllegalArgumentException("reps must not be null");
        }
        for (int r : reps) {
            if (r < 0) {
                throw new IllegalArgumentException("reps must be non-negative, got " + r);
            }
        }
        if (reps.length == 0) return copy();

        int nd = Math.max(shape.length, reps.length);
        int leadS = nd - shape.length;
        int leadR = nd - reps.length;

        int[] baseShape = new int[nd];
        int[] baseStrides = new int[nd];
        int[] outShape = new int[nd];
        for (int i = 0; i < nd; i++) {
            baseShape[i] = (i < leadS) ? 1 : shape[i - leadS];
            baseStrides[i] = (i < leadS) ? 0 : strides[i - leadS];
            outShape[i] = baseShape[i] * ((i < leadR) ? 1 : reps[i - leadR]);
        }

        double[] out = new double[sizeOf(outShape)];
        if (out.length == 0) return new NumJArray(out, outShape);

        int[] idx = new int[nd];
        for (int i = 0; i < out.length; i++) {
            int pos = offset;
            for (int d = 0; d < nd; d++) {
                pos += (idx[d] % baseShape[d]) * baseStrides[d];
            }
            out[i] = data[pos];
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
    public NumJArray pad(int[][] padding) {

        if (padding == null || padding.length != shape.length) {
            throw new IllegalArgumentException("Padding array length must match array dimensions");
        }

        int nd = shape.length;
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

            newShape[d] = shape[d] + before + after;
        }

        NumJArray result = NumJArray.zeros(newShape);
        int totalElements = sizeOf(shape);
        int[] srcIdx = new int[nd];
        int[] destIdx = new int[nd];

        for (int i = 0; i < totalElements; i++) {

            int srcFlat = offset;

            for (int d = 0; d < nd; d++) {
                srcFlat += srcIdx[d] * strides[d];
                destIdx[d] = srcIdx[d] + padding[d][0];
            }

            int destFlat = result.getFlatIndex(destIdx);

            result.data[destFlat] = this.data[srcFlat];

            for (int d = nd - 1; d >= 0; d--) {

                srcIdx[d]++;

                if (srcIdx[d] < shape[d]) {
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
    public NumJArray flip() {
        int[] all = new int[shape.length];
        for (int i = 0; i < all.length; i++) all[i] = i;
        return flip(all);
    }

    /**
     * Reverses the order of elements along the given axes. Returns a view (negative strides).
     *
     * @param axes the axes to reverse; negative values count from the end
     * @return a flipped view sharing data with this array
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public NumJArray flip(int... axes) {
        if (axes == null) {
            throw new IllegalArgumentException("axes must not be null");
        }
        boolean[] seen = new boolean[shape.length];
        int[] newStrides = strides.clone();
        int newOffset = offset;

        for (int axis : axes) {
            int ax = normalizeAxis(axis);
            if (seen[ax]) {
                throw new IllegalArgumentException("Axis " + ax + " is listed more than once");
            }
            seen[ax] = true;
            if (shape[ax] > 0) {
                newOffset += (shape[ax] - 1) * strides[ax];
                newStrides[ax] = -strides[ax];
            }
        }
        return new NumJArray(data, shape, newStrides, newOffset);
    }

    /**
     * Shifts the elements of the flattened array, wrapping around, and keeps the shape.
     * Example: {@code [1, 2, 3, 4, 5].roll(2)} gives {@code [4, 5, 1, 2, 3]}.
     *
     * @param shift how far to move elements forward; negative values move them backward
     * @return a new array of the same shape
     */
    public NumJArray roll(int shift) {
        if (size == 0) return copy();

        int s = Math.floorMod(shift, size);
        int[] idx = new int[size];
        for (int i = 0; i < size; i++) idx[i] = Math.floorMod(i - s, size);
        return take(idx).reshape(shape);
    }

    /**
     * Shifts the slices along one axis, wrapping around (e.g. {@code roll(1, 0)} moves each row down one).
     *
     * @param shift how far to move slices forward; negative values move them backward
     * @param axis  the axis to shift along; negative values count from the end
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray roll(int shift, int axis) {
        int ax = normalizeAxis(axis);
        int len = shape[ax];
        if (len == 0) return copy();

        int s = Math.floorMod(shift, len);
        int[] idx = new int[len];
        for (int i = 0; i < len; i++) idx[i] = Math.floorMod(i - s, len);
        return take(ax, idx);
    }

    /**
     * Running total over the flattened array. (Use {@link #cumSumAxis(int)} to keep the shape.)
     *
     * @return a new 1-D array of {@link #size()} partial sums
     */
    public NumJArray cumSum() {
        double[] d = toContiguousData();
        for (int i = 1; i < d.length; i++) d[i] += d[i - 1];
        return new NumJArray(d, new int[]{d.length});
    }

    /**
     * Running product over the flattened array.
     *
     * @return a new 1-D array of {@link #size()} partial products
     */
    public NumJArray cumProd() {
        double[] d = toContiguousData();
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
    public NumJArray cumProdAxis(int axis) {
        int ax = normalizeAxis(axis);
        NumJArray out = copy();
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
     * Differences between neighbouring elements along the last axis ({@code a[i+1] - a[i]}).
     *
     * @return a new array that is one shorter along the last axis
     * @throws IllegalArgumentException if the array is 0-D
     */
    public NumJArray diff() {
        return diff(1, -1);
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
    public NumJArray diff(int n, int axis) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative, got " + n);
        }
        int ax = normalizeAxis(axis);

        NumJArray cur = this;
        for (int step = 0; step < n; step++) {
            int len = cur.shape[ax];
            if (len == 0) break;
            cur = cur.sliceAxis(ax, 1, len).sub(cur.sliceAxis(ax, 0, len - 1));
        }
        return (cur == this) ? copy() : cur;
    }

    /**
     * Median of all elements.
     *
     * @return the middle value (the mean of the two middle values for an even count); NaN if any element is NaN
     * @throws IllegalStateException if the array is empty
     */
    public double median() {
        return quantile(0.5);
    }

    /**
     * Median along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public NumJArray median(int axis) {
        return quantile(0.5, axis);
    }

    /**
     * Percentile of all elements.
     *
     * @param p the percentile from 0 to 100
     * @return the interpolated value; NaN if any element is NaN
     * @throws IllegalArgumentException if {@code p} is outside {@code [0, 100]}
     * @throws IllegalStateException    if the array is empty
     */
    public double percentile(double p) {
        if (!(p >= 0 && p <= 100)) {
            throw new IllegalArgumentException("percentile must be between 0 and 100, got " + p);
        }
        return quantile(p / 100.0);
    }

    /**
     * Percentile along an axis.
     *
     * @param p    the percentile from 0 to 100
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if {@code p} is outside {@code [0, 100]}, or the axis is out of range or has length 0
     */
    public NumJArray percentile(double p, int axis) {
        if (!(p >= 0 && p <= 100)) {
            throw new IllegalArgumentException("percentile must be between 0 and 100, got " + p);
        }
        return quantile(p / 100.0, axis);
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
    public double quantile(double q) {
        checkQuantile(q);
        requireNonEmpty("quantile");
        double[] d = toContiguousData();
        Arrays.sort(d);
        return interpolateSorted(d, q);
    }

    /**
     * Quantile along an axis.
     *
     * @param q    the quantile from 0 to 1
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if {@code q} is outside {@code [0, 1]}, or the axis is out of range or has length 0
     */
    public NumJArray quantile(double q, int axis) {
        checkQuantile(q);
        int ax = normalizeAxis(axis);
        int len = shape[ax];
        if (len == 0) {
            throw new IllegalArgumentException("Cannot take a quantile over axis " + ax + " because it has length 0");
        }

        int st = strides[ax];
        int[] lanes = laneOffsets(ax);
        double[] out = new double[lanes.length];
        double[] lane = new double[len];

        for (int l = 0; l < lanes.length; l++) {
            for (int k = 0; k < len; k++) lane[k] = data[lanes[l] + k * st];
            Arrays.sort(lane);
            out[l] = interpolateSorted(lane, q);
        }
        return new NumJArray(out, removeAt(shape, ax));
    }

    /**
     * Creates an {@code n x n} identity matrix (same as {@link #identity(int)}).
     *
     * @param n the number of rows and columns
     * @return a new 2-D array with ones on the diagonal
     * @throws IllegalArgumentException if {@code n} is negative
     */
    public static NumJArray eye(int n) {
        return eye(n, n, 0);
    }

    /**
     * Creates an {@code n x m} matrix with ones on the main diagonal.
     *
     * @param n the number of rows
     * @param m the number of columns
     * @return a new 2-D array
     * @throws IllegalArgumentException if a size is negative
     */
    public static NumJArray eye(int n, int m) {
        return eye(n, m, 0);
    }

    /**
     * Creates an {@code n x m} matrix with ones on a chosen diagonal.
     *
     * @param n the number of rows
     * @param m the number of columns
     * @param k which diagonal: 0 is the main one, positive values are above it, negative values below
     * @return a new 2-D array
     * @throws IllegalArgumentException if a size is negative
     */
    public static NumJArray eye(int n, int m, int k) {
        double[] d = new double[sizeOf(checkShape(new int[]{n, m}))];
        for (int i = 0; i < n; i++) {
            long j = (long) i + k;
            if (j >= 0 && j < m) d[i * m + (int) j] = 1.0;
        }
        return new NumJArray(d, new int[]{n, m});
    }

    /**
     * Builds a diagonal matrix from a 1-D array, or extracts the main diagonal of a 2-D array.
     *
     * @return for 1-D input an {@code n x n} matrix; for 2-D input a new 1-D array (a copy)
     * @throws IllegalArgumentException if the array is not 1-D or 2-D
     */
    public NumJArray diag() {
        return diag(0);
    }

    /**
     * Builds a matrix with this 1-D array on a chosen diagonal, or extracts that diagonal from a 2-D array.
     *
     * @param k which diagonal: 0 is the main one, positive values are above it, negative values below
     * @return for 1-D input a square matrix of size {@code n + |k|}; for 2-D input a new 1-D array (a copy,
     *         empty if the diagonal does not exist)
     * @throws IllegalArgumentException if the array is not 1-D or 2-D
     */
    public NumJArray diag(int k) {
        if (shape.length == 1) {
            int n = shape[0];
            int dim = n + Math.abs(k);
            double[] out = new double[sizeOf(checkShape(new int[]{dim, dim}))];
            double[] v = toContiguousData();
            for (int i = 0; i < n; i++) {
                int row = (k >= 0) ? i : i - k;
                int col = (k >= 0) ? i + k : i;
                out[row * dim + col] = v[i];
            }
            return new NumJArray(out, new int[]{dim, dim});
        }

        if (shape.length == 2) {
            int len;
            int start = offset;
            if (k >= 0) {
                len = (int) Math.max(0, Math.min(shape[0], (long) shape[1] - k));
                if (len > 0) start += k * strides[1];
            } else {
                len = (int) Math.max(0, Math.min((long) shape[0] + k, shape[1]));
                if (len > 0) start += -k * strides[0];
            }
            return new NumJArray(data, new int[]{len}, new int[]{strides[0] + strides[1]}, start).copy();
        }

        throw new IllegalArgumentException("diag() needs a 1D or 2D array, got " + shape.length + "D");
    }

    /**
     * Keeps the lower triangle (main diagonal and below) of every matrix in the last two axes, zeroing the rest.
     *
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public NumJArray tril() {
        return tril(0);
    }

    /**
     * Keeps everything on or below a chosen diagonal of every matrix in the last two axes, zeroing the rest.
     *
     * @param k which diagonal: 0 is the main one, positive values include more above it, negative values exclude more
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public NumJArray tril(int k) {
        return triangle(k, true);
    }

    /**
     * Keeps the upper triangle (main diagonal and above) of every matrix in the last two axes, zeroing the rest.
     *
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public NumJArray triu() {
        return triu(0);
    }

    /**
     * Keeps everything on or above a chosen diagonal of every matrix in the last two axes, zeroing the rest.
     *
     * @param k which diagonal: 0 is the main one, positive values exclude more below, negative values include more
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public NumJArray triu(int k) {
        return triangle(k, false);
    }

    /**
     * Outer product of two arrays, each treated as a flat list: {@code result[i][j] = this[i] * b[j]}.
     *
     * @param b the other array (any shape; flattened)
     * @return a new 2-D array of shape {@code [this.size, b.size]}
     * @throws IllegalArgumentException if {@code b} is null
     */
    public NumJArray outer(NumJArray b) {
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }
        double[] x = readOnlyData();
        double[] y = b.readOnlyData();
        double[] out = new double[sizeOf(new int[]{x.length, y.length})];

        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < y.length; j++) out[i * y.length + j] = x[i] * y[j];
        }
        return new NumJArray(out, new int[]{x.length, y.length});
    }

    /**
     * Marks the NaN elements.
     *
     * @return a new array of 1.0 where the value is NaN, otherwise 0.0
     */
    public NumJArray isNaN() {
        return map(x -> flag(Double.isNaN(x)));
    }

    /**
     * Marks the infinite elements (positive or negative).
     *
     * @return a new array of 1.0 where the value is +Infinity or -Infinity, otherwise 0.0
     */
    public NumJArray isInf() {
        return map(x -> flag(Double.isInfinite(x)));
    }

    /**
     * Marks the finite elements (neither NaN nor infinite).
     *
     * @return a new array of 1.0 where the value is finite, otherwise 0.0
     */
    public NumJArray isFinite() {
        return map(x -> flag(Double.isFinite(x)));
    }

    /**
     * Element-wise larger of two arrays, with broadcasting. NaN in either input gives NaN.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code max(this, b)}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray maximum(NumJArray b) {
        return broadcastOp(b, Math::max);
    }

    /**
     * Element-wise larger of every element and a scalar (e.g. {@code maximum(0)} is ReLU).
     *
     * @param b the scalar
     * @return a new array of {@code max(x, b)}
     */
    public NumJArray maximum(double b) {
        return map(x -> Math.max(x, b));
    }

    /**
     * Element-wise smaller of two arrays, with broadcasting. NaN in either input gives NaN.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code min(this, b)}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray minimum(NumJArray b) {
        return broadcastOp(b, Math::min);
    }

    /**
     * Element-wise smaller of every element and a scalar.
     *
     * @param b the scalar
     * @return a new array of {@code min(x, b)}
     */
    public NumJArray minimum(double b) {
        return map(x -> Math.min(x, b));
    }

    // HELPERS

    /** Turns flat indices into slots in {@code data}, failing before anything is written. */
    private int[] resolveFlat(int[] flatIndices) {
        if (flatIndices == null) {
            throw new IllegalArgumentException("flatIndices must not be null");
        }
        int[] slots = new int[flatIndices.length];
        for (int i = 0; i < slots.length; i++) slots[i] = pointer(flatIndices[i]);
        return slots;
    }

    /** Ordering for sorting: normal numeric order, NaN last, and 0.0 equal to -0.0. */
    private static int compareNaNLast(double x, double y) {
        if (x < y) return -1;
        if (x > y) return 1;
        boolean nx = Double.isNaN(x);
        boolean ny = Double.isNaN(y);
        return (nx == ny) ? 0 : (nx ? 1 : -1);
    }

    private static void checkQuantile(double q) {
        if (!(q >= 0 && q <= 1)) {
            throw new IllegalArgumentException("quantile must be between 0 and 1, got " + q);
        }
    }

    private static double interpolateSorted(double[] sorted, double q) {
        int n = sorted.length;
        if (Double.isNaN(sorted[n - 1])) return Double.NaN;

        double pos = q * (n - 1);
        int lo = (int) Math.floor(pos);
        int hi = Math.min(lo + 1, n - 1);
        double frac = pos - lo;
        if (frac == 0 || sorted[lo] == sorted[hi]) return sorted[lo];
        return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
    }

    private NumJArray triangle(int k, boolean lower) {
        int nd = shape.length;
        if (nd < 2) {
            throw new IllegalArgumentException("tril/triu need at least 2 dimensions, got " + nd + "D");
        }
        double[] d = toContiguousData();
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

    /** Converts a boolean to the librarys truth values: 1.0 for true, 0.0 for false. */
    private static double flag(boolean condition) {
        return condition ? 1.0 : 0.0;
    }

    private static boolean closeEnough(double x, double y, double rtol, double atol) {
        if (x == y) return true;
        if (Double.isInfinite(x) || Double.isInfinite(y)) return false;
        return Math.abs(x - y) <= atol + rtol * Math.abs(y);
    }

    private static void checkTolerances(double rtol, double atol) {
        if (rtol < 0 || atol < 0) {
            throw new IllegalArgumentException("rtol and atol must be non-negative");
        }
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
    public double reduce(DoubleBinaryOperator op) {
        requireOperator(op);
        requireNonEmpty("reduce");
        double[] d = readOnlyData();
        double acc = d[0];
        for (int i = 1; i < d.length; i++) acc = op.applyAsDouble(acc, d[i]);
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
    public double reduce(DoubleBinaryOperator op, double init) {
        requireOperator(op);
        double acc = init;
        for (double v : readOnlyData()) acc = op.applyAsDouble(acc, v);
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
    public NumJArray reduceAxis(int axis, DoubleBinaryOperator op) {
        requireOperator(op);
        return reduceLanes(axis, op, null);
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
    public NumJArray reduceAxis(int axis, DoubleBinaryOperator op, double init) {
        requireOperator(op);
        return reduceLanes(axis, op, init);
    }

    /**
     * Sums over several axes at once, e.g. {@code sum(0, 2)} on shape {@code [2, 3, 4]} gives shape {@code [3]}.
     * (Call {@link #sum()} with no arguments for a plain {@code double} total.)
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public NumJArray sum(int... axes) {
        return reduceAxes(axes, NumJArray::sumAxis);
    }

    /**
     * Multiplies over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public NumJArray prod(int... axes) {
        return reduceAxes(axes, NumJArray::prodAxis);
    }

    /**
     * Averages over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public NumJArray mean(int... axes) {
        NumJArray total = sum(axes);
        long count = 1;
        for (int ax : axes) count *= shape[normalizeAxis(ax)];
        return total.div(count);
    }

    /**
     * Minimum over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range, listed twice, or has length 0
     */
    public NumJArray min(int... axes) {
        return reduceAxes(axes, NumJArray::minAxis);
    }

    /**
     * Maximum over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range, listed twice, or has length 0
     */
    public NumJArray max(int... axes) {
        return reduceAxes(axes, NumJArray::maxAxis);
    }

    /**
     * Position of the largest element in the flattened (row-major) array.
     *
     * @return the flat index of the first maximum
     * @throws IllegalStateException if the array is empty
     */
    public int argMaxFlat() {
        return argFlat(true);
    }

    /**
     * Position of the smallest element in the flattened (row-major) array.
     *
     * @return the flat index of the first minimum
     * @throws IllegalStateException if the array is empty
     */
    public int argMinFlat() {
        return argFlat(false);
    }

    /**
     * Combines this array with another using your own operator, with broadcasting.
     *
     * @param b  the other array; its shape must be broadcastable with this one
     * @param op called as {@code op(thisValue, bValue)}
     * @return a new array holding the results
     * @throws IllegalArgumentException if {@code b} or {@code op} is null, or the shapes cannot be broadcast
     */
    public NumJArray elementwise(NumJArray b, DoubleBinaryOperator op) {
        requireOperator(op);
        return broadcastOp(b, op);
    }

    /**
     * Combines every element with a scalar using your own operator.
     *
     * @param b  the scalar
     * @param op called as {@code op(thisValue, b)}
     * @return a new array holding the results
     * @throws IllegalArgumentException if {@code op} is null
     */
    public NumJArray elementwise(double b, DoubleBinaryOperator op) {
        requireOperator(op);
        return map(x -> op.applyAsDouble(x, b));
    }

    /**
     * Applies your own function to every element (same as {@link #map(DoubleUnaryOperator)}).
     *
     * @param op the function to apply
     * @return a new array of the same shape
     * @throws IllegalArgumentException if {@code op} is null
     */
    public NumJArray elementwise(DoubleUnaryOperator op) {
        return map(op);
    }

    /** Computes a value from an elements coordinates; used by {@link #fill(CoordinateFunction)}. */
    @FunctionalInterface
    public interface CoordinateFunction {
        /**
         * @param coordinate one index per dimension; the array is reused between calls, so copy it if you keep it
         * @return the value to store at that position
         */
        double apply(int[] coordinate);
    }

    /**
     * Applies a function to every element, storing the result back into this array.
     *
     * @param op the function to apply
     * @return this array
     * @throws IllegalArgumentException if {@code op} is null
     */
    public NumJArray elementwisei(DoubleUnaryOperator op) {
        if (op == null) {
            throw new IllegalArgumentException("op must not be null");
        }
        for (int p : positions()) data[p] = op.applyAsDouble(data[p]);
        return this;
    }

    /**
     * Combines every element with a scalar using your operator, storing the result in this array.
     *
     * @param b  the scalar
     * @param op called as {@code op(thisValue, b)}
     * @return this array
     * @throws IllegalArgumentException if {@code op} is null
     */
    public NumJArray elementwisei(double b, DoubleBinaryOperator op) {
        requireOperator(op);
        for (int p : positions()) data[p] = op.applyAsDouble(data[p], b);
        return this;
    }

    /**
     * Combines this array with another using your operator, storing the result in this array.
     * {@code b} is broadcast to this arrays shape; the result can never be larger than this array.
     * Do not call this on a {@link #broadcastTo} view (several positions share one slot).
     *
     * @param b  an array broadcastable to this arrays shape
     * @param op called as {@code op(thisValue, bValue)}
     * @return this array
     * @throws IllegalArgumentException if {@code b} or {@code op} is null, or {@code b} cannot be broadcast to this shape
     */
    public NumJArray elementwisei(NumJArray b, DoubleBinaryOperator op) {
        requireOperator(op);
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }
        double[] other = b.broadcastTo(shape).toContiguousData();
        int[] pos = positions();
        for (int i = 0; i < pos.length; i++) {
            data[pos[i]] = op.applyAsDouble(data[pos[i]], other[i]);
        }
        return this;
    }

    /**
     * In-place addition of a scalar.
     *
     * @param b the scalar
     * @return this array
     */
    public NumJArray addi(double b) {
        return elementwisei(b, (x, y) -> x + y);
    }

    /**
     * In-place addition of an array (broadcast to this shape).
     *
     * @param b an array broadcastable to this arrays shape
     * @return this array
     * @throws IllegalArgumentException if {@code b} cannot be broadcast to this shape
     */
    public NumJArray addi(NumJArray b) {
        return elementwisei(b, (x, y) -> x + y);
    }

    /**
     * In-place subtraction of a scalar.
     *
     * @param b the scalar
     * @return this array
     */
    public NumJArray subi(double b) {
        return elementwisei(b, (x, y) -> x - y);
    }

    /**
     * In-place subtraction of an array (broadcast to this shape).
     *
     * @param b an array broadcastable to this arrays shape
     * @return this array
     * @throws IllegalArgumentException if {@code b} cannot be broadcast to this shape
     */
    public NumJArray subi(NumJArray b) {
        return elementwisei(b, (x, y) -> x - y);
    }

    /**
     * In-place multiplication by a scalar.
     *
     * @param b the scalar
     * @return this array
     */
    public NumJArray muli(double b) {
        return elementwisei(b, (x, y) -> x * y);
    }

    /**
     * In-place multiplication by an array (broadcast to this shape).
     *
     * @param b an array broadcastable to this arrays shape
     * @return this array
     * @throws IllegalArgumentException if {@code b} cannot be broadcast to this shape
     */
    public NumJArray muli(NumJArray b) {
        return elementwisei(b, (x, y) -> x * y);
    }

    /**
     * In-place division by a scalar.
     *
     * @param b the scalar
     * @return this array
     */
    public NumJArray divi(double b) {
        return elementwisei(b, (x, y) -> x / y);
    }

    /**
     * In-place division by an array (broadcast to this shape).
     *
     * @param b an array broadcastable to this arrays shape
     * @return this array
     * @throws IllegalArgumentException if {@code b} cannot be broadcast to this shape
     */
    public NumJArray divi(NumJArray b) {
        return elementwisei(b, (x, y) -> x / y);
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
    public NumJArray assign(NumJArray src) {
        return elementwisei(src, (x, y) -> y);
    }

    /**
     * Sets every element to one value.
     *
     * @param value the value to store
     * @return this array
     */
    public NumJArray fill(double value) {
        for (int p : positions()) data[p] = value;
        return this;
    }

    /**
     * Sets every element from a function of its coordinates, e.g.
     * {@code zeros(3, 3).fill(c -> c[0] == c[1] ? 1 : 0)} builds an identity matrix.
     *
     * @param f computes the value for the element at the given coordinates
     * @return this array
     * @throws IllegalArgumentException if {@code f} is null
     */
    public NumJArray fill(CoordinateFunction f) {
        if (f == null) {
            throw new IllegalArgumentException("f must not be null");
        }
        if (size == 0) return this;

        int nd = shape.length;
        int[] idx = new int[nd];
        int[] pos = positions();

        for (int i = 0; i < pos.length; i++) {
            data[pos[i]] = f.apply(idx);
            for (int d = nd - 1; d >= 0; d--) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return this;
    }

    /**
     * Sets elements to one value wherever a mask is non-zero (e.g. {@code x.putWhere(x.less(0), 0)}).
     *
     * @param mask  non-zero = true; must be broadcastable to this arrays shape
     * @param value the value to store where the mask is true
     * @return this array
     * @throws IllegalArgumentException if {@code mask} is null or cannot be broadcast to this shape
     */
    public NumJArray putWhere(NumJArray mask, double value) {
        if (mask == null) {
            throw new IllegalArgumentException("mask must not be null");
        }
        double[] m = mask.broadcastTo(shape).toContiguousData();
        int[] pos = positions();
        for (int i = 0; i < pos.length; i++) {
            if (m[i] != 0) data[pos[i]] = value;
        }
        return this;
    }

    /**
     * Copies values from another array wherever a mask is non-zero.
     *
     * @param mask   non-zero = true; must be broadcastable to this arrays shape
     * @param values the replacement values; must be broadcastable to this arrays shape
     * @return this array
     * @throws IllegalArgumentException if an argument is null or cannot be broadcast to this shape
     */
    public NumJArray putWhere(NumJArray mask, NumJArray values) {
        if (mask == null || values == null) {
            throw new IllegalArgumentException("mask and values must not be null");
        }
        double[] m = mask.broadcastTo(shape).toContiguousData();
        double[] v = values.broadcastTo(shape).toContiguousData();
        int[] pos = positions();
        for (int i = 0; i < pos.length; i++) {
            if (m[i] != 0) data[pos[i]] = v[i];
        }
        return this;
    }

    /**
     * Sets elements to a value wherever a test on the element returns true.
     * Example: {@code x.conditionalPut(v -> v < 0, 0)} clamps negatives to zero.
     *
     * @param test  decides, per element, whether to overwrite it
     * @param value the replacement value
     * @return this array
     * @throws IllegalArgumentException if {@code test} is null
     */
    public NumJArray conditionalPut(DoublePredicate test, double value) {
        if (test == null) {
            throw new IllegalArgumentException("test must not be null");
        }
        for (int p : positions()) {
            if (test.test(data[p])) data[p] = value;
        }
        return this;
    }

    /**
     * Builds a mask from a test on each element.
     *
     * @param test the condition to check
     * @return a new array of 1.0 where the test is true, otherwise 0.0
     * @throws IllegalArgumentException if {@code test} is null
     */
    public NumJArray where(DoublePredicate test) {
        if (test == null) {
            throw new IllegalArgumentException("test must not be null");
        }
        return map(x -> flag(test.test(x)));
    }

    /**
     * Picks values from {@code x} where {@code cond} is non-zero and from {@code y} elsewhere
     * (NumPys three-argument {@code where}). All three arrays are broadcast together.
     *
     * @param cond non-zero = true
     * @param x    values used where {@code cond} is true
     * @param y    values used where {@code cond} is false
     * @return a new array of the broadcast shape
     * @throws IllegalArgumentException if an argument is null or the shapes cannot be broadcast
     */
    public static NumJArray where(NumJArray cond, NumJArray x, NumJArray y) {
        if (cond == null || x == null || y == null) {
            throw new IllegalArgumentException("cond, x and y must not be null");
        }
        int[] outShape = broadcastShape(broadcastShape(cond.shape, x.shape), y.shape);
        double[] c = cond.broadcastTo(outShape).toContiguousData();
        double[] xs = x.broadcastTo(outShape).toContiguousData();
        double[] ys = y.broadcastTo(outShape).toContiguousData();

        double[] out = new double[c.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (c[i] != 0) ? xs[i] : ys[i];
        }
        return new NumJArray(out, outShape);
    }

    /**
     * Picks one of two scalars per element: {@code x} where {@code cond} is non-zero, otherwise {@code y}.
     *
     * @param cond non-zero = true
     * @param x    value used where {@code cond} is true
     * @param y    value used where {@code cond} is false
     * @return a new array with the shape of {@code cond}
     * @throws IllegalArgumentException if {@code cond} is null
     */
    public static NumJArray where(NumJArray cond, double x, double y) {
        if (cond == null) {
            throw new IllegalArgumentException("cond must not be null");
        }
        return cond.map(c -> c != 0 ? x : y);
    }

    /**
     * Maps a position in the flattened row-major order to a slot in {@link #data}.
     * On a contiguous array this is {@code offset + flatIndex}; on a transposed or sliced view it is not.
     *
     * @param flatIndex the logical position (0 to size-1); negative values count from the end
     * @return the index into {@code data}
     * @throws IndexOutOfBoundsException if {@code flatIndex} is outside the array
     */
    public int pointer(int flatIndex) {
        int idx = flatIndex < 0 ? flatIndex + size : flatIndex;
        if (idx < 0 || idx >= size) {
            throw new IndexOutOfBoundsException("Flat index " + flatIndex + " is out of bounds for size " + size);
        }
        int pos = offset;
        for (int d = shape.length - 1; d >= 0; d--) {
            pos += (idx % shape[d]) * strides[d];
            idx /= shape[d];
        }
        return pos;
    }

    /**
     * Reads an element by its position in the flattened row-major order.
     *
     * @param flatIndex the logical position; negative values count from the end
     * @return the value
     * @throws IndexOutOfBoundsException if {@code flatIndex} is outside the array
     */
    public double getFlat(int flatIndex) {
        return data[pointer(flatIndex)];
    }

    /**
     * Writes an element by its position in the flattened row-major order.
     *
     * @param value     the value to store
     * @param flatIndex the logical position; negative values count from the end
     * @throws IndexOutOfBoundsException if {@code flatIndex} is outside the array
     */
    public void setFlat(double value, int flatIndex) {
        data[pointer(flatIndex)] = value;
    }

    /**
     * Checks whether this arrays shape can be broadcast with anothers (NumPy rules: shapes are
     * aligned from the right, and each pair of sizes must match or one of them must be 1).
     *
     * @param other the other array
     * @return {@code true} if element-wise operations between the two would work
     */
    public boolean canBroadcast(NumJArray other) {
        if (other == null) return false;
        try {
            broadcastShape(shape, other.shape);
            return true;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    /**
     * Stretches the array to a larger shape without copying. Returns a read-mostly view:
     * repeated positions share one slot, so write to a {@link #copy()} instead.
     *
     * @param target the shape to broadcast to; leading dimensions may be added and size-1 dimensions stretched
     * @return a view with the target shape
     * @throws IllegalArgumentException if this shape cannot be broadcast to {@code target}
     */
    public NumJArray broadcastTo(int... target) {
        checkShape(target);
        if (target.length < shape.length) {
            throw new IllegalArgumentException(
                "Cannot broadcast " + Arrays.toString(shape) + " to " + Arrays.toString(target));
        }

        int lead = target.length - shape.length;
        int[] newStrides = new int[target.length];

        for (int i = 0; i < shape.length; i++) {
            if (shape[i] == target[i + lead]) {
                newStrides[i + lead] = strides[i];
            } else if (shape[i] == 1) {
                newStrides[i + lead] = 0;
            } else {
                throw new IllegalArgumentException(
                    "Cannot broadcast " + Arrays.toString(shape) + " to " + Arrays.toString(target));
            }
        }
        return new NumJArray(data, target, newStrides, offset);
    }

    /**
     * Reads an element as if this array had been broadcast to a larger shape: leading coordinates are
     * ignored, and coordinates along size-1 dimensions are ignored.
     *
     * @param coordinates coordinates in the broadcast space; at least {@link #ndim()} of them
     *                    (extra leading ones are skipped)
     * @return the value at the matching position of this array
     * @throws IllegalArgumentException  if fewer coordinates than dimensions are given
     * @throws IndexOutOfBoundsException if a coordinate is outside a non-singleton dimension, or the array is empty
     */
    public double broadcastGet(int... coordinates) {
        if (coordinates == null || coordinates.length < shape.length) {
            throw new IllegalArgumentException(
                "Expected at least " + shape.length + " coordinates, got " + (coordinates == null ? 0 : coordinates.length));
        }
        if (size == 0) {
            throw new IndexOutOfBoundsException("Cannot read from an empty array");
        }

        int lead = coordinates.length - shape.length;
        int pos = offset;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] == 1) continue;
            int c = coordinates[i + lead];
            if (c < 0) c += shape[i];
            if (c < 0 || c >= shape[i]) {
                throw new IndexOutOfBoundsException(
                    "Coordinate " + coordinates[i + lead] + " is out of bounds for axis " + i + " with size " + shape[i]);
            }
            pos += c * strides[i];
        }
        return data[pos];
    }

    /**
     * Two arrays are equal when they have the same shape and the same values in the same logical order,
     * whether or not one of them is a view. (NaN equals NaN here so that the equals/hashCode contract holds;
     * use {@link #allClose(NumJArray)} for tolerance-based comparison.)
     * The array is mutable, so do not change it while it is a key in a hash map.
     *
     * @param o the object to compare with
     * @return {@code true} if {@code o} is a {@code NumJArray} with identical shape and values
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NumJArray)) return false;
        NumJArray other = (NumJArray) o;
        return Arrays.equals(shape, other.shape)
            && Arrays.equals(toContiguousData(), other.toContiguousData());
    }

    /**
     * Hash code consistent with {@link #equals(Object)}: based on the shape and the logical values.
     *
     * @return the hash code
     */
    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(shape) + Arrays.hashCode(toContiguousData());
    }

    /**
     * Multi-line, NumPy-style printout with right-aligned columns, for arrays of any rank.
     * Example for a 2x3 array: {@code [[1.0, 2.0, 3.0],} newline {@code  [4.0, 5.0, 6.0]]}.
     *
     * @return the formatted text (just the number for a 0-D array, {@code []} for an empty one)
     */
    public String toPrettyString() {
        if (shape.length == 0) return String.valueOf(data[offset]);
        if (size == 0) return "[]";

        double[] flat = toContiguousData();
        String[] cells = new String[flat.length];
        int width = 0;
        for (int i = 0; i < flat.length; i++) {
            cells[i] = String.valueOf(flat[i]);
            width = Math.max(width, cells[i].length());
        }

        StringBuilder sb = new StringBuilder();
        appendPretty(sb, cells, width, 0, 0, 0);
        return sb.toString();
    }

    // HELPERS

    private static void requireOperator(Object op) {
        if (op == null) {
            throw new IllegalArgumentException("operator must not be null");
        }
    }

    private static int[] concatInts(int[] head, int... tail) {
        int[] out = new int[head.length + tail.length];
        System.arraycopy(head, 0, out, 0, head.length);
        System.arraycopy(tail, 0, out, head.length, tail.length);
        return out;
    }

    /** Slots in {@code data} of every element, in logical row-major order. */
    private int[] positions() {
        int[] out = new int[size];
        if (size == 0) return out;

        int nd = shape.length;
        int[] idx = new int[nd];
        int pos = offset;

        for (int i = 0; i < size; i++) {
            out[i] = pos;
            for (int d = nd - 1; d >= 0; d--) {
                idx[d]++;
                pos += strides[d];
                if (idx[d] < shape[d]) break;
                pos -= strides[d] * shape[d];
                idx[d] = 0;
            }
        }
        return out;
    }

    /** Applies a single-axis reduction once per listed axis, highest axis first so indices stay valid. */
    private NumJArray reduceAxes(int[] axes, BiFunction<NumJArray, Integer, NumJArray> single) {
        if (axes == null) {
            throw new IllegalArgumentException("axes must not be null");
        }
        int[] sorted = new int[axes.length];
        for (int i = 0; i < axes.length; i++) sorted[i] = normalizeAxis(axes[i]);
        Arrays.sort(sorted);
        for (int i = 1; i < sorted.length; i++) {
            if (sorted[i] == sorted[i - 1]) {
                throw new IllegalArgumentException("Axis " + sorted[i] + " is listed more than once");
            }
        }
        if (sorted.length == 0) return copy();

        NumJArray result = this;
        for (int i = sorted.length - 1; i >= 0; i--) {
            result = single.apply(result, sorted[i]);
        }
        return result;
    }

    private void appendPretty(StringBuilder sb, String[] cells, int width, int dim, int start, int indent) {
        int nd = shape.length;
        sb.append('[');

        if (dim == nd - 1) {
            for (int i = 0; i < shape[dim]; i++) {
                if (i > 0) sb.append(", ");
                String cell = cells[start + i];
                for (int pad = cell.length(); pad < width; pad++) sb.append(' ');
                sb.append(cell);
            }
        } else {
            int block = 1;
            for (int d = dim + 1; d < nd; d++) block *= shape[d];

            for (int i = 0; i < shape[dim]; i++) {
                if (i > 0) {
                    sb.append(',');
                    for (int n = 0; n < nd - dim - 1; n++) sb.append('\n');
                    for (int n = 0; n <= indent; n++) sb.append(' ');
                }
                appendPretty(sb, cells, width, dim + 1, start + i * block, indent + 1);
            }
        }
        sb.append(']');
    }

    /**
     * Describes the array with its shape and nested values, read through the strides
     * (so views print in their logical order).
     *
     * @return a readable string such as {@code shape=[2, 2], data=[[1.0, 2.0], [3.0, 4.0]]}
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("shape=").append(Arrays.toString(shape)).append(", data=");
        appendNested(sb, 0, offset);
        return sb.toString();
    }

    //HELPERS

    private static int[] createStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int running = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = running;
            running *= shape[i];
        }
        return strides;
    }

    private static int[] checkShape(int[] shape) {
        if (shape == null) {
            throw new IllegalArgumentException("shape must not be null");
        }
        for (int dim : shape) {
            if (dim < 0) {
                throw new IllegalArgumentException("Dimensions must be non-negative, got " + Arrays.toString(shape));
            }
        }
        return shape;
    }

    private static int sizeOf(int[] shape) {
        try {
            int total = 1;
            for (int dim : shape) total = Math.multiplyExact(total, dim);
            return total;
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("Shape is too large: " + Arrays.toString(shape));
        }
    }

    private static double[] requireLength(double[] data, int[] shape) {
        if (data == null) {
            throw new IllegalArgumentException("data must not be null");
        }
        int expected = sizeOf(checkShape(shape));
        if (data.length != expected) {
            throw new IllegalArgumentException(
                "data has " + data.length + " elements but shape " + Arrays.toString(shape) + " needs " + expected);
        }
        return data;
    }

    private static int normalizeAxis(int axis, int rank) {
        int ax = axis < 0 ? axis + rank : axis;
        if (ax < 0 || ax >= rank) {
            throw new IllegalArgumentException(
                "Axis " + axis + " is out of bounds for an array with " + rank + " dimension(s)");
        }
        return ax;
    }

    private int normalizeAxis(int axis) {
        return normalizeAxis(axis, shape.length);
    }

    private static long clamp(long v, long lo, long hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static int[] removeAt(int[] arr, int index) {
        int[] out = new int[arr.length - 1];
        for (int i = 0, j = 0; i < arr.length; i++) {
            if (i != index) out[j++] = arr[i];
        }
        return out;
    }

    private void requireNonEmpty(String op) {
        if (size == 0) {
            throw new IllegalStateException(op + "() is not defined for an empty array");
        }
    }

    private int getFlatIndex(int[] indices) {
        if (indices == null || indices.length != shape.length) {
            throw new IllegalArgumentException(
                "Dimension mismatch. Expected " + shape.length + " indices, got " + (indices == null ? 0 : indices.length));
        }
        int pos = offset;
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i] < 0 ? indices[i] + shape[i] : indices[i];
            if (idx < 0 || idx >= shape[i]) {
                throw new IndexOutOfBoundsException(
                    "Index " + indices[i] + " is out of bounds for axis " + i + " with size " + shape[i]);
            }
            pos += idx * strides[i];
        }
        return pos;
    }

    /** True if the elements sit back-to-back in row-major order starting at {@code offset}. */
    private boolean isContiguous() {
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (shape[i] != 1 && strides[i] != expected) return false;
            expected *= shape[i];
        }
        return true;
    }

    /** Always returns a fresh row-major copy of the logical values. Safe to modify. */
    private double[] toContiguousData() {
        double[] out = new double[size];
        if (size == 0) return out;

        if (isContiguous()) {
            System.arraycopy(data, offset, out, 0, size);
            return out;
        }

        int nd = shape.length;
        int[] idx = new int[nd];
        int pos = offset;

        for (int i = 0; i < size; i++) {
            out[i] = data[pos];
            for (int d = nd - 1; d >= 0; d--) {
                idx[d]++;
                pos += strides[d];
                if (idx[d] < shape[d]) break;
                pos -= strides[d] * shape[d];
                idx[d] = 0;
            }
        }
        return out;
    }

    /** Row-major logical values; may be the backing array itself, so never modify the result. */
    private double[] readOnlyData() {
        if (offset == 0 && data.length == size && isContiguous()) return data;
        return toContiguousData();
    }

    private int argFlat(boolean findMax) {
        requireNonEmpty(findMax ? "argMax" : "argMin");
        double[] d = readOnlyData();
        int best = 0;
        for (int i = 1; i < d.length; i++) {
            if (findMax ? d[i] > d[best] : d[i] < d[best]) best = i;
        }
        return best;
    }

    private int[] unravel(int flat) {
        int[] idx = new int[shape.length];
        for (int i = shape.length - 1; i >= 0; i--) {
            idx[i] = flat % shape[i];
            flat /= shape[i];
        }
        return idx;
    }

    /**
     * Start offsets (into {@code data}) of every 1-D "lane" running along {@code axis},
     * ordered row-major over the remaining dimensions.
     */
    private int[] laneOffsets(int axis) {
        int nd = shape.length;
        int lanes = 1;
        for (int i = 0; i < nd; i++) if (i != axis) lanes *= shape[i];

        int[] out = new int[lanes];
        if (lanes == 0) return out;

        int[] idx = new int[nd];
        int pos = offset;

        for (int l = 0; l < lanes; l++) {
            out[l] = pos;
            for (int d = nd - 1; d >= 0; d--) {
                if (d == axis) continue;
                idx[d]++;
                pos += strides[d];
                if (idx[d] < shape[d]) break;
                pos -= strides[d] * shape[d];
                idx[d] = 0;
            }
        }
        return out;
    }

    /** Folds each lane along {@code axis} with {@code op}; {@code init == null} starts from the first element. */
    private NumJArray reduceLanes(int axis, DoubleBinaryOperator op, Double init) {
        int ax = normalizeAxis(axis);
        int len = shape[ax];
        int st = strides[ax];

        if (init == null && len == 0) {
            throw new IllegalArgumentException("Cannot reduce over axis " + ax + " because it has length 0");
        }

        int[] lanes = laneOffsets(ax);
        double[] out = new double[lanes.length];

        for (int l = 0; l < lanes.length; l++) {
            int base = lanes[l];
            double acc;
            int k;
            if (init == null) {
                acc = data[base];
                k = 1;
            } else {
                acc = init;
                k = 0;
            }
            for (; k < len; k++) {
                acc = op.applyAsDouble(acc, data[base + k * st]);
            }
            out[l] = acc;
        }
        return new NumJArray(out, removeAt(shape, ax));
    }

    private NumJArray argReduceAxis(int axis, boolean findMax) {
        int ax = normalizeAxis(axis);
        int len = shape[ax];
        int st = strides[ax];

        if (len == 0) {
            throw new IllegalArgumentException("Cannot search axis " + ax + " because it has length 0");
        }

        int[] lanes = laneOffsets(ax);
        double[] out = new double[lanes.length];

        for (int l = 0; l < lanes.length; l++) {
            int base = lanes[l];
            int best = 0;
            for (int k = 1; k < len; k++) {
                double cur = data[base + k * st];
                double bestVal = data[base + best * st];
                if (findMax ? cur > bestVal : cur < bestVal) best = k;
            }
            out[l] = best;
        }
        return new NumJArray(out, removeAt(shape, ax));
    }

    /** View of {@code [start, stop)} along one axis, keeping all other axes whole. */
    private NumJArray sliceAxis(int ax, int start, int stop) {
        int[] newShape = shape.clone();
        newShape[ax] = stop - start;
        int newOffset = (stop > start) ? offset + start * strides[ax] : offset;
        return new NumJArray(data, newShape, strides, newOffset);
    }

    private static int[] broadcastShape(int[] a, int[] b) {
        int n = Math.max(a.length, b.length);
        int[] out = new int[n];

        for (int i = 0; i < n; i++) {
            int dimA = (i < a.length) ? a[a.length - 1 - i] : 1;
            int dimB = (i < b.length) ? b[b.length - 1 - i] : 1;

            if (dimA == dimB || dimB == 1) {
                out[n - 1 - i] = dimA;
            } else if (dimA == 1) {
                out[n - 1 - i] = dimB;
            } else {
                throw new IllegalArgumentException(
                    "Shapes are not broadcastable: " + Arrays.toString(a) + " and " + Arrays.toString(b));
            }
        }
        return out;
    }

    /** Strides that make an array of {@code inShape} look like {@code outShape} (stride 0 where it repeats). */
    private static int[] broadcastStrides(int[] inShape, int[] inStrides, int[] outShape) {
        int[] out = new int[outShape.length];
        int lead = outShape.length - inShape.length;
        for (int i = 0; i < inShape.length; i++) {
            out[i + lead] = (inShape[i] == 1) ? 0 : inStrides[i];
        }
        return out;
    }

    private NumJArray broadcastOp(NumJArray b, DoubleBinaryOperator op) {
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }

        int[] outShape = broadcastShape(shape, b.shape);
        int n = sizeOf(outShape);
        double[] result = new double[n];
        if (n == 0) return new NumJArray(result, outShape);

        int[] sa = broadcastStrides(shape, strides, outShape);
        int[] sb = broadcastStrides(b.shape, b.strides, outShape);

        int nd = outShape.length;
        int[] idx = new int[nd];
        int pa = offset;
        int pb = b.offset;

        for (int i = 0; i < n; i++) {
            result[i] = op.applyAsDouble(data[pa], b.data[pb]);
            for (int d = nd - 1; d >= 0; d--) {
                idx[d]++;
                pa += sa[d];
                pb += sb[d];
                if (idx[d] < outShape[d]) break;
                pa -= sa[d] * outShape[d];
                pb -= sb[d] * outShape[d];
                idx[d] = 0;
            }
        }
        return new NumJArray(result, outShape);
    }

    private void appendNested(StringBuilder sb, int dim, int pos) {
        if (dim == shape.length) {
            sb.append(data[pos]);
            return;
        }
        sb.append('[');
        for (int i = 0; i < shape[dim]; i++) {
            if (i > 0) sb.append(", ");
            appendNested(sb, dim + 1, pos + i * strides[dim]);
        }
        sb.append(']');
    }

    /**
     * Saves this array in NumJ's binary format.
     *
     * @param filePath the destination .nj file
     */
    public void save(String filePath) {
        if (filePath == null ||
                !filePath.toLowerCase().endsWith(".nj")) {
            throw new IllegalArgumentException(
                "File type must be .nj"
            );
        }

        NumJArray array = this.copy();

        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(
                    new FileOutputStream(filePath)))) {

            out.writeInt(0x4E554D4A);
            out.writeInt(1);
            out.writeInt(array.shape.length);

            for (int dimension : array.shape) {
                out.writeInt(dimension);
            }

            for (double value : array.data) {
                out.writeDouble(value);
            }

        } catch (IOException e) {
            throw new UncheckedIOException(
                "Failed to save NumJ array: " + filePath, e
            );
        }
    }

    /**
     * Loads an array from NumJ's binary format.
     *
     * @param filePath the source .nj file
     * @return the loaded array
     */
    public static NumJArray load(String filePath) {
        if (filePath == null ||!filePath.toLowerCase().endsWith(".nj")) {
            throw new IllegalArgumentException("File type must be .nj");
        }

        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {

            if (in.readInt() != 0x4E554D4A) {
                throw new IOException("Invalid NumJ file");
            }

            if (in.readInt() != 1) {
                throw new IOException("Unsupported NumJ file version");
            }

            int ndim = in.readInt();

            if (ndim < 0 || ndim > 32) {
                throw new IOException("Invalid number of dimensions: " + ndim);
            }

            int[] shape = new int[ndim];

            for (int i = 0; i < ndim; i++) {
                shape[i] = in.readInt();

                if (shape[i] < 0) {
                    throw new IOException("Negative dimension");
                }
            }

            int size;

            try {
                size = sizeOf(shape);
            } catch (RuntimeException e) {
                throw new IOException("Invalid array shape", e);
            }

            double[] data = new double[size];

            for (int i = 0; i < size; i++) {
                data[i] = in.readDouble();
            }

            return new NumJArray(data, shape);

        } catch (IOException e) {
            throw new UncheckedIOException("Failed to load NumJ array: " + filePath, e);
        }
    }

    /**
     * Saves this array in NumPy's .npy format.
     *
     * // SHOULD WORK, BUT HAVEN'T FULLY TESTED.
     *
     * @param filePath the destination .npy file
     */
    public void saveNpy(String filePath) {
        // SHOULD WORK, BUT HAVEN'T FULLY TESTED.
        if (filePath == null ||!filePath.toLowerCase().endsWith(".npy")) {
            throw new IllegalArgumentException("File type must be .npy");
        }

        NumJArray array = this.copy();

        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(filePath))) {

            out.write(new byte[] {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0});

            StringBuilder shapeString = new StringBuilder("(");

            for (int i = 0; i < array.shape.length; i++) {
                if (i > 0) shapeString.append(", ");
                shapeString.append(array.shape[i]);
            }

            if (array.shape.length == 1) {
                shapeString.append(",");
            }

            shapeString.append(")");

            String header ="{'descr': '<f8', 'fortran_order': False, 'shape': "+ shapeString + ", }";
            int padding =(16 - ((10 + header.length() + 1) % 16)) % 16;
            String paddedHeader = header + " ".repeat(padding) + "\n";
            byte[] headerBytes = paddedHeader.getBytes(StandardCharsets.US_ASCII);

            if (headerBytes.length > 65535) {
                throw new IOException("NPY header is too large");
            }

            out.write(headerBytes.length & 0xFF);
            out.write((headerBytes.length >>> 8) & 0xFF);
            out.write(headerBytes);

            ByteBuffer buffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);

            for (double value : array.data) {
                buffer.clear();
                buffer.putDouble(value);
                out.write(buffer.array());
            }

        } catch (IOException e) {
            throw new UncheckedIOException("Failed to save NPY array: " + filePath, e);
        }
    }

    /**
     * Loads a C-order float64 array from a NumPy .npy file.
     *
     * // SHOULD WORK, BUT HAVEN'T FULLY TESTED.
     *
     * @param filePath the source .npy file
     * @return the loaded array
     */
    public static NumJArray loadNpy(String filePath) {
        // SHOULD WORK, BUT HAVEN'T FULLY TESTED.
        if (filePath == null || !filePath.toLowerCase().endsWith(".npy")) {
            throw new IllegalArgumentException("File type must be .npy"
            );
        }

        try (InputStream in = new BufferedInputStream(new FileInputStream(filePath))) {

            byte[] magic = in.readNBytes(6);

            if (magic.length != 6 || magic[0] != (byte) 0x93 || magic[1] != 'N' || magic[2] != 'U' || magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
                throw new IOException("Invalid NPY file");
            }

            int major = in.read();
            int minor = in.read();

            if (major < 1 || major > 3 || minor < 0) {
                throw new IOException("Unsupported NPY version");
            }

            int headerLength;

            if (major == 1) {
                int lo = in.read();
                int hi = in.read();

                if (lo < 0 || hi < 0) {
                    throw new EOFException("Incomplete NPY header");
                }

                headerLength = lo | (hi << 8);

            } else {
                byte[] lengthBytes = in.readNBytes(4);

                if (lengthBytes.length != 4) {
                    throw new EOFException("Incomplete NPY header");
                }

                headerLength = ByteBuffer.wrap(lengthBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();

                if (headerLength < 0) {
                    throw new IOException("Invalid NPY header length");
                }
            }

            if (headerLength <= 0 || headerLength > 1_000_000) {
                throw new IOException("Invalid NPY header length");
            }

            byte[] headerBytes = in.readNBytes(headerLength);

            if (headerBytes.length != headerLength) {
                throw new EOFException("Incomplete NPY header");
            }

            String header = new String(headerBytes,major == 3 ? StandardCharsets.UTF_8 : StandardCharsets.US_ASCII);

            Matcher dtypeMatcher = Pattern.compile("['\"]descr['\"]\\s*:\\s*['\"]([<>=|][fiu]\\d+)['\"]").matcher(header);

            if (!dtypeMatcher.find()) {
                throw new IOException("Missing or unsupported NPY dtype");
            }

            String dtype = dtypeMatcher.group(1);

            if (!dtype.matches("[<>=|]f8")) {
                throw new IOException("Only float64 NPY arrays are supported: " + dtype);
            }

            Matcher fortranMatcher = Pattern.compile("['\"]fortran_order['\"]\\s*:\\s*(True|False)").matcher(header);

            if (!fortranMatcher.find() ||fortranMatcher.group(1).equals("True")) {
                throw new IOException("Only C-order NPY arrays are supported");
            }

            Matcher shapeMatcher = Pattern.compile( "['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)").matcher(header);

            if (!shapeMatcher.find()) {
                throw new IOException("Missing NPY shape");
            }

            String shapeText = shapeMatcher.group(1).trim();
            int[] shape;

            if (shapeText.isEmpty()) {
                shape = new int[0];
            } else {
                String[] dimensions = shapeText.split(",");
                java.util.ArrayList<Integer> dims = new java.util.ArrayList<>();

                for (String dimension : dimensions) {
                    dimension = dimension.trim();

                    if (!dimension.isEmpty()) {
                        dims.add(Integer.parseInt(dimension));
                    }
                }

                shape = new int[dims.size()];

                for (int i = 0; i < shape.length; i++) {
                    shape[i] = dims.get(i);

                    if (shape[i] < 0) {
                        throw new IOException("Negative NPY dimension");
                    }
                }
            }

            int size;

            try {
                size = sizeOf(shape);
            } catch (RuntimeException e) {
                throw new IOException("Invalid NPY shape", e);
            }

            ByteOrder order = dtype.charAt(0) == '>' ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN;

            double[] data = new double[size];
            byte[] valueBytes = new byte[8];
            ByteBuffer valueBuffer = ByteBuffer.wrap(valueBytes).order(order);

            for (int i = 0; i < size; i++) {
                int read = 0;

                while (read < 8) {
                    int count = in.read(valueBytes, read, 8 - read);

                    if (count < 0) {
                        throw new EOFException("Incomplete NPY data");
                    }

                    read += count;
                }

                valueBuffer.position(0);
                data[i] = valueBuffer.getDouble();
            }

            return new NumJArray(data, shape);

        } catch (IOException e) {
            throw new UncheckedIOException("Failed to load NPY array: " + filePath, e);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid NPY shape: " + filePath, e);
        }
    }
}