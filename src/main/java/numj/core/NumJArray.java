package numj.core;
import java.util.Arrays;
import java.util.function.BiFunction;

import numj.io.njio;
import numj.io.npio;
import numj.math.FloatBinaryOperator;
import numj.math.FloatPredicate;
import numj.math.FloatUnaryOperator;
import numj.math.arithmatic;
import numj.math.lina;
import numj.math.sorts;
import numj.math.stat;
import numj.math.trig;
import numj.ml.activation;


/**
 * A small, NumPy-style N-dimensional array of {@code float}s.
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
    public final float[] data;

    /** Size of each dimension. Do not modify; use {@link #shape()} for a safe copy. */
    public final int[] shape;

    /** Step (in {@code data} slots) to move one index along each dimension. */
    public final int[] strides;

    /** Index in {@code data} of the element at coordinates {@code [0, 0, ...]}. */
    public final int offset;

    public final int size;

    /**
     * Creates a contiguous (row-major) array that wraps {@code data} without copying it.
     *
     * @param data  the values in row-major order; its length must equal the product of {@code shape}
     * @param shape the size of each dimension (an empty shape creates a 0-D scalar array)
     * @throws IllegalArgumentException if {@code data} or {@code shape} is null, a dimension is
     *                                  negative, or {@code data.length} does not match the shape
     */
    public NumJArray(float[] data, int[] shape) {
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
    public NumJArray(float[] data, int[] shape, int[] strides) {
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
    public NumJArray(float[] data, int[] shape, int[] strides, int offset) {
        if (data == null) {
            throw new IllegalArgumentException("data must not be null");
        }
        Validator.checkShape(shape);
        if (strides == null || strides.length != shape.length) {
            throw new IllegalArgumentException("strides must have the same length as shape");
        }

        this.data = data;
        this.shape = shape.clone();
        this.strides = strides.clone();
        this.offset = offset;
        this.size = Validator.sizeOf(shape);

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
    public float item() {
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
    public float get(int... indices) {
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
    public void set(float value, int... indices) {
        data[getFlatIndex(indices)] = value;
    }

    /**
     * Creates an array of zeros.
     *
     * @param shape the size of each dimension
     * @return a new contiguous array filled with {@code 0.0}
     * @throws IllegalArgumentException if the shape is invalid
     */
    public static NumJArray zeros(int... shape) {
        return creation.zeros(shape);
    }

    /**
     * Creates an array of ones.
     *
     * @param shape the size of each dimension
     * @return a new contiguous array filled with {@code 1.0}
     * @throws IllegalArgumentException if the shape is invalid
     */
    public static NumJArray ones(int... shape) {
        return creation.ones(shape);
    }

    /**
     * Creates an array filled with one value.
     *
     * @param value the value for every element
     * @param shape the size of each dimension
     * @return a new contiguous array filled with {@code value}
     * @throws IllegalArgumentException if the shape is invalid
     */
    public static NumJArray full(float value, int... shape) {
        return creation.full(value, shape);
    }

    /**
     * Creates a 1-D array from the given values (the values are copied).
     *
     * @param values the elements
     * @return a new 1-D array
     * @throws IllegalArgumentException if {@code values} is null
     */
    public static NumJArray of(float... values) {
        return creation.of(values);
    }

    /**
     * Creates a 2-D array from a rectangular {@code float[][]} (the values are copied).
     *
     * @param rows the rows; every row must have the same length
     * @return a new 2-D array of shape {@code [rows.length, rows[0].length]}
     * @throws IllegalArgumentException if {@code rows} is null, or the rows are ragged
     */
    public static NumJArray of(float[][] rows) {
        return creation.of(rows);
    }

    /**
     * Creates a 1-D array counting up from 0 to {@code stop} in steps of 1, like Pythons {@code range(stop)}.
     * For example {@code arange(5)} gives {@code [0, 1, 2, 3, 4]}.
     *
     * @param stop the end of the range (exclusive)
     * @return a new 1-D array (empty if {@code stop <= 0})
     */
    public static NumJArray arange(float stop) {
        return creation.arange(0, stop, 1);
    }

    /**
     * Creates a 1-D array counting up from {@code start} to {@code stop} in steps of 1.
     * For example {@code arange(1, 6)} gives {@code [1, 2, 3, 4, 5]}.
     *
     * @param start the first value
     * @param stop  the end of the range (exclusive)
     * @return a new 1-D array (empty if {@code stop <= start})
     */
    public static NumJArray arange(float start, float stop) {
        return creation.arange(start, stop, 1);
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
    public static NumJArray arange(float start, float stop, float step) {
        return creation.arange(start, stop, step);
    }

    /**
     * Creates a 1-D array of {@code num} evenly spaced values from {@code start} to {@code stop},
     * including {@code stop} (unlike {@link #arange(float, float, float)}).
     * For example {@code linspace(1, 4, 4)} gives {@code [1, 2, 3, 4]}.
     *
     * @param start the first value
     * @param stop  the last value (inclusive)
     * @param num   how many values to generate; must be non-negative ({@code 1} gives just {@code [start]})
     * @return a new 1-D array of length {@code num}
     * @throws IllegalArgumentException if {@code num} is negative
     */
    public static NumJArray linspace(float start, float stop, int num) {
        return creation.linspace(start, stop, num);
    }

    /**
     * Creates an {@code n x n} identity matrix.
     *
     * @param n the number of rows and columns; must be non-negative
     * @return a new contiguous 2-D array with ones on the diagonal
     * @throws IllegalArgumentException if {@code n} is negative
     */
    public static NumJArray identity(int n) {
        return creation.identity(n);
    }

    /**
     * Creates an array of normally distributed random numbers.
     *
     * @param mean   the mean of the distribution
     * @param stdDev the standard deviation; must be non-negative
     * @param shape  the size of each
     * @return a new contiguous array of random samples
     * @throws IllegalArgumentException if {@code stdDev} is negative or the shape is invalid
     */
    public static NumJArray normal(float mean, float stdDev, int... shape) {
        return creation.normal(mean, stdDev, shape);
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
    public static NumJArray uniform(float low, float high, int... shape) {
        return creation.uniform(low, high, shape);
    }

     /**
     * Creates an {@code n x n} identity matrix (same as {@link #identity(int)}).
     *
     * @param n the number of rows and columns
     * @return a new 2-D array with ones on the diagonal
     * @throws IllegalArgumentException if {@code n} is negative
     */
    public static NumJArray eye(int n) {
        return creation.eye(n, n, 0);
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
        return creation.eye(n, m, 0);
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
        return creation.eye(n, m, k);
    }

    /**
     * Element-wise absolute value.
     *
     * @return a new array of {@code |x|}
     */
    public NumJArray abs() {
        return arithmatic.abs(this);
    }

    /**
     * Element-wise negation.
     *
     * @return a new array of {@code -x}
     */
    public NumJArray neg() {
        return arithmatic.neg(this);
    }

    /**
     * Element-wise square root. Negative inputs give NaN.
     *
     * @return a new array of {@code sqrt(x)}
     */
    public NumJArray sqrt() {
        return arithmatic.sqrt(this);
    }

    /**
     * Element-wise power.
     *
     * @param power the exponent
     * @return a new array of {@code x^power}
     */
    public NumJArray pow(float power) {
        return arithmatic.pow(this, power);
    }

    /**
     * Element-wise {@code e^x}.
     *
     * @return a new array of {@code exp(x)}
     */
    public NumJArray exp() {
        return arithmatic.exp(this);
    }

    /**
     * Element-wise natural logarithm. Zero gives -Infinity, negatives give NaN.
     *
     * @return a new array of {@code ln(x)}
     */
    public NumJArray log() {
        return arithmatic.log(this);
    }

    /**
     * Element-wise floor.
     *
     * @return a new array with each value rounded down
     */
    public NumJArray floor() {
        return arithmatic.floor(this);
    }

    /**
     * Element-wise ceiling.
     *
     * @return a new array with each value rounded up
     */
    public NumJArray ceil() {
        return arithmatic.ceil(this);
    }

    /**
     * Element-wise rounding to the nearest integer, ties to even (same as NumPys {@code round}).
     *
     * @return a new array of rounded values
     */
    public NumJArray round() {
        return arithmatic.round(this);
    }

    /**
     * Element-wise sign.
     *
     * @return a new array containing -1, 0 or 1 (NaN stays NaN)
     */
    public NumJArray sign() {
        return arithmatic.sign(this);
    }

    /**
     * Element-wise sin.
     *
     * @return a new array of {@code sin(x)}
     */
    public NumJArray sin() {
        return trig.sin(this);
    }

    /**
     * Element-wise cos.
     *
     * @return a new array of {@code cos(x)}
     */
    public NumJArray cos() {
        return trig.cos(this);
    }

    /**
     * Element-wise tan.
     *
     * @return a new array of {@code tan(x)}
     */
    public NumJArray tan() {
        return trig.tan(this);
    }

    /**
     * Element-wise hyperbolic sine.
     *
     * @return a new array of {@code sinh(x)}
     */
    public NumJArray sinh() {
        return trig.sinh(this);
    }

    /**
     * Element-wise hyperbolic cosine.
     *
     * @return a new array of {@code cosh(x)}
     */
    public NumJArray cosh() {
        return trig.cosh(this);
    }

    /**
     * Element-wise hyperbolic tangent.
     *
     * @return a new array of {@code tanh(x)}
     */
    public NumJArray tanh() {
        return trig.tanh(this);
    }


    // Inverse trigonometric functions

    /**
     * Element-wise inverse sine.
     *
     * @return a new array of {@code asin(x)}, in radians
     */
    public NumJArray asin() {
        return trig.asin(this);
    }

    /**
     * Element-wise inverse cosine.
     *
     * @return a new array of {@code acos(x)}, in radians
     */
    public NumJArray acos() {
        return trig.acos(this);
    }

    /**
     * Element-wise inverse tangent.
     *
     * @return a new array of {@code atan(x)}, in radians
     */
    public NumJArray atan() {
        return trig.atan(this);
    }

    /**
     * Element-wise two-argument inverse tangent.
     * Computes {@code atan2(y, x)} using corresponding elements.
     *
     * @param x the x-coordinate array
     * @return a new array of angles in radians
     */
    public NumJArray atan2(NumJArray x) {
        return trig.atan2(this, x);
    }


    // In-place trigonometric functions

    /**
     * Applies sin to each element in place.
     *
     * @return this array after applying {@code sin(x)}
     */
    public NumJArray sini() {
        return trig.sini(this);
    }

    /**
     * Applies cos to each element in place.
     *
     * @return this array after applying {@code cos(x)}
     */
    public NumJArray cosi() {
        return trig.cosi(this);
    }

    /**
     * Applies tan to each element in place.
     *
     * @return this array after applying {@code tan(x)}
     */
    public NumJArray tani() {
        return trig.tani(this);
    }

    /**
     * Applies hyperbolic sine to each element in place.
     *
     * @return this array after applying {@code sinh(x)}
     */
    public NumJArray sinhi() {
        return trig.sinhi(this);
    }

    /**
     * Applies hyperbolic cosine to each element in place.
     *
     * @return this array after applying {@code cosh(x)}
     */
    public NumJArray coshi() {
        return trig.coshi(this);
    }

    /**
     * Applies hyperbolic tangent to each element in place.
     *
     * @return this array after applying {@code tanh(x)}
     */
    public NumJArray tanhi() {
        return trig.tanhi(this);
    }


    // In-place inverse trigonometric functions

    /**
     * Applies inverse sine to each element in place.
     *
     * @return this array after applying {@code asin(x)}
     */
    public NumJArray asini() {
        return trig.asini(this);
    }

    /**
     * Applies inverse cosine to each element in place.
     *
     * @return this array after applying {@code acos(x)}
     */
    public NumJArray acosi() {
        return trig.acosi(this);
    }

    /**
     * Applies inverse tangent to each element in place.
     *
     * @return this array after applying {@code atan(x)}
     */
    public NumJArray atani() {
        return trig.atani(this);
    }

    /**
     * Applies two-argument inverse tangent in place.
     * Computes {@code atan2(y, x)} and stores the results in this array.
     *
     * @param x the x-coordinate array
     * @return this array after applying {@code atan2(y, x)}
     */
    public NumJArray atan2i(NumJArray x) {
        return trig.atan2i(this, x);
    }

    /**
     * Limits every element to the range {@code [min, max]}.
     *
     * @param min the lower bound
     * @param max the upper bound; must be {@code >= min}
     * @return a new array with values clamped into the range
     * @throws IllegalArgumentException if {@code min > max}
     */
    public NumJArray clip(float min, float max) {
        return arithmatic.clip(this, min, max);
    }

    /**
     * Adds a scalar to every element.
     *
     * @param b the scalar
     * @return a new array of {@code x + b}
     */
    public NumJArray add(float b) {
        return arithmatic.add(this, b);
    }

    /**
     * Subtracts a scalar from every element.
     *
     * @param b the scalar
     * @return a new array of {@code x - b}
     */
    public NumJArray sub(float b) {
        return arithmatic.sub(this, b);
    }

    /**
     * Multiplies every element by a scalar.
     *
     * @param b the scalar
     * @return a new array of {@code x * b}
     */
    public NumJArray mul(float b) {
        return arithmatic.mul(this, b);
    }

    /**
     * Divides every element by a scalar.
     *
     * @param b the scalar (dividing by 0 gives Infinity/NaN, as with plain floats)
     * @return a new array of {@code x / b}
     */
    public NumJArray div(float b) {
        return arithmatic.div(this, b);
    }

    /**
     * Element-wise modulo. The result takes the sign of {@code b}, like NumPy and Python
     * (Javas {@code %} would take the sign of {@code x}).
     *
     * @param b the divisor
     * @return a new array of {@code x mod b}
     */
    public NumJArray mod(float b) {
        return arithmatic.mod(this, b);
    }

    /**
     * Element-wise addition with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code this + b}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray add(NumJArray b) {
        return arithmatic.add(this, b);
    }

    /**
     * Element-wise subtraction with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code this - b}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray sub(NumJArray b) {
        return arithmatic.sub(this, b);
    }

    /**
     * Element-wise multiplication with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code this * b}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray mul(NumJArray b) {
        return arithmatic.mul(this, b);
    }

    /**
     * Element-wise division with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code this / b}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray div(NumJArray b) {
        return arithmatic.div(this, b);
    }

    /**
     * Sum of all elements.
     *
     * @return the sum ({@code 0.0} for an empty array)
     */
    public float sum() {
        return stat.sum(this);
    }

    /**
     * Product of all elements.
     *
     * @return the product ({@code 1.0} for an empty array)
     */
    public float prod() {
        return stat.prod(this);
    }

    /**
     * Arithmetic mean of all elements.
     *
     * @return the mean ({@code NaN} for an empty array)
     */
    public float mean() {
        return stat.mean(this);
    }

    /**
     * Smallest element.
     *
     * @return the minimum value
     * @throws IllegalStateException if the array is empty
     */
    public float min() {
        return stat.min(this);
    }

    /**
     * Largest element.
     *
     * @return the maximum value
     * @throws IllegalStateException if the array is empty
     */
    public float max() {
        return stat.max(this);
    }

    /**
     * Population variance of all elements (divides by N, like NumPys default).
     *
     * @return the variance ({@code NaN} for an empty array)
     */
    public float var() {
        return stat.var(this);
    }

    /**
     * Population standard deviation of all elements.
     *
     * @return the square root of {@link #var()}
     */
    public float std() {
        return stat.std(this);
    }

    /**
     * Euclidean (L2) norm of all elements, i.e. {@code sqrt(sum(x^2))}.
     *
     * @return the norm
     */
    public float norm() {
        return lina.norm(this);
    }

    /**
     * Sum of the main diagonal of a 2-D array. For N-D arrays use {@link #trace(int, int)}.
     *
     * @return the trace
     * @throws IllegalArgumentException if the array is not 2-D
     */
    public float trace() {
        return lina.trace(this);
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
        return lina.trace(this, axis1, axis2);
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
        return lina.diagonal(this, axis1, axis2);
    }

    /**
     * Inner product of two arrays of the same shape: {@code sum(this * other)}.
     * (For NumPy-style {@code dot} on vectors, matrices and N-D arrays use {@link #dot(NumJArray)}.)
     *
     * @param other an array with exactly the same shape
     * @return the scalar inner product
     * @throws IllegalArgumentException if {@code other} is null or the shapes differ
     */
    public float inner(NumJArray other) {
        return lina.inner(this, other);
    }

    /**
     * Builds a diagonal matrix from a 1-D array, or extracts the main diagonal of a 2-D array.
     *
     * @return for 1-D input an {@code n x n} matrix; for 2-D input a new 1-D array (a copy)
     * @throws IllegalArgumentException if the array is not 1-D or 2-D
     */
    public NumJArray diag() {
        return lina.diag(this, 0);
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
        return lina.diag(this, k);
    }

    /**
     * Keeps the lower triangle (main diagonal and below) of every matrix in the last two axes, zeroing the rest.
     *
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public NumJArray tril() {
        return manipulation.tril(this);
    }

    /**
     * Keeps everything on or below a chosen diagonal of every matrix in the last two axes, zeroing the rest.
     *
     * @param k which diagonal: 0 is the main one, positive values include more above it, negative values exclude more
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public NumJArray tril(int k) {
        return manipulation.tril(this, k);
    }

    /**
     * Keeps the upper triangle (main diagonal and above) of every matrix in the last two axes, zeroing the rest.
     *
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public NumJArray triu() {
        return manipulation.triu(this);
    }

    /**
     * Keeps everything on or above a chosen diagonal of every matrix in the last two axes, zeroing the rest.
     *
     * @param k which diagonal: 0 is the main one, positive values exclude more below, negative values include more
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the array has fewer than 2 dimensions
     */
    public NumJArray triu(int k) {
        return manipulation.triu(this, k);
    }

    /**
     * Outer product of two arrays, each treated as a flat list: {@code result[i][j] = this[i] * b[j]}.
     *
     * @param b the other array (any shape; flattened)
     * @return a new 2-D array of shape {@code [this.size, b.size]}
     * @throws IllegalArgumentException if {@code b} is null
     */
    public NumJArray outer(NumJArray b) {
        return lina.outer(this, b);
    }

    public NumJArray triangle(int k, boolean lower) {
        return manipulation.triangle(this, k, lower);
    }

    /**
     * Finds the position of the largest element.
     *
     * @return the coordinates (one per dimension) of the first maximum
     * @throws IllegalStateException if the array is empty
     */
    public int[] argMax() {
        return stat.argMax(this);
    }

    /**
     * Finds the position of the smallest element.
     *
     * @return the coordinates (one per dimension) of the first minimum
     * @throws IllegalStateException if the array is empty
     */
    public int[] argMin() {
        return stat.argMin(this);
    }

    /**
     * Sums along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end ({@code -1} = last)
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray sumAxis(int axis) {
        return stat.sumAxis(this, axis);
    }

    /**
     * Multiplies along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray prodAxis(int axis) {
        return stat.prodAxis(this, axis);
    }

    /**
     * Averages along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray meanAxis(int axis) {
        return stat.meanAxis(this, axis);
    }

    /**
     * Minimum along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public NumJArray minAxis(int axis) {
        return stat.minAxis(this, axis);
    }

    /**
     * Maximum along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public NumJArray maxAxis(int axis) {
        return stat.maxAxis(this, axis);
    }

    /**
     * Population variance along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray varAxis(int axis) {
        return stat.varAxis(this, axis);
    }

    /**
     * Population standard deviation along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray stdAxis(int axis) {
        return stat.stdAxis(this, axis);
    }

    /**
     * Euclidean (L2) norm along an axis, e.g. the length of each row vector.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray normAxis(int axis) {
        return lina.normAxis(this, axis);
    }

    /**
     * Index of the largest value along an axis (first one wins ties).
     *
     * @param axis the axis to search; negative values count from the end
     * @return a new array (same shape minus {@code axis}) holding indices stored as floats
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public NumJArray argMaxAxis(int axis) {
        return stat.argMaxAxis(this, axis);
    }

    /**
     * Index of the smallest value along an axis (first one wins ties).
     *
     * @param axis the axis to search; negative values count from the end
     * @return a new array (same shape minus {@code axis}) holding indices stored as floats
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public NumJArray argMinAxis(int axis) {
        return stat.argMinAxis(this, axis);
    }

    /**
     * Running total along an axis.
     *
     * @param axis the axis to accumulate along; negative values count from the end
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray cumSumAxis(int axis) {
        return stat.cumSumAxis(this, axis);
    }

    /**
     * Logistic sigmoid of a single number, {@code 1 / (1 + e^-x)}, computed in a numerically stable way.
     *
     * @param x the input
     * @return a value in {@code (0, 1)}
     */
    public static float sig(float x) {
        if (x >= 0) {
            return (float) (1.0f / (1.0f + Math.exp(-x)));
        }
        float e = (float) Math.exp(x);
        return (float) (e / (1.0f + e));
    }

    /**
     * Hyperbolic tangent of a single number.
     *
     * @param x the input
     * @return a value in {@code (-1, 1)}
     */
    public static float tn(float x) {
        return (float) Math.tanh(x);
    }

    /**
     * ReLU activation: {@code max(0, x)} for every element.
     *
     * @return a new array of the same shape
     */
    public NumJArray relu() {
        return activation.relu(this);
    }

    /**
     * Derivative of ReLU: 1 where {@code x > 0}, otherwise 0.
     *
     * @return a new array of 0s and 1s (input is the pre-activation value)
     */
    public NumJArray reluD() {
        return activation.reluD(this);
    }

    /**
     * Sigmoid activation applied to every element.
     *
     * @return a new array of the same shape with values in {@code (0, 1)}
     */
    public NumJArray sigmoid() {
        return activation.sigmoid(this);
    }

    /**
     * Derivative of sigmoid, {@code s * (1 - s)} where {@code s = sigmoid(x)}.
     *
     * @return a new array (input is the pre-activation value, not the sigmoid output)
     */
    public NumJArray sigmoidD() {
        return activation.sigmoidD(this);
    }

    /**
     * Tanh activation applied to every element.
     *
     * @return a new array of the same shape with values in {@code (-1, 1)}
     */
    public NumJArray tanhActivation() {
        return activation.tanh(this);
    }

    /**
     * Derivative of tanh, {@code 1 - tanh(x)^2}.
     *
     * @return a new array (input is the pre-activation value, not the tanh output)
     */
    public NumJArray tanhD() {
        return activation.tanhD(this);
    }

    /**
     * Softmax over all elements (they sum to 1 afterwards). Stable against overflow.
     *
     * @return a new array of the same shape
     */
    public NumJArray softmax() {
        return activation.softmax(this);
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
        return activation.softmaxAxis(this, axis);
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
        return lina.dot(this, b);
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
        return lina.matMul(this, b);
    }

    /**
     * Reverses the order of all axes. Returns a view; no data is copied.
     *
     * @return a transposed view sharing data with this array
     */
    public NumJArray transpose() {
        return manipulation.transpose(this);
    }

    /**
     * Reorders the axes. Returns a view.
     *
     * @param axes a permutation of {@code 0..ndim-1}; {@code axes[i]} is the old axis that becomes axis {@code i}
     * @return a permuted view sharing data with this array
     * @throws IllegalArgumentException if {@code axes} is not a valid permutation
     */
    public NumJArray transpose(int... axes) {
        return manipulation.transpose(this, axes);
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
        return manipulation.swapAxes(this, axis1, axis2);
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
        return manipulation.reshape(this, newShape);
    }

    /**
     * Flattens to 1-D. Returns a view when possible, otherwise a copy (like NumPys {@code ravel}).
     *
     * @return a 1-D array with {@link #size()} elements
     */
    public NumJArray ravel() {
        return manipulation.reshape(this, size);
    }

    /**
     * Flattens to 1-D, always returning an independent copy.
     *
     * @return a new 1-D array with {@link #size()} elements
     */
    public NumJArray flatten() {
        return manipulation.reshape(this, size);
    }

    /**
     * Makes an independent, contiguous copy (views become normal arrays).
     *
     * @return a new array with the same shape and values
     */
    public NumJArray copy() {
        return manipulation.copy(this);
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
        return manipulation.expandDims(this, axis);
    }

    /**
     * Removes every dimension of size 1. Returns a view.
     *
     * @return a view without singleton dimensions
     */
    public NumJArray squeeze() {
        return manipulation.squeeze(this);
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
        return manipulation.select(this, axis, index);
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
        return manipulation.slice(this, slices);
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
        return manipulation.concat(this, b, axis);
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
        return manipulation.stack(this, b, axis);
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
        return manipulation.split(this, axis, sections);
    }

    /**
     * Element-wise {@code ==} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 (equal) and 0.0 (not equal); NaN is never equal to anything
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray equal(NumJArray b) {
        return arithmatic.equal(this, b);
    }

    /**
     * Element-wise {@code ==} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x == b}, otherwise 0.0
     */
    public NumJArray equal(float b) {
        return arithmatic.equal(this, b);
    }

    /**
     * Element-wise {@code !=} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 (different) and 0.0 (equal); NaN is always "not equal"
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray notEqual(NumJArray b) {
        return arithmatic.notEqual(this, b);
    }

    /**
     * Element-wise {@code !=} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x != b}, otherwise 0.0
     */
    public NumJArray notEqual(float b) {
        return arithmatic.notEqual(this, b);
    }

    /**
     * Element-wise {@code >} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where {@code this > b}, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray greater(NumJArray b) {
        return arithmatic.greater(this, b);
    }

    /**
     * Element-wise {@code >} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x > b}, otherwise 0.0
     */
    public NumJArray greater(float b) {
        return arithmatic.greater(this, b);
    }

    /**
     * Element-wise {@code >=} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where {@code this >= b}, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray greaterEqual(NumJArray b) {
        return arithmatic.greaterEqual(this, b);
    }

    /**
     * Element-wise {@code >=} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x >= b}, otherwise 0.0
     */
    public NumJArray greaterEqual(float b) {
        return arithmatic.greaterEqual(this, b);
    }

    /**
     * Element-wise {@code <} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where {@code this < b}, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray less(NumJArray b) {
        return arithmatic.less(this, b);
    }

    /**
     * Element-wise {@code <} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x < b}, otherwise 0.0
     */
    public NumJArray less(float b) {
        return arithmatic.less(this, b);
    }

    /**
     * Element-wise {@code <=} with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where {@code this <= b}, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray lessEqual(NumJArray b) {
        return arithmatic.lessEqual(this, b);
    }

    /**
     * Element-wise {@code <=} against a scalar.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where {@code x <= b}, otherwise 0.0
     */
    public NumJArray lessEqual(float b) {
        return arithmatic.lessEqual(this, b);
    }

    /**
     * Element-wise logical AND with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where both values are non-zero, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray logicalAnd(NumJArray b) {
        return arithmatic.logicalAnd(this, b);
    }

    /**
     * Element-wise logical OR with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where at least one value is non-zero, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray logicalOr(NumJArray b) {
        return arithmatic.logicalOr(this, b);
    }

    /**
     * Element-wise logical XOR with broadcasting.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where exactly one of the two values is non-zero, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray logicalXor(NumJArray b) {
        return arithmatic.logicalXor(this, b);
    }

    /**
     * Element-wise logical NOT.
     *
     * @return a new array of 1.0 where the value is 0, otherwise 0.0
     */
    public NumJArray logicalNot() {
        return arithmatic.logicalNot(this);
    }

    /**
     * Checks whether every element is non-zero.
     *
     * @return {@code true} if all elements are non-zero (also {@code true} for an empty array)
     */
    public boolean all() {
        return arithmatic.all(this);
    }

    /**
     * Checks whether at least one element is non-zero.
     *
     * @return {@code true} if any element is non-zero ({@code false} for an empty array)
     */
    public boolean any() {
        return arithmatic.any(this);
    }

    /**
     * {@link #all()} along one axis, e.g. "is every value in this row non-zero?".
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed, holding 1.0 (all non-zero) or 0.0
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray allAxis(int axis) {
        return arithmatic.allAxis(this, axis);
    }

    /**
     * {@link #any()} along one axis, e.g. "is any value in this row non-zero?".
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed, holding 1.0 (any non-zero) or 0.0
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray anyAxis(int axis) {
        return arithmatic.anyAxis(this, axis);
    }

    /** Default relative tolerance for {@code isClose} / {@code allClose} (same as NumPy). */
    public static final float DEFAULT_RTOL = 1e-5f;

    /** Default absolute tolerance for {@code isClose} / {@code allClose} (same as NumPy). */
    public static final float DEFAULT_ATOL = 1e-8f;

    /**
     * Element-wise "nearly equal" using the default tolerances ({@code rtol = 1e-5}, {@code atol = 1e-8}).
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of 1.0 where the values are close, otherwise 0.0
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray isClose(NumJArray b) {
        return arithmatic.isClose(this, b, DEFAULT_RTOL, DEFAULT_ATOL);
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
    public NumJArray isClose(NumJArray b, float rtol, float atol) {
        return arithmatic.isClose(this, b, rtol, atol);
    }

    /**
     * Element-wise "nearly equal" against a scalar, using the default tolerances.
     *
     * @param b the value to compare with
     * @return a new array of 1.0 where the values are close, otherwise 0.0
     */
    public NumJArray isClose(float b) {
        return arithmatic.isClose(this, b, DEFAULT_RTOL, DEFAULT_ATOL);
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
    public NumJArray isClose(float b, float rtol, float atol) {
        return arithmatic.isClose(this, b, rtol, atol);
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
        return arithmatic.allClose(this, b);
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
    public boolean allClose(NumJArray b, float rtol, float atol) {
        return arithmatic.allClose(this, b, rtol, atol);
    }

    /**
     * Counts the non-zero elements (NaN counts as non-zero).
     *
     * @return the number of elements that are not 0
     */
    public int countNonzero() {
        return arithmatic.countNonzero(this);
    }

    /**
     * Counts the non-zero elements along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed, holding the counts (stored as floats)
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray countNonzero(int axis) {
        return arithmatic.countNonzero(this, axis);
    }

    /**
     * Finds the coordinates of every non-zero element, in row-major order.
     *
     * @return one {@code int[]} per dimension: {@code result[d][i]} is the index along axis {@code d}
     *         of the i-th non-zero element (so {@code get(result[0][i], result[1][i])} reads it on a 2-D array)
     * @throws IllegalArgumentException if the array is 0-D
     */
    public int[][] nonzero() {
        return manipulation.nonzero(this);
    }

    /**
     * Finds the coordinates of every non-zero element as rows of a table.
     *
     * @return a new array of shape {@code [count, ndim]}; row i holds the coordinates of the i-th
     *         non-zero element (stored as floats)
     * @throws IllegalArgumentException if the array is 0-D
     */
    public NumJArray argwhere() {
        return manipulation.argwhere(this);
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
        return manipulation.take(this, indices);
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
       return manipulation.take(this, axis, indices);
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
    public NumJArray put(int[] flatIndices, float value) {
        return manipulation.put(this, flatIndices, value);
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
        return manipulation.put(this, flatIndices, values);
    }

    /**
     * Sorts along the last axis (ascending). The original is not changed.
     *
     * @return a new, sorted array of the same shape
     * @throws IllegalArgumentException if the array is 0-D
     */
    public NumJArray sort() {
        return sorts.sort(this);
    }

    /**
     * Sorts along an axis (ascending). The original is not changed.
     *
     * @param axis the axis to sort along; negative values count from the end
     * @return a new, sorted array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray sort(int axis) {
        return sorts.sort(this, axis);
    }

    /**
     * Sorts all elements as one flat list (ascending). The original is not changed.
     *
     * @return a new 1-D array with {@link #size()} sorted values
     */
    public NumJArray sortFlat() {
        return sorts.sortFlat(this);
    }

    /**
     * Finds the order that would sort the array along the last axis. Ties keep their original order.
     *
     * @return a new array of the same shape holding indices (stored as floats)
     * @throws IllegalArgumentException if the array is 0-D
     */
    public NumJArray argsort() {
        return sorts.argsort(this);
    }

    /**
     * Finds the order that would sort the array along an axis. Ties keep their original order.
     * Use {@code take(axis, order)} or {@code flip} on the result for other orderings.
     *
     * @param axis the axis to sort along; negative values count from the end
     * @return a new array of the same shape holding indices along {@code axis} (stored as floats)
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray argsort(int axis) {
        return sorts.argsort(this, axis);
    }

    /**
     * Finds the order that would sort all elements as one flat list. Ties keep their original order.
     *
     * @return a new 1-D array of flat positions (stored as floats)
     */
    public NumJArray argsortFlat() {
        return sorts.argsortFlat(this);
    }

    /**
     * Finds the distinct values, sorted ascending. All NaNs count as one value, and 0.0 equals -0.0.
     *
     * @return a new 1-D array of the unique values
     */
    public NumJArray unique() {
        return sorts.unique(this);
    }

    /**
     * Finds the distinct values together with how often each occurs.
     *
     * @return an array of two 1-D arrays: {@code [0]} the sorted unique values, {@code [1]} their counts
     *         (stored as floats)
     */
    public NumJArray[] uniqueCounts() {
        return sorts.uniqueCounts(this);
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
        return manipulation.repeat(this, repeats);
    }

    /**
     * Repeats every slice along an axis, e.g. {@code repeat(2, 0)} floats each row in place.
     *
     * @param repeats how many times to repeat each slice; must be non-negative
     * @param axis    the axis to repeat along; negative values count from the end
     * @return a new array that is {@code repeats} times longer along {@code axis}
     * @throws IllegalArgumentException if {@code repeats} is negative or the axis is out of range
     */
    public NumJArray repeat(int repeats, int axis) {
        return manipulation.repeat(this, repeats, axis);
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
        return manipulation.repeat(this, repeats, axis);
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
        return manipulation.tile(this, reps);
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
        return manipulation.pad(this, padding);
    }

    /**
     * Reverses the order of elements along evry axis. Returns a view (negative strides).
     *
     * @return a flipped view sharing data with this array
     */
    public NumJArray flip() {
        return manipulation.flip(this);
    }

    /**
     * Reverses the order of elements along the given axes. Returns a view (negative strides).
     *
     * @param axes the axes to reverse; negative values count from the end
     * @return a flipped view sharing data with this array
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public NumJArray flip(int... axes) {
        return manipulation.flip(this, axes);
    }

    /**
     * Shifts the elements of the flattened array, wrapping around, and keeps the shape.
     * Example: {@code [1, 2, 3, 4, 5].roll(2)} gives {@code [4, 5, 1, 2, 3]}.
     *
     * @param shift how far to move elements forward; negative values move them backward
     * @return a new array of the same shape
     */
    public NumJArray roll(int shift) {
        return manipulation.roll(this, shift);
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
        return manipulation.roll(this, shift, axis);
    }

    /**
     * Running total over the flattened array. (Use {@link #cumSumAxis(int)} to keep the shape.)
     *
     * @return a new 1-D array of {@link #size()} partial sums
     */
    public NumJArray cumSum() {
        return stat.cumSum(this);
    }

    /**
     * Running product over the flattened array.
     *
     * @return a new 1-D array of {@link #size()} partial products
     */
    public NumJArray cumProd() {
        return stat.cumProd(this);
    }

    /**
     * Running product along an axis.
     *
     * @param axis the axis to accumulate along; negative values count from the end
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public NumJArray cumProdAxis(int axis) {
        return stat.cumProdAxis(this, axis);
    }

    /**
     * Differences between neighbouring elements along the last axis ({@code a[i+1] - a[i]}).
     *
     * @return a new array that is one shorter along the last axis
     * @throws IllegalArgumentException if the array is 0-D
     */
    public NumJArray diff() {
        return manipulation.diff(this, 1, -1);
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
        return manipulation.diff(this, n, axis);
    }

    /**
     * Median of all elements.
     *
     * @return the middle value (the mean of the two middle values for an even count); NaN if any element is NaN
     * @throws IllegalStateException if the array is empty
     */
    public float median() {
        return stat.median(this);
    }

    /**
     * Median along an axis.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or has length 0
     */
    public NumJArray median(int axis) {
        return stat.median(this, axis);
    }

    /**
     * Percentile of all elements.
     *
     * @param p the percentile from 0 to 100
     * @return the interpolated value; NaN if any element is NaN
     * @throws IllegalArgumentException if {@code p} is outside {@code [0, 100]}
     * @throws IllegalStateException    if the array is empty
     */
    public float percentile(float p) {
        return stat.percentile(this, p);
    }

    /**
     * Percentile along an axis.
     *
     * @param p    the percentile from 0 to 100
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if {@code p} is outside {@code [0, 100]}, or the axis is out of range or has length 0
     */
    public NumJArray percentile(float p, int axis) {
        return stat.percentile(this, p, axis);
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
    public float quantile(float q) {
        return stat.quantile(this, q);
    }

    /**
     * Quantile along an axis.
     *
     * @param q    the quantile from 0 to 1
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if {@code q} is outside {@code [0, 1]}, or the axis is out of range or has length 0
     */
    public NumJArray quantile(float q, int axis) {
        return stat.quantile(this, q, axis);
    }

    /**
     * Marks the NaN elements.
     *
     * @return a new array of 1.0 where the value is NaN, otherwise 0.0
     */
    public NumJArray isNaN() {
        return arithmatic.isNaN(this);
    }

    /**
     * Marks the infinite elements (positive or negative).
     *
     * @return a new array of 1.0 where the value is +Infinity or -Infinity, otherwise 0.0
     */
    public NumJArray isInf() {
        return arithmatic.isInf(this);
    }

    /**
     * Marks the finite elements (neither NaN nor infinite).
     *
     * @return a new array of 1.0 where the value is finite, otherwise 0.0
     */
    public NumJArray isFinite() {
        return arithmatic.isFinite(this);
    }

    /**
     * Element-wise larger of two arrays, with broadcasting. NaN in either input gives NaN.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code max(this, b)}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray maximum(NumJArray b) {
        return arithmatic.maximum(this, b);
    }

    /**
     * Element-wise larger of every element and a scalar (e.g. {@code maximum(0)} is ReLU).
     *
     * @param b the scalar
     * @return a new array of {@code max(x, b)}
     */
    public NumJArray maximum(float b) {
        return arithmatic.maximum(this, b);
    }

    /**
     * Element-wise smaller of two arrays, with broadcasting. NaN in either input gives NaN.
     *
     * @param b the other array; its shape must be broadcastable with this one
     * @return a new array of {@code min(this, b)}
     * @throws IllegalArgumentException if {@code b} is null or the shapes cannot be broadcast
     */
    public NumJArray minimum(NumJArray b) {
        return arithmatic.minimum(this, b);
    }

    /**
     * Element-wise smaller of every element and a scalar.
     *
     * @param b the scalar
     * @return a new array of {@code min(x, b)}
     */
    public NumJArray minimum(float b) {
        return arithmatic.minimum(this, b);
    }

    // HELPERS

    /** Turns flat indices into slots in {@code data}, failing before anything is written. */
    public int[] resolveFlat(int[] flatIndices) {
        return manipulation.resolveFlat(this, flatIndices);
    }

    /** Ordering for sorting: normal numeric order, NaN last, and 0.0 equal to -0.0. */
    public static int compareNaNLast(float x, float y) {
        return sorts.compareNaNLast(x, y);
    }

    public static void checkQuantile(float q) {
        stat.checkQuantile(q);
    }

    public static float interpolateSorted(float[] sorted, float q) {
        return stat.interpolateSorted(sorted, q);
    }

    public static boolean closeEnough(float x, float y, float rtol, float atol) {
        return arithmatic.closeEnough(x, y, rtol, atol);
    }

    public static void checkTolerances(float rtol, float atol) {
        arithmatic.checkTolerances(rtol, atol);
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
    public float reduce(FloatBinaryOperator op) {
        return stat.reduce(this, op);
    }

    /**
     * Folds every element into one number with your own operator, starting from {@code init}.
     *
     * @param op   the operator combining the running result with the next element
     * @param init the starting value (also the result for an empty array)
     * @return the folded value
     * @throws IllegalArgumentException if {@code op} is null
     */
    public float reduce(FloatBinaryOperator op, float init) {
        return stat.reduce(this, op, init);
    }

    /**
     * Folds along one axis with your own operator, starting from the first element of each lane.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @param op   the operator combining the running result with the next element
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range or has length 0, or {@code op} is null
     */
    public NumJArray reduceAxis(int axis, FloatBinaryOperator op) {
        return stat.reduceAxis(this, axis, op);
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
    public NumJArray reduceAxis(int axis, FloatBinaryOperator op, float init) {
        return stat.reduceAxis(this, axis, op, init);
    }

    /**
     * Sums over several axes at once, e.g. {@code sum(0, 2)} on shape {@code [2, 3, 4]} gives shape {@code [3]}.
     * (Call {@link #sum()} with no arguments for a plain {@code float} total.)
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public NumJArray sum(int... axes) {
        return stat.sum(this, axes);
    }

    /**
     * Multiplies over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public NumJArray prod(int... axes) {
        return stat.prod(this, axes);
    }

    /**
     * Averages over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range or listed twice
     */
    public NumJArray mean(int... axes) {
        return stat.mean(this, axes);
    }

    /**
     * Minimum over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range, listed twice, or has length 0
     */
    public NumJArray min(int... axes) {
        return stat.min(this, axes);
    }

    /**
     * Maximum over several axes at once.
     *
     * @param axes the axes to collapse; negative values count from the end. An empty list returns a copy
     * @return a new array with those dimensions removed
     * @throws IllegalArgumentException if an axis is out of range, listed twice, or has length 0
     */
    public NumJArray max(int... axes) {
        return stat.max(this, axes);
    }

    /**
     * Position of the largest element in the flattened (row-major) array.
     *
     * @return the flat index of the first maximum
     * @throws IllegalStateException if the array is empty
     */
    public int argMaxFlat() {
        return stat.argMaxFlat(this);
    }

    /**
     * Position of the smallest element in the flattened (row-major) array.
     *
     * @return the flat index of the first minimum
     * @throws IllegalStateException if the array is empty
     */
    public int argMinFlat() {
        return stat.argMinFlat(this);
    }

    /**
     * Combines this array with another using your own operator, with broadcasting.
     *
     * @param b  the other array; its shape must be broadcastable with this one
     * @param op called as {@code op(thisValue, bValue)}
     * @return a new array holding the results
     * @throws IllegalArgumentException if {@code b} or {@code op} is null, or the shapes cannot be broadcast
     */
    public NumJArray elementwise(NumJArray b, FloatBinaryOperator op) {
        return arithmatic.elementwise(this, b, op);
    }

    /**
     * Combines every element with a scalar using your own operator.
     *
     * @param b  the scalar
     * @param op called as {@code op(thisValue, b)}
     * @return a new array holding the results
     * @throws IllegalArgumentException if {@code op} is null
     */
    public NumJArray elementwise(float b, FloatBinaryOperator op) {
        return arithmatic.elementwise(this, b, op);
    }

    /**
     * Applies your own function to every element (same as {@link #map(floatUnaryOperator)}).
     *
     * @param op the function to apply
     * @return a new array of the same shape
     * @throws IllegalArgumentException if {@code op} is null
     */
    public NumJArray elementwise(FloatUnaryOperator op) {
        return arithmatic.elementwise(this, op);
    }

    /** Computes a value from an elements coordinates; used by {@link #fill(CoordinateFunction)}. */
    @FunctionalInterface
    public interface CoordinateFunction {
        /**
         * @param coordinate one index per dimension; the array is reused between calls, so copy it if you keep it
         * @return the value to store at that position
         */
        float apply(int[] coordinate);
    }

    /**
     * Applies a function to every element, storing the result back into this array.
     *
     * @param op the function to apply
     * @return this array
     * @throws IllegalArgumentException if {@code op} is null
     */
    public NumJArray elementwisei(FloatUnaryOperator op) {
        return arithmatic.elementwisei(this, op);
    }

    /**
     * Combines every element with a scalar using your operator, storing the result in this array.
     *
     * @param b  the scalar
     * @param op called as {@code op(thisValue, b)}
     * @return this array
     * @throws IllegalArgumentException if {@code op} is null
     */
    public NumJArray elementwisei(float b, FloatBinaryOperator op) {
        return arithmatic.elementwisei(this, b, op);
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
    public NumJArray elementwisei(NumJArray b, FloatBinaryOperator op) {
        return arithmatic.elementwisei(this, b, op);
    }

    /**
     * Applies a function to every element.
     *
     * @param a the array
     * @param f the function to apply
     * @return a new array of the same shape
     * @throws IllegalArgumentException if {@code f} is null
     */
    public NumJArray map(FloatUnaryOperator f) {
        return arithmatic.map(this, f);
    }

    /**
     * Applies a function to every element in-place, mutating this array.
     * Works safely across contiguous arrays, slices, and strided views.
     *
     * @param a the array
     * @param f the function to apply
     * @return this array instance
     * @throws IllegalArgumentException if {@code f} is null
     */
    public NumJArray mapi(FloatUnaryOperator f) {
        return arithmatic.mapi(this, f);
    }

    /**
     * In-place addition of a scalar.
     *
     * @param b the scalar
     * @return this array
     */
    public NumJArray addi(float b) {
        return arithmatic.addi(this, b);
    }

    /**
     * In-place addition of an array (broadcast to this shape).
     *
     * @param b an array broadcastable to this arrays shape
     * @return this array
     * @throws IllegalArgumentException if {@code b} cannot be broadcast to this shape
     */
    public NumJArray addi(NumJArray b) {
        return arithmatic.addi(this, b);
    }

    /**
     * In-place subtraction of a scalar.
     *
     * @param b the scalar
     * @return this array
     */
    public NumJArray subi(float b) {
        return arithmatic.subi(this, b);
    }

    /**
     * In-place subtraction of an array (broadcast to this shape).
     *
     * @param b an array broadcastable to this arrays shape
     * @return this array
     * @throws IllegalArgumentException if {@code b} cannot be broadcast to this shape
     */
    public NumJArray subi(NumJArray b) {
        return arithmatic.subi(this, b);
    }

    /**
     * In-place multiplication by a scalar.
     *
     * @param b the scalar
     * @return this array
     */
    public NumJArray muli(float b) {
        return arithmatic.muli(this, b);
    }

    /**
     * In-place multiplication by an array (broadcast to this shape).
     *
     * @param b an array broadcastable to this arrays shape
     * @return this array
     * @throws IllegalArgumentException if {@code b} cannot be broadcast to this shape
     */
    public NumJArray muli(NumJArray b) {
        return arithmatic.muli(this, b);
    }

    /**
     * In-place division by a scalar.
     *
     * @param b the scalar
     * @return this array
     */
    public NumJArray divi(float b) {
        return arithmatic.divi(this, b);
    }

    /**
     * In-place division by an array (broadcast to this shape).
     *
     * @param b an array broadcastable to this arrays shape
     * @return this array
     * @throws IllegalArgumentException if {@code b} cannot be broadcast to this shape
     */
    public NumJArray divi(NumJArray b) {
        return arithmatic.divi(this, b);
    }

    /**
     * In-place element-wise square root.
     * 
     * @return this array
     */
    public NumJArray sqrti() {
        return arithmatic.sqrti(this);
    }

    /**
     * In-place element-wise power.
     * 
     * @param b the exponent
     * @return this array
     */
    public NumJArray powi(float b) {
        return arithmatic.powi(this, b);
    }

    /**
     * In-place element-wise e^x.
     * 
     * @return this array
     */
    public NumJArray expi() {
        return arithmatic.expi(this);
    }

    /**
     * In-place element-wise natural logarithm.
     * 
     * @return this array
     */
    public NumJArray logi() {
        return arithmatic.logi(this);
    }

    /**
     * In-place modulo by a scalar.
     * 
     * @param b the divisor
     * @return this array
     */
    public NumJArray modi(float b) {
        return arithmatic.modi(this, b);
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
        return manipulation.assign(this, src);
    }

    /**
     * Sets every element to one value.
     *
     * @param value the value to store
     * @return this array
     */
    public NumJArray fill(float value) {
        return manipulation.fill(this, value);
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
        return manipulation.fill(this, f);
    }

    /**
     * Sets elements to one value wherever a mask is non-zero (e.g. {@code x.putWhere(x.less(0), 0)}).
     *
     * @param mask  non-zero = true; must be broadcastable to this arrays shape
     * @param value the value to store where the mask is true
     * @return this array
     * @throws IllegalArgumentException if {@code mask} is null or cannot be broadcast to this shape
     */
    public NumJArray putWhere(NumJArray mask, float value) {
        return arithmatic.putWhere(this, mask, value);
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
        return arithmatic.putWhere(this, mask, values);
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
    public NumJArray conditionalPut(FloatPredicate test, float value) {
        return arithmatic.conditionalPut(this, test, value);
    }

    /**
     * Builds a mask from a test on each element.
     *
     * @param test the condition to check
     * @return a new array of 1.0 where the test is true, otherwise 0.0
     * @throws IllegalArgumentException if {@code test} is null
     */
    public NumJArray where(FloatPredicate test) {
        return arithmatic.where(this, test);
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
        return arithmatic.where(cond, x, y);
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
    public static NumJArray where(NumJArray cond, float x, float y) {
        return arithmatic.where(cond, x, y);
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
    public float getFlat(int flatIndex) {
        return data[pointer(flatIndex)];
    }

    /**
     * Writes an element by its position in the flattened row-major order.
     *
     * @param value     the value to store
     * @param flatIndex the logical position; negative values count from the end
     * @throws IndexOutOfBoundsException if {@code flatIndex} is outside the array
     */
    public void setFlat(float value, int flatIndex) {
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
        Validator.checkShape(target);
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
    public float broadcastGet(int... coordinates) {
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

        float[] flat = toContiguousData();
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

    public static void requireOperator(Object op) {
        if (op == null) {
            throw new IllegalArgumentException("operator must not be null");
        }
    }

    public static int[] concatInts(int[] head, int... tail) {
        int[] out = new int[head.length + tail.length];
        System.arraycopy(head, 0, out, 0, head.length);
        System.arraycopy(tail, 0, out, head.length, tail.length);
        return out;
    }

    /** Slots in {@code data} of every element, in logical row-major order. */
    public int[] positions() {
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
    public NumJArray reduceAxes(int[] axes, BiFunction<NumJArray, Integer, NumJArray> single) {
        return stat.reduceAxes(this, axes, single);
    }

    public void appendPretty(StringBuilder sb, String[] cells, int width, int dim, int start, int indent) {
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

    public static int[] createStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int running = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = running;
            running *= shape[i];
        }
        return strides;
    }

    public static float[] requireLength(float[] data, int[] shape) {
        if (data == null) {
            throw new IllegalArgumentException("data must not be null");
        }
        int expected = Validator.sizeOf(Validator.checkShape(shape));
        if (data.length != expected) {
            throw new IllegalArgumentException(
                "data has " + data.length + " elements but shape " + Arrays.toString(shape) + " needs " + expected);
        }
        return data;
    }

    public static int normalizeAxis(int axis, int rank) {
        int ax = axis < 0 ? axis + rank : axis;
        if (ax < 0 || ax >= rank) {
            throw new IllegalArgumentException(
                "Axis " + axis + " is out of bounds for an array with " + rank + " dimension(s)");
        }
        return ax;
    }

    public int normalizeAxis(int axis) {
        return normalizeAxis(axis, shape.length);
    }

    public static long clamp(long v, long lo, long hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    public static int[] removeAt(int[] arr, int index) {
        int[] out = new int[arr.length - 1];
        for (int i = 0, j = 0; i < arr.length; i++) {
            if (i != index) out[j++] = arr[i];
        }
        return out;
    }

    public void requireNonEmpty(String op) {
        if (size == 0) {
            throw new IllegalStateException(op + "() is not defined for an empty array");
        }
    }

    public int getFlatIndex(int[] indices) {
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
    public boolean isContiguous() {
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (shape[i] != 1 && strides[i] != expected) return false;
            expected *= shape[i];
        }
        return true;
    }

    /** Always returns a fresh row-major copy of the logical values. Safe to modify. */
    public float[] toContiguousData() {
        float[] out = new float[size];
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
    public float[] readOnlyData() {
        if (offset == 0 && data.length == size && isContiguous()) return data;
        return toContiguousData();
    }

    public int argFlat(boolean findMax) {
        return stat.argFlat(this, findMax);
    }

    public int[] unravel(int flat) {
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
    public int[] laneOffsets(int axis) {
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
    public NumJArray reduceLanes(int axis, FloatBinaryOperator op, Float init) {
        int ax = normalizeAxis(axis);
        int len = shape[ax];
        int st = strides[ax];

        if (init == null && len == 0) {
            throw new IllegalArgumentException("Cannot reduce over axis " + ax + " because it has length 0");
        }

        int[] lanes = laneOffsets(ax);
        float[] out = new float[lanes.length];

        for (int l = 0; l < lanes.length; l++) {
            int base = lanes[l];
            float acc;
            int k;
            if (init == null) {
                acc = data[base];
                k = 1;
            } else {
                acc = init;
                k = 0;
            }
            for (; k < len; k++) {
                acc = op.applyAsFloat(acc, data[base + k * st]);
            }
            out[l] = acc;
        }
        return new NumJArray(out, removeAt(shape, ax));
    }

    public NumJArray argReduceAxis(int axis, boolean findMax) {
        return stat.argReduceAxis(this, axis, findMax);
    }

    /** View of {@code [start, stop)} along one axis, keeping all other axes whole. */
    public NumJArray sliceAxis(int ax, int start, int stop) {
        return manipulation.sliceAxis(this, ax, start, stop);
    }

    public static int[] broadcastShape(int[] a, int[] b) {
        return arithmatic.broadcastShape(a, b);
    }

    /** Strides that make an array of {@code inShape} look like {@code outShape} (stride 0 where it repeats). */
    public static int[] broadcastStrides(int[] inShape, int[] inStrides, int[] outShape) {
        int[] out = new int[outShape.length];
        int lead = outShape.length - inShape.length;
        for (int i = 0; i < inShape.length; i++) {
            out[i + lead] = (inShape[i] == 1) ? 0 : inStrides[i];
        }
        return out;
    }

    public NumJArray broadcastOp(NumJArray b, FloatBinaryOperator op) {
        return arithmatic.broadcastOp(this, b, op);
    }

    public void appendNested(StringBuilder sb, int dim, int pos) {
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
        njio.save(this, filePath);
    }

    /**
     * Loads an array from NumJ's binary format.
     *
     * @param filePath the source .nj file
     * @return the loaded array
     */
    public static NumJArray load(String filePath) {
        return njio.load(filePath);
    }

    /**
     * Saves this array in NumPy's .npy format.
     *
     * // SHOULD WORK, BUT HAVEN'T FULLY TESTED.
     *
     * @param filePath the destination .npy file
     */
    public void saveNpy(String filePath) {
        npio.saveNpy(this, filePath);
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
        return npio.loadNpy(filePath);
    }
}