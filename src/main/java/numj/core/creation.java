package numj.core;

import java.util.Arrays;
import java.util.Random;

public class creation {

    /**
     * Creates an array of zeros.
     *
     * @param shape the size of each dimension
     * @return a new contiguous array filled with {@code 0.0}
     * @throws IllegalArgumentException if the shape is invalid
     */
    public static NumJArray zeros(int... shape) {
        return new NumJArray(new float[Validator.sizeOf(Validator.checkShape(shape))], shape);
    }

    /**
     * Creates an array of ones.
     *
     * @param shape the size of each dimension
     * @return a new contiguous array filled with {@code 1.0}
     * @throws IllegalArgumentException if the shape is invalid
     */
    public static NumJArray ones(int... shape) {
        return full(1.0f, shape);
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
        float[] d = new float[Validator.sizeOf(Validator.checkShape(shape))];
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
    public static NumJArray of(float... values) {
        if (values == null) {
            throw new IllegalArgumentException("values must not be null");
        }
        return new NumJArray(values.clone(), new int[]{values.length});
    }

    /**
     * Creates a 2-D array from a rectangular {@code float[][]} (the values are copied).
     *
     * @param rows the rows; every row must have the same length
     * @return a new 2-D array of shape {@code [rows.length, rows[0].length]}
     * @throws IllegalArgumentException if {@code rows} is null, or the rows are ragged
     */
    public static NumJArray of(float[][] rows) {
        if (rows == null) {
            throw new IllegalArgumentException("rows must not be null");
        }
        int cols = rows.length == 0 ? 0 : rows[0].length;
        float[] d = new float[rows.length * cols];
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
    public static NumJArray arange(float stop) {
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
    public static NumJArray arange(float start, float stop) {
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
    public static NumJArray arange(float start, float stop, float step) {
        if (step == 0 || Float.isNaN(step) || Float.isInfinite(step)) {
            throw new IllegalArgumentException("step must be a non-zero finite number");
        }
        int n = (int) Math.max(0, Math.ceil((stop - start) / step));
        float[] d = new float[n];
        for (int i = 0; i < n; i++) {
            d[i] = start + i * step;
        }
        return new NumJArray(d, new int[]{n});
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
        if (num < 0) {
            throw new IllegalArgumentException("num must be non-negative, got " + num);
        }
        float[] d = new float[num];
        if (num == 0) {
            return new NumJArray(d, new int[]{0});
        }
        if (num == 1) {
            d[0] = start;
        } else {
            float step = (stop - start) / (num - 1);
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
        float[] d = new float[n * n];
        for (int i = 0; i < n; i++) {
            d[i * n + i] = 1.0f;
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
    public static NumJArray normal(float mean, float stdDev, int... shape) {
        if (stdDev < 0) {
            throw new IllegalArgumentException("stdDev must be non-negative, got " + stdDev);
        }
        float[] d = new float[Validator.sizeOf(Validator.checkShape(shape))];
        Random rand = new Random();
        for (int i = 0; i < d.length; i++) {
            d[i] = (float) (mean + rand.nextGaussian() * stdDev);
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
    public static NumJArray uniform(float low, float high, int... shape) {
        if (high < low) {
            throw new IllegalArgumentException("high must be >= low");
        }
        float[] d = new float[Validator.sizeOf(Validator.checkShape(shape))];
        Random rand = new Random();
        for (int i = 0; i < d.length; i++) {
            d[i] = low + rand.nextFloat() * (high - low);
        }
        return new NumJArray(d, shape);
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
        float[] d = new float[Validator.sizeOf(Validator.checkShape(new int[]{n, m}))];
        for (int i = 0; i < n; i++) {
            long j = (long) i + k;
            if (j >= 0 && j < m) d[i * m + (int) j] = 1.0f;
        }
        return new NumJArray(d, new int[]{n, m});
    }

}
