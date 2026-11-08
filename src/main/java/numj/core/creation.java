package numj.core;

import java.util.Arrays;
import java.util.Random;
import java.util.function.IntToDoubleFunction;

public class creation {

    /** Shared generator used by every random creator that is not given its own {@link Random}. */
    private static Random defaultRng = new Random();

    /**
     * Seeds the shared generator so that all random creators called without an explicit
     * {@link Random} become reproducible (like {@code np.random.seed}).
     *
     * @param seed the seed
     */
    public static void setSeed(long seed) {
        defaultRng = new Random(seed);
    }

    public static Random getSeed() {
        return defaultRng;
    }

    // ------------------------------------------------------------------ basics

    public static NumJArray zeros(int... shape) {
        return new NumJArray(new float[Validator.sizeOf(Validator.checkShape(shape))], shape);
    }

    public static NumJArray ones(int... shape) {
        return full(1.0f, shape);
    }

    public static NumJArray full(float value, int... shape) {
        float[] d = new float[Validator.sizeOf(Validator.checkShape(shape))];
        Arrays.fill(d, value);
        return new NumJArray(d, shape);
    }

    public static NumJArray of(float... values) {
        if (values == null) {
            throw new IllegalArgumentException("values must not be null");
        }
        return new NumJArray(values.clone(), new int[]{values.length});
    }

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

    // ------------------------------------------------------------------ ranges

    public static NumJArray arange(float stop) {
        return arange(0, stop, 1);
    }

    public static NumJArray arange(float start, float stop) {
        return arange(start, stop, 1);
    }

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
     * Creates a 1-D array of {@code num} values spaced evenly on a log scale, from
     * {@code base^startExp} to {@code base^stopExp} (inclusive). For example
     * {@code logspace(0, 3, 4, 10)} gives {@code [1, 10, 100, 1000]}.
     *
     * @param startExp the exponent of the first value
     * @param stopExp  the exponent of the last value
     * @param num      how many values to generate; must be non-negative
     * @param base     the base of the logarithm; must be positive
     * @return a new 1-D array of length {@code num}
     * @throws IllegalArgumentException if {@code num} is negative or {@code base <= 0}
     */
    public static NumJArray logspace(float startExp, float stopExp, int num, float base) {
        if (base <= 0) {
            throw new IllegalArgumentException("base must be positive, got " + base);
        }
        NumJArray exps = linspace(startExp, stopExp, num);
        float[] d = new float[num];
        for (int i = 0; i < num; i++) {
            d[i] = (float) Math.pow(base, exps.get(i));
        }
        return new NumJArray(d, new int[]{num});
    }

    // ------------------------------------------------------------------ matrices

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

    public static NumJArray eye(int n) {
        return eye(n, n, 0);
    }

    public static NumJArray eye(int n, int m) {
        return eye(n, m, 0);
    }

    public static NumJArray eye(int n, int m, int k) {
        float[] d = new float[Validator.sizeOf(Validator.checkShape(new int[]{n, m}))];
        for (int i = 0; i < n; i++) {
            long j = (long) i + k;
            if (j >= 0 && j < m) d[i * m + (int) j] = 1.0f;
        }
        return new NumJArray(d, new int[]{n, m});
    }

    /**
     * Creates a square matrix with the given values on the main diagonal and zeros elsewhere.
     *
     * @param values the diagonal entries
     * @return a new {@code n x n} 2-D array where {@code n = values.length}
     * @throws IllegalArgumentException if {@code values} is null
     */
    public static NumJArray diag(float... values) {
        if (values == null) {
            throw new IllegalArgumentException("values must not be null");
        }
        int n = values.length;
        float[] d = new float[n * n];
        for (int i = 0; i < n; i++) {
            d[i * n + i] = values[i];
        }
        return new NumJArray(d, new int[]{n, n});
    }

    /**
     * Creates an {@code n x m} matrix with ones at and below the {@code k}-th diagonal
     * and zeros elsewhere (a lower-triangular mask).
     *
     * @param n the number of rows
     * @param m the number of columns
     * @param k the diagonal: 0 is the main one, positive is above it, negative below
     * @return a new 2-D array
     * @throws IllegalArgumentException if a size is negative
     */
    public static NumJArray tri(int n, int m, int k) {
        float[] d = new float[Validator.sizeOf(Validator.checkShape(new int[]{n, m}))];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if ((long) j <= (long) i + k) d[i * m + j] = 1.0f;
            }
        }
        return new NumJArray(d, new int[]{n, m});
    }

    // ------------------------------------------------------------------ generic

    /**
     * Creates an array by calling a function with each element's flat (row-major) index.
     * For example {@code fromFunction(i -> i * i, 5)} gives {@code [0, 1, 4, 9, 16]}.
     *
     * @param f     maps a flat index to the value stored there
     * @param shape the size of each dimension
     * @return a new contiguous array
     * @throws IllegalArgumentException if {@code f} is null or the shape is invalid
     */
    public static NumJArray fromFunction(IntToDoubleFunction f, int... shape) {
        if (f == null) {
            throw new IllegalArgumentException("f must not be null");
        }
        float[] d = new float[Validator.sizeOf(Validator.checkShape(shape))];
        for (int i = 0; i < d.length; i++) {
            d[i] = (float) f.applyAsDouble(i);
        }
        return new NumJArray(d, shape);
    }

    // ------------------------------------------------------------------ random: uniform

    /**
     * Creates an array of uniformly distributed random numbers in {@code [low, high)}
     * using the shared generator (see {@link #setSeed(long)}).
     */
    public static NumJArray uniform(float low, float high, int... shape) {
        return uniform(defaultRng, low, high, shape);
    }

    /**
     * Creates an array of uniformly distributed random numbers in {@code [low, high)}
     * using the given generator. Pass {@code new Random(seed)} for reproducible output.
     *
     * @param rng   the random source
     * @param low   the lower bound (inclusive)
     * @param high  the upper bound (exclusive); must be {@code >= low}
     * @param shape the size of each dimension
     * @return a new contiguous array of random samples
     * @throws IllegalArgumentException if {@code rng} is null, {@code high < low}, or the shape is invalid
     */
    public static NumJArray uniform(Random rng, float low, float high, int... shape) {
        requireRng(rng);
        if (high < low) {
            throw new IllegalArgumentException("high must be >= low");
        }
        float[] d = new float[Validator.sizeOf(Validator.checkShape(shape))];
        for (int i = 0; i < d.length; i++) {
            d[i] = low + rng.nextFloat() * (high - low);
        }
        return new NumJArray(d, shape);
    }

    /** Uniform random numbers in {@code [0, 1)}, like {@code np.random.rand}. */
    public static NumJArray rand(int... shape) {
        return uniform(defaultRng, 0f, 1f, shape);
    }

    /** Uniform random numbers in {@code [0, 1)} from the given generator. */
    public static NumJArray rand(Random rng, int... shape) {
        return uniform(rng, 0f, 1f, shape);
    }

    // ------------------------------------------------------------------ random: normal

    /**
     * Creates an array of normally distributed random numbers using the shared generator
     * (see {@link #setSeed(long)}).
     */
    public static NumJArray normal(float mean, float stdDev, int... shape) {
        return normal(defaultRng, mean, stdDev, shape);
    }

    /**
     * Creates an array of normally distributed random numbers using the given generator.
     *
     * @param rng    the random source
     * @param mean   the mean of the distribution
     * @param stdDev the standard deviation; must be non-negative
     * @param shape  the size of each dimension
     * @return a new contiguous array of random samples
     * @throws IllegalArgumentException if {@code rng} is null, {@code stdDev} is negative, or the shape is invalid
     */
    public static NumJArray normal(Random rng, float mean, float stdDev, int... shape) {
        requireRng(rng);
        if (stdDev < 0) {
            throw new IllegalArgumentException("stdDev must be non-negative, got " + stdDev);
        }
        float[] d = new float[Validator.sizeOf(Validator.checkShape(shape))];
        for (int i = 0; i < d.length; i++) {
            d[i] = (float) (mean + rng.nextGaussian() * stdDev);
        }
        return new NumJArray(d, shape);
    }

    /** Standard normal samples (mean 0, std 1), like {@code np.random.randn}. */
    public static NumJArray randn(int... shape) {
        return normal(defaultRng, 0f, 1f, shape);
    }

    /** Standard normal samples (mean 0, std 1) from the given generator. */
    public static NumJArray randn(Random rng, int... shape) {
        return normal(rng, 0f, 1f, shape);
    }

    // ------------------------------------------------------------------ random: other

    /**
     * Creates an array of random whole numbers (stored as floats) in {@code [low, high)}
     * using the shared generator.
     */
    public static NumJArray randint(int low, int high, int... shape) {
        return randint(defaultRng, low, high, shape);
    }

    /**
     * Creates an array of random whole numbers (stored as floats) in {@code [low, high)}.
     *
     * @param rng   the random source
     * @param low   the lower bound (inclusive)
     * @param high  the upper bound (exclusive); must be {@code > low}
     * @param shape the size of each dimension
     * @return a new contiguous array
     * @throws IllegalArgumentException if {@code rng} is null, {@code high <= low}, or the shape is invalid
     */
    public static NumJArray randint(Random rng, int low, int high, int... shape) {
        requireRng(rng);
        if (high <= low) {
            throw new IllegalArgumentException("high must be > low");
        }
        long range = (long) high - low;
        float[] d = new float[Validator.sizeOf(Validator.checkShape(shape))];
        for (int i = 0; i < d.length; i++) {
            d[i] = (float) (low + (long) (rng.nextDouble() * range));
        }
        return new NumJArray(d, shape);
    }

    /**
     * Creates an array of Bernoulli samples: each element is {@code 1.0} with probability
     * {@code p} and {@code 0.0} otherwise (handy for dropout masks).
     *
     * @param rng   the random source
     * @param p     the probability of a one, in {@code [0, 1]}
     * @param shape the size of each dimension
     * @return a new contiguous array of zeros and ones
     * @throws IllegalArgumentException if {@code rng} is null, {@code p} is outside [0, 1], or the shape is invalid
     */
    public static NumJArray bernoulli(Random rng, float p, int... shape) {
        requireRng(rng);
        if (!(p >= 0f && p <= 1f)) {
            throw new IllegalArgumentException("p must be in [0, 1], got " + p);
        }
        float[] d = new float[Validator.sizeOf(Validator.checkShape(shape))];
        for (int i = 0; i < d.length; i++) {
            d[i] = rng.nextFloat() < p ? 1.0f : 0.0f;
        }
        return new NumJArray(d, shape);
    }

    /** Bernoulli samples using the shared generator. */
    public static NumJArray bernoulli(float p, int... shape) {
        return bernoulli(defaultRng, p, shape);
    }

    /**
     * Creates a 1-D array holding a random permutation of {@code 0 .. n-1}
     * (Fisher-Yates shuffle) using the shared generator.
     */
    public static NumJArray permutation(int n) {
        return permutation(defaultRng, n);
    }

    /**
     * Creates a 1-D array holding a random permutation of {@code 0 .. n-1} (Fisher-Yates shuffle).
     *
     * @param rng the random source
     * @param n   the number of elements; must be non-negative
     * @return a new 1-D array of length {@code n}
     * @throws IllegalArgumentException if {@code rng} is null or {@code n} is negative
     */
    public static NumJArray permutation(Random rng, int n) {
        requireRng(rng);
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative, got " + n);
        }
        float[] d = new float[n];
        for (int i = 0; i < n; i++) d[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            float t = d[i]; d[i] = d[j]; d[j] = t;
        }
        return new NumJArray(d, new int[]{n});
    }

    private static void requireRng(Random rng) {
        if (rng == null) {
            throw new IllegalArgumentException("rng must not be null");
        }
    }
}