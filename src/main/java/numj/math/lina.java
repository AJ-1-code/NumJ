package numj.math;

import java.util.Arrays;

import numj.core.NumJArray;
import numj.core.Validator;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class lina {

    private static final VectorSpecies<Float> MATMUL_SPECIES = FloatVector.SPECIES_PREFERRED;

    /**
     * Builds a diagonal matrix from a 1-D array, or extracts the main diagonal of a 2-D array.
     *
     * @return for 1-D input an {@code n x n} matrix; for 2-D input a new 1-D array (a copy)
     * @throws IllegalArgumentException if the array is not 1-D or 2-D
     */
    public static NumJArray diag(NumJArray a) {
        return diag(a, 0);
    }

    /**
     * Builds a matrix with this 1-D array on a chosen diagonal, or extracts that diagonal from a 2-D array.
     *
     * @param k which diagonal: 0 is the main one, positive values are above it, negative values below
     * @return for 1-D input a square matrix of size {@code n + |k|}; for 2-D input a new 1-D array (a copy,
     *         empty if the diagonal does not exist)
     * @throws IllegalArgumentException if the array is not 1-D or 2-D
     */
    public static NumJArray diag(NumJArray a, int k) {
        int[] shape = a.shape;
        int[] strides = a.strides;
        int offset = a.offset;

        if (shape.length == 1) {
            int n = shape[0];
            int dim = n + Math.abs(k);
            float[] out = new float[Validator.sizeOf(new int[]{dim, dim})];
            float[] v = a.toContiguousData();

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

            return new NumJArray(
                a.data,
                new int[]{len},
                new int[]{strides[0] + strides[1]},
                start
            ).copy();
        }

        throw new IllegalArgumentException("diag() needs a 1D or 2D array, got " + shape.length + "D");
    }


    /**
     * Outer product of two arrays, each treated as a flat list: {@code result[i][j] = this[i] * b[j]}.
     *
     * @param b the other array (any shape; flattened)
     * @return a new 2-D array of shape {@code [this.size, b.size]}
     * @throws IllegalArgumentException if {@code b} is null
     */
    public static NumJArray outer(NumJArray a, NumJArray b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Arrays must not be null");
        }

        float[] x = a.readOnlyData();
        float[] y = b.readOnlyData();
        float[] out = new float[Validator.sizeOf(new int[]{x.length, y.length})];

        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < y.length; j++) {
                out[i * y.length + j] = x[i] * y[j];
            }
        }

        return new NumJArray(out, new int[]{x.length, y.length});
    }

    

    /**
     * Euclidean (L2) norm of all elements, i.e. {@code sqrt(sum(x^2))}.
     *
     * @return the norm
     */
    public static float norm(NumJArray array) {
        float acc = 0;
        for (float v : array.readOnlyData()) acc += v * v;
        return (float) Math.sqrt(acc);
    }

    /**
     * Euclidean (L2) norm along an axis, e.g. the length of each row vector.
     *
     * @param axis the axis to collapse; negative values count from the end
     * @return a new array with that dimension removed
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray normAxis(NumJArray array, int axis) {
        return array.mul(array).sumAxis(axis).sqrt();
    }

    /**
     * Sum of the main diagonal of a 2-D array. For N-D arrays use {@link #trace(int, int)}.
     *
     * @return the trace
     * @throws IllegalArgumentException if the array is not 2-D
     */
    public static float trace(NumJArray array) {
        if (array.shape.length != 2) {
            throw new IllegalArgumentException(
                "trace() needs a 2D array, got " + array.shape.length + "D; use trace(axis1, axis2) for N-D arrays");
        }
        return diagonal(array, 0, 1).sum();
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
    public static NumJArray trace(NumJArray array, int axis1, int axis2) {
        return diagonal(array, axis1, axis2).sumAxis(-1);
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
    public static NumJArray diagonal(NumJArray array, int axis1, int axis2) {
        int a1 = array.normalizeAxis(axis1);
        int a2 = array.normalizeAxis(axis2);
        if (a1 == a2) {
            throw new IllegalArgumentException("diagonal axes must be different, got " + axis1 + " and " + axis2);
        }

        int nd = array.shape.length;
        int[] newShape = new int[nd - 1];
        int[] newStrides = new int[nd - 1];
        int j = 0;
        for (int i = 0; i < nd; i++) {
            if (i != a1 && i != a2) {
                newShape[j] = array.shape[i];
                newStrides[j++] = array.strides[i];
            }
        }
        newShape[j] = Math.min(array.shape[a1], array.shape[a2]);
        newStrides[j] = array.strides[a1] + array.strides[a2];
        return new NumJArray(array.data, newShape, newStrides, array.offset);
    }

    /**
     * Inner product of two arrays of the same shape: {@code sum(this * other)}.
     * (For NumPy-style {@code dot} on vectors, matrices and N-D arrays use {@link #dot(NumJArray)}.)
     *
     * @param other an array with exactly the same shape
     * @return the scalar inner product
     * @throws IllegalArgumentException if {@code other} is null or the shapes differ
     */
    public static float inner(NumJArray array, NumJArray other) {
        if (other == null) {
            throw new IllegalArgumentException("other must not be null");
        }
        if (!Arrays.equals(array.shape, other.shape)) {
            throw new IllegalArgumentException(
                "Shapes must match for inner: " + Arrays.toString(array.shape) + " vs " + Arrays.toString(other.shape));
        }
        float[] a = array.readOnlyData();
        float[] b = other.readOnlyData();
        float acc = 0;
        for (int i = 0; i < a.length; i++) acc += a[i] * b[i];
        return acc;
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
    public static NumJArray dot(NumJArray a, NumJArray b) {
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }
        if (a.shape.length == 0 || b.shape.length == 0) {
            return a.mul(b);
        }

        int nd = a.shape.length;
        int nb = b.shape.length;
        int k = a.shape[nd - 1];
        int kb = (nb == 1) ? b.shape[0] : b.shape[nb - 2];
        if (k != kb) {
            throw new IllegalArgumentException(
                "Shapes not aligned for dot: " + Arrays.toString(a.shape) + " and " + Arrays.toString(b.shape)
                    + " (" + k + " != " + kb + ")");
        }

        int[] leadShape = Arrays.copyOfRange(a.shape, 0, nd - 1);
        NumJArray a2 = a.reshape(Validator.sizeOf(leadShape), k);

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
        System.arraycopy(a.shape, 0, outShape, 0, nd - 1);
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
    public static NumJArray matMul(NumJArray a, NumJArray b) {
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }
        if (a.shape.length == 0 || b.shape.length == 0) {
            throw new IllegalArgumentException("matMul does not accept scalars; use mul instead");
        }

        boolean aVec = a.shape.length == 1;
        boolean bVec = b.shape.length == 1;
        NumJArray a2 = aVec ? a.expandDims(0) : a;
        NumJArray b2 = bVec ? b.expandDims(1) : b;

        int na = a2.shape.length;
        int nb = b2.shape.length;
        int m = a2.shape[na - 2];
        int k = a2.shape[na - 1];
        int p = b2.shape[nb - 1];

        if (k != b2.shape[nb - 2]) {
            throw new IllegalArgumentException(
                "Inner dimensions do not match for matMul: " + Arrays.toString(a.shape) + " x " + Arrays.toString(b.shape));
        }

        int[] batch = NumJArray.broadcastShape(
            Arrays.copyOfRange(a2.shape, 0, na - 2),
            Arrays.copyOfRange(b2.shape, 0, nb - 2));
        int nBatch = batch.length;

        NumJArray aB = a2.broadcastTo(NumJArray.concatInts(batch, m, k));
        NumJArray bB = b2.broadcastTo(NumJArray.concatInts(batch, k, p));

        int count = Validator.sizeOf(batch);
        float[] out = new float[count * m * p];

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
                        
                if (bCol == 1 && p >= MATMUL_SPECIES.length() * 4) {
                    int lanes = MATMUL_SPECIES.length();
                    int tileWidth = lanes * 4;
                    int vectorBound = (p / tileWidth) * tileWidth;
                
                    // SIMD fast path: four independent vector accumulators.
                    for (int j0 = 0; j0 < vectorBound; j0 += tileWidth) {
                        FloatVector acc0 = FloatVector.zero(MATMUL_SPECIES);
                        FloatVector acc1 = FloatVector.zero(MATMUL_SPECIES);
                        FloatVector acc2 = FloatVector.zero(MATMUL_SPECIES);
                        FloatVector acc3 = FloatVector.zero(MATMUL_SPECIES);
                    
                        for (int kk = 0; kk < k; kk++) {
                            float valA = aB.data[rowA + kk * aCol];
                            int rowB = baseB + kk * bRow + j0;
                        
                            acc0 = acc0.add(
                                    FloatVector.fromArray(MATMUL_SPECIES, bB.data, rowB)
                                            .mul(valA));
                            
                            acc1 = acc1.add(
                                    FloatVector.fromArray(
                                            MATMUL_SPECIES, bB.data, rowB + lanes)
                                            .mul(valA));
                                    
                            acc2 = acc2.add(
                                    FloatVector.fromArray(
                                            MATMUL_SPECIES, bB.data, rowB + 2 * lanes)
                                            .mul(valA));
                                    
                            acc3 = acc3.add(
                                    FloatVector.fromArray(
                                            MATMUL_SPECIES, bB.data, rowB + 3 * lanes)
                                            .mul(valA));
                        }
                    
                        acc0.intoArray(out, rowOut + j0);
                        acc1.intoArray(out, rowOut + j0 + lanes);
                        acc2.intoArray(out, rowOut + j0 + 2 * lanes);
                        acc3.intoArray(out, rowOut + j0 + 3 * lanes);
                    }
                
                    // Scalar tail for columns not covered by complete vector tiles.
                    for (int j = vectorBound; j < p; j++) {
                        float sum = 0f;
                    
                        for (int kk = 0; kk < k; kk++) {
                            float valA = aB.data[rowA + kk * aCol];
                            int rowB = baseB + kk * bRow;
                        
                            sum += valA * bB.data[rowB + j];
                        }
                    
                        out[rowOut + j] = sum;
                    }
                } else {
                    // Original general-stride fallback.
                    for (int kk = 0; kk < k; kk++) {
                        float valA = aB.data[rowA + kk * aCol];
                        int rowB = baseB + kk * bRow;
                    
                        for (int j = 0; j < p; j++) {
                            out[rowOut + j] +=
                                    valA * bB.data[rowB + j * bCol];
                        }
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

}