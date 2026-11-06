package numj.math;

import java.util.Arrays;
import java.util.function.IntConsumer;

import numj.core.NumJArray;
import numj.core.Validator;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import java.util.Arrays;
import java.util.stream.IntStream;

public class lina {

    private static final VectorSpecies<Float> SP = FloatVector.SPECIES_PREFERRED;
    private static final int VL = SP.length();
 
    // Cache blocks (floats). Packed B panel = KC*NC*4 bytes = 256 KB.
    private static final int MC = 64;
    private static final int KC = 256;
    private static final int NC = 256;
 
    // Below this many multiply-adds, threading overhead isn't worth it.
    private static final long PARALLEL_THRESHOLD = 1L << 18;
 
    // Row counts at or below this use the skinny (matrix-vector style) paths.
    private static final int SKINNY_M = 4;
 
    private static final int MODE_BLOCKED = 0;
    private static final int MODE_SKINNY_STREAM = 1; // B rows contiguous (bCol == 1)
    private static final int MODE_SKINNY_DOT = 2;    // B columns contiguous (bRow == 1), A rows contiguous

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
 
    final int aRow = aB.strides[nBatch];
    final int aCol = aB.strides[nBatch + 1];
    final int bRow = bB.strides[nBatch];
    final int bCol = bB.strides[nBatch + 1];
 
    // Precompute per-batch offsets so batches/blocks can run independently.
    final int[] offA = new int[count];
    final int[] offB = new int[count];
    int[] idx = new int[nBatch];
    int baseA = aB.offset;
    int baseB = bB.offset;
    for (int n = 0; n < count; n++) {
        offA[n] = baseA;
        offB[n] = baseB;
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
 
    final float[] ad = aB.data;
    final float[] bd = bB.data;
    final int fm = m, fk = k, fp = p;
 
    // Pick a kernel.
    final int mode;
    if (m <= SKINNY_M && bCol == 1) {
        mode = MODE_SKINNY_STREAM;
    } else if (m <= SKINNY_M && bRow == 1 && aCol == 1) {
        mode = MODE_SKINNY_DOT;
    } else {
        mode = MODE_BLOCKED;
    }
 
    final int chunk = skinnyChunk(p);
    final int colBlocks = (p + chunk - 1) / chunk;
    final int rowBlocks = (m + MC - 1) / MC;
    final int perBatch = (mode == MODE_BLOCKED) ? rowBlocks : colBlocks;
    final int totalTasks = count * perBatch;
 
    // One task = one (batch, block). Tasks write disjoint output regions -> no races.
    IntConsumer task = t -> {
        int n = t / perBatch;
        int r = t % perBatch;
        int outBase = n * fm * fp;
        switch (mode) {
            case MODE_SKINNY_STREAM: {
                int j0 = r * chunk;
                int j1 = Math.min(j0 + chunk, fp);
                skinnyStreamKernel(ad, offA[n], aRow, aCol, bd, offB[n], bRow,
                                   out, outBase, fm, fk, fp, j0, j1);
                break;
            }
            case MODE_SKINNY_DOT: {
                int j0 = r * chunk;
                int j1 = Math.min(j0 + chunk, fp);
                skinnyDotKernel(ad, offA[n], aRow, bd, offB[n], bCol,
                                out, outBase, fm, fk, fp, j0, j1);
                break;
            }
            default: {
                int i0 = r * MC;
                int i1 = Math.min(i0 + MC, fm);
                matMulKernel(ad, offA[n], aRow, aCol, bd, offB[n], bRow, bCol,
                             out, outBase, i0, i1, fk, fp);
            }
        }
    };
 
    boolean parallel = totalTasks > 1 && (long) count * m * k * p >= PARALLEL_THRESHOLD;
    if (parallel) {
        IntStream.range(0, totalTasks).parallel().forEach(task);
    } else {
        for (int t = 0; t < totalTasks; t++) task.accept(t);
    }
 
    int[] outShape = new int[nBatch + (aVec ? 0 : 1) + (bVec ? 0 : 1)];
    System.arraycopy(batch, 0, outShape, 0, nBatch);
    int pos = nBatch;
    if (!aVec) outShape[pos++] = m;
    if (!bVec) outShape[pos] = p;
 
    return new NumJArray(out, outShape);
}
 
// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
 
/** Column chunk size for skinny paths: ~2 tasks per thread, multiple of VL, 512..2048 floats. */
private static int skinnyChunk(int p) {
    int threads = Math.max(1, Runtime.getRuntime().availableProcessors());
    int c = (p + threads * 2 - 1) / (threads * 2);
    c = ((c + VL - 1) / VL) * VL;
    //System.out.println("test skinny chunk: ");
    return Math.max(512, Math.min(2048, c));
}
 
// ---------------------------------------------------------------------------
// Kernel 1: blocked + vectorized (general case, larger m)
// out[i0:i1, :] += A[i0:i1, :] * B
// ---------------------------------------------------------------------------
private static void matMulKernel(
        float[] ad, int aBase, int aRow, int aCol,
        float[] bd, int bBase, int bRow, int bCol,
        float[] out, int outBase, int i0, int i1, int k, int p) {
    //System.out.println("test matmul kernel");
 
    // SIMD needs contiguous B rows. If B is strided/transposed, pack each panel.
    boolean needPack = bCol != 1;
    float[] pack = needPack ? new float[Math.min(KC, k) * Math.min(NC, p)] : null;
 
    for (int jc = 0; jc < p; jc += NC) {
        int nc = Math.min(NC, p - jc);
 
        for (int kc = 0; kc < k; kc += KC) {
            int kb = Math.min(KC, k - kc);
 
            float[] bs;
            int bOff, bStride;
            if (needPack) {
                for (int kk = 0; kk < kb; kk++) {
                    int src = bBase + (kc + kk) * bRow + jc * bCol;
                    int dst = kk * nc;
                    for (int j = 0; j < nc; j++) {
                        pack[dst + j] = bd[src + j * bCol];
                    }
                }
                bs = pack;
                bOff = 0;
                bStride = nc;
            } else {
                bs = bd;
                bOff = bBase + kc * bRow + jc;
                bStride = bRow;
            }
 
            for (int i = i0; i < i1; i++) {
                int aOff = aBase + i * aRow + kc * aCol;
                int oOff = outBase + i * p + jc;
                int j = 0;
 
                // 4 vector accumulators held in registers across the whole k-block
                for (; j + 4 * VL <= nc; j += 4 * VL) {
                    FloatVector c0 = FloatVector.fromArray(SP, out, oOff + j);
                    FloatVector c1 = FloatVector.fromArray(SP, out, oOff + j + VL);
                    FloatVector c2 = FloatVector.fromArray(SP, out, oOff + j + 2 * VL);
                    FloatVector c3 = FloatVector.fromArray(SP, out, oOff + j + 3 * VL);
                    for (int kk = 0; kk < kb; kk++) {
                        FloatVector va = FloatVector.broadcast(SP, ad[aOff + kk * aCol]);
                        int bo = bOff + kk * bStride + j;
                        c0 = FloatVector.fromArray(SP, bs, bo).fma(va, c0);
                        c1 = FloatVector.fromArray(SP, bs, bo + VL).fma(va, c1);
                        c2 = FloatVector.fromArray(SP, bs, bo + 2 * VL).fma(va, c2);
                        c3 = FloatVector.fromArray(SP, bs, bo + 3 * VL).fma(va, c3);
                    }
                    c0.intoArray(out, oOff + j);
                    c1.intoArray(out, oOff + j + VL);
                    c2.intoArray(out, oOff + j + 2 * VL);
                    c3.intoArray(out, oOff + j + 3 * VL);
                }
 
                for (; j + VL <= nc; j += VL) {
                    FloatVector c = FloatVector.fromArray(SP, out, oOff + j);
                    for (int kk = 0; kk < kb; kk++) {
                        FloatVector va = FloatVector.broadcast(SP, ad[aOff + kk * aCol]);
                        c = FloatVector.fromArray(SP, bs, bOff + kk * bStride + j).fma(va, c);
                    }
                    c.intoArray(out, oOff + j);
                }
 
                for (; j < nc; j++) {  // scalar tail
                    float s = out[oOff + j];
                    for (int kk = 0; kk < kb; kk++) {
                        s += ad[aOff + kk * aCol] * bs[bOff + kk * bStride + j];
                    }
                    out[oOff + j] = s;
                }
            }
        }
    }
}
 
// ---------------------------------------------------------------------------
// Kernel 2: skinny, streaming over contiguous B rows (bCol == 1)
// out[0:m, j0:j1] += A[0:m, :] * B[:, j0:j1]
// ---------------------------------------------------------------------------
private static void skinnyStreamKernel(
        float[] ad, int aBase, int aRow, int aCol,
        float[] bd, int bBase, int bRow,
        float[] out, int outBase, int m, int k, int p, int j0, int j1) {
    //System.out.println("test skinny stream kernel");
 
    final int vEnd = j0 + ((j1 - j0) / VL) * VL;
 
    for (int i = 0; i < m; i++) {
        int aOff = aBase + i * aRow;
        int oOff = outBase + i * p;
 
        int kk = 0;
        for (; kk + 4 <= k; kk += 4) {
            float a0 = ad[aOff + kk * aCol];
            float a1 = ad[aOff + (kk + 1) * aCol];
            float a2 = ad[aOff + (kk + 2) * aCol];
            float a3 = ad[aOff + (kk + 3) * aCol];
            FloatVector va0 = FloatVector.broadcast(SP, a0);
            FloatVector va1 = FloatVector.broadcast(SP, a1);
            FloatVector va2 = FloatVector.broadcast(SP, a2);
            FloatVector va3 = FloatVector.broadcast(SP, a3);
            int b0 = bBase + kk * bRow;
            int b1 = b0 + bRow, b2 = b1 + bRow, b3 = b2 + bRow;
 
            int j = j0;
            for (; j < vEnd; j += VL) {
                FloatVector c = FloatVector.fromArray(SP, out, oOff + j);
                c = FloatVector.fromArray(SP, bd, b0 + j).fma(va0, c);
                c = FloatVector.fromArray(SP, bd, b1 + j).fma(va1, c);
                c = FloatVector.fromArray(SP, bd, b2 + j).fma(va2, c);
                c = FloatVector.fromArray(SP, bd, b3 + j).fma(va3, c);
                c.intoArray(out, oOff + j);
            }
            for (; j < j1; j++) {
                out[oOff + j] += a0 * bd[b0 + j] + a1 * bd[b1 + j]
                               + a2 * bd[b2 + j] + a3 * bd[b3 + j];
            }
        }
        for (; kk < k; kk++) {  // leftover k rows
            float a = ad[aOff + kk * aCol];
            FloatVector va = FloatVector.broadcast(SP, a);
            int b0 = bBase + kk * bRow;
            int j = j0;
            for (; j < vEnd; j += VL) {
                FloatVector.fromArray(SP, bd, b0 + j)
                           .fma(va, FloatVector.fromArray(SP, out, oOff + j))
                           .intoArray(out, oOff + j);
            }
            for (; j < j1; j++) out[oOff + j] += a * bd[b0 + j];
        }
    }
}
 
// ---------------------------------------------------------------------------
// Kernel 3: skinny, dot-product form for transposed B (bRow == 1) and
// contiguous A rows (aCol == 1). Each output is a contiguous dot product.
// out[0:m, j0:j1] += A[0:m, :] * B[:, j0:j1]
// ---------------------------------------------------------------------------
private static void skinnyDotKernel(
        float[] ad, int aBase, int aRow,
        float[] bd, int bBase, int bCol,
        float[] out, int outBase, int m, int k, int p, int j0, int j1) {
 
    //System.out.println("test skinny dot kernel");

    final int step = 2 * VL;
    final int kEnd = (k / step) * step;
 
    for (int i = 0; i < m; i++) {
        int aOff = aBase + i * aRow;
        int oOff = outBase + i * p;
 
        for (int j = j0; j < j1; j++) {
            int bOff = bBase + j * bCol;
 
            FloatVector acc0 = FloatVector.zero(SP);
            FloatVector acc1 = FloatVector.zero(SP);
            int kk = 0;
            for (; kk < kEnd; kk += step) {
                acc0 = FloatVector.fromArray(SP, ad, aOff + kk)
                                  .fma(FloatVector.fromArray(SP, bd, bOff + kk), acc0);
                acc1 = FloatVector.fromArray(SP, ad, aOff + kk + VL)
                                  .fma(FloatVector.fromArray(SP, bd, bOff + kk + VL), acc1);
            }
            float s = acc0.add(acc1).reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
            for (; kk < k; kk++) {
                s += ad[aOff + kk] * bd[bOff + kk];
            }
            out[oOff + j] += s;
        }
    }
}

}