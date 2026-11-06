package numj.core;

import java.util.Arrays;
import java.util.Objects;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class KernelBinary {

    private static final VectorSpecies<Float> SPECIES =
            FloatVector.SPECIES_PREFERRED;

    private static final int ADD = 0, SUB = 1, MUL = 2, DIV = 3;

    private static final int MODE_VV = 0, MODE_VS = 1, MODE_SV = 2, MODE_STRIDED = 3;

    private KernelBinary() {}


    public static NumJArray add(NumJArray a, NumJArray b) {
        return allocate(a, b, ADD);
    }

    public static NumJArray addi(NumJArray a, NumJArray dest, NumJArray b) {
        return compute(a, dest, b, ADD);
    }

    public static NumJArray sub(NumJArray a, NumJArray b) {
        return allocate(a, b, SUB);
    }

    public static NumJArray subi(NumJArray a, NumJArray dest, NumJArray b) {
        return compute(a, dest, b, SUB);
    }

    public static NumJArray mul(NumJArray a, NumJArray b) {
        return allocate(a, b, MUL);
    }

    public static NumJArray muli(NumJArray a, NumJArray dest, NumJArray b) {
        return compute(a, dest, b, MUL);
    }

    public static NumJArray div(NumJArray a, NumJArray b) {
        return allocate(a, b, DIV);
    }

    public static NumJArray divi(NumJArray a, NumJArray dest, NumJArray b) {
        return compute(a, dest, b, DIV);
    }


    private static NumJArray allocate(NumJArray a, NumJArray b, int op) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");

        int[] outShape = NumJArray.broadcastShape(a.shape, b.shape);
        int n = 1;
        for (int s : outShape) n *= s;

        NumJArray out = new NumJArray(new float[n], outShape);
        return compute(a, out, b, op);
    }

    private static NumJArray compute(
            NumJArray a, NumJArray dest, NumJArray b, int op) {

        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(dest, "dest");

        int[] outShape = NumJArray.broadcastShape(a.shape, b.shape);
        if (!Arrays.equals(outShape, dest.shape)) {
            throw new IllegalArgumentException(
                    "Destination shape " + Arrays.toString(dest.shape)
                    + " does not match broadcast result shape "
                    + Arrays.toString(outShape));
        }

        int nd = outShape.length;
        int n = 1;
        for (int s : outShape) n *= s;
        if (n == 0) return dest;

        int[] sa = NumJArray.broadcastStrides(a.shape, a.strides, outShape);
        int[] sb = NumJArray.broadcastStrides(b.shape, b.strides, outShape);
        int[] sd = dest.strides;

        checkDestStrides(outShape, sd);
        checkExtent(a.data, a.offset, outShape, sa);
        checkExtent(b.data, b.offset, outShape, sb);
        checkExtent(dest.data, dest.offset, outShape, sd);
        checkAlias(a, sa, dest, outShape, sd);
        checkAlias(b, sb, dest, outShape, sd);

        float[] ad = a.data, bd = b.data, dd = dest.data;

        if (nd == 0) {
            dd[dest.offset] = scalar(op, ad[a.offset], bd[b.offset]);
            return dest;
        }

        if (isDense(outShape, sa) && isDense(outShape, sb) && isDense(outShape, sd)) {
            vv(op, ad, a.offset, bd, b.offset, dd, dest.offset, n);
            return dest;
        }

        int last = nd - 1;
        int len = outShape[last];
        int ia = sa[last], ib = sb[last], id = sd[last];

        int mode;
        if (id == 1 && ia == 1 && ib == 1) mode = MODE_VV;
        else if (id == 1 && ia == 1 && ib == 0) mode = MODE_VS;
        else if (id == 1 && ia == 0 && ib == 1) mode = MODE_SV;
        else mode = MODE_STRIDED;

        int rows = n / len;
        int[] idx = new int[last];
        int pa = a.offset, pb = b.offset, pd = dest.offset;

        for (int r = 0; r < rows; r++) {
            switch (mode) {
                case MODE_VV -> vv(op, ad, pa, bd, pb, dd, pd, len);
                case MODE_VS -> vs(op, ad, pa, bd[pb], dd, pd, len);
                case MODE_SV -> sv(op, ad[pa], bd, pb, dd, pd, len);
                default -> strided(op, ad, pa, ia, bd, pb, ib, dd, pd, id, len);
            }

            for (int d = last - 1; d >= 0; d--) {
                idx[d]++;
                pa += sa[d];
                pb += sb[d];
                pd += sd[d];
                if (idx[d] < outShape[d]) break;
                pa -= sa[d] * outShape[d];
                pb -= sb[d] * outShape[d];
                pd -= sd[d] * outShape[d];
                idx[d] = 0;
            }
        }

        return dest;
    }


    private static void vv(
            int op, float[] a, int ao, float[] b, int bo,
            float[] d, int dof, int n) {

        int i = 0;
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();

        switch (op) {
            case ADD -> {
                for (; i < bound; i += step) {
                    FloatVector x = FloatVector.fromArray(SPECIES, a, ao + i);
                    FloatVector y = FloatVector.fromArray(SPECIES, b, bo + i);
                    x.add(y).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a[ao + i] + b[bo + i];
            }
            case SUB -> {
                for (; i < bound; i += step) {
                    FloatVector x = FloatVector.fromArray(SPECIES, a, ao + i);
                    FloatVector y = FloatVector.fromArray(SPECIES, b, bo + i);
                    x.sub(y).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a[ao + i] - b[bo + i];
            }
            case MUL -> {
                for (; i < bound; i += step) {
                    FloatVector x = FloatVector.fromArray(SPECIES, a, ao + i);
                    FloatVector y = FloatVector.fromArray(SPECIES, b, bo + i);
                    x.mul(y).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a[ao + i] * b[bo + i];
            }
            case DIV -> {
                for (; i < bound; i += step) {
                    FloatVector x = FloatVector.fromArray(SPECIES, a, ao + i);
                    FloatVector y = FloatVector.fromArray(SPECIES, b, bo + i);
                    x.div(y).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a[ao + i] / b[bo + i];
            }
            default -> throw new AssertionError("Unknown operation");
        }
    }

    private static void vs(
            int op, float[] a, int ao, float b,
            float[] d, int dof, int n) {

        int i = 0;
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();
        FloatVector s = FloatVector.broadcast(SPECIES, b);

        switch (op) {
            case ADD -> {
                for (; i < bound; i += step) {
                    FloatVector x = FloatVector.fromArray(SPECIES, a, ao + i);
                    x.add(s).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a[ao + i] + b;
            }
            case SUB -> {
                for (; i < bound; i += step) {
                    FloatVector x = FloatVector.fromArray(SPECIES, a, ao + i);
                    x.sub(s).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a[ao + i] - b;
            }
            case MUL -> {
                for (; i < bound; i += step) {
                    FloatVector x = FloatVector.fromArray(SPECIES, a, ao + i);
                    x.mul(s).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a[ao + i] * b;
            }
            case DIV -> {
                for (; i < bound; i += step) {
                    FloatVector x = FloatVector.fromArray(SPECIES, a, ao + i);
                    x.div(s).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a[ao + i] / b;
            }
            default -> throw new AssertionError("Unknown operation");
        }
    }

    private static void sv(
            int op, float a, float[] b, int bo,
            float[] d, int dof, int n) {

        int i = 0;
        int bound = SPECIES.loopBound(n);
        int step = SPECIES.length();
        FloatVector s = FloatVector.broadcast(SPECIES, a);

        switch (op) {
            case ADD -> {
                for (; i < bound; i += step) {
                    FloatVector y = FloatVector.fromArray(SPECIES, b, bo + i);
                    s.add(y).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a + b[bo + i];
            }
            case SUB -> {
                for (; i < bound; i += step) {
                    FloatVector y = FloatVector.fromArray(SPECIES, b, bo + i);
                    s.sub(y).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a - b[bo + i];
            }
            case MUL -> {
                for (; i < bound; i += step) {
                    FloatVector y = FloatVector.fromArray(SPECIES, b, bo + i);
                    s.mul(y).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a * b[bo + i];
            }
            case DIV -> {
                for (; i < bound; i += step) {
                    FloatVector y = FloatVector.fromArray(SPECIES, b, bo + i);
                    s.div(y).intoArray(d, dof + i);
                }
                for (; i < n; i++) d[dof + i] = a / b[bo + i];
            }
            default -> throw new AssertionError("Unknown operation");
        }
    }

    private static void strided(
            int op, float[] a, int ao, int sa, float[] b, int bo, int sb,
            float[] d, int dof, int sd, int n) {

        switch (op) {
            case ADD -> {
                for (int i = 0; i < n; i++)
                    d[dof + i * sd] = a[ao + i * sa] + b[bo + i * sb];
            }
            case SUB -> {
                for (int i = 0; i < n; i++)
                    d[dof + i * sd] = a[ao + i * sa] - b[bo + i * sb];
            }
            case MUL -> {
                for (int i = 0; i < n; i++)
                    d[dof + i * sd] = a[ao + i * sa] * b[bo + i * sb];
            }
            case DIV -> {
                for (int i = 0; i < n; i++)
                    d[dof + i * sd] = a[ao + i * sa] / b[bo + i * sb];
            }
            default -> throw new AssertionError("Unknown operation");
        }
    }

    private static float scalar(int op, float x, float y) {
        return switch (op) {
            case ADD -> x + y;
            case SUB -> x - y;
            case MUL -> x * y;
            case DIV -> x / y;
            default -> throw new AssertionError("Unknown operation");
        };
    }


    
    private static boolean isDense(int[] shape, int[] strides) {
        int expected = 1;
        for (int d = shape.length - 1; d >= 0; d--) {
            if (shape[d] > 1) {
                if (strides[d] != expected) return false;
                expected *= shape[d];
            }
        }
        return true;
    }

    private static void checkDestStrides(int[] shape, int[] strides) {
        for (int d = 0; d < shape.length; d++) {
            if (shape[d] > 1 && strides[d] == 0) {
                throw new IllegalArgumentException(
                        "Destination has a zero stride (self-overlapping view)");
            }
        }
    }

    private static int lowest(int offset, int[] shape, int[] strides) {
        int lo = offset;
        for (int d = 0; d < shape.length; d++) {
            if (shape[d] > 1 && strides[d] < 0) lo += strides[d] * (shape[d] - 1);
        }
        return lo;
    }

    private static int highest(int offset, int[] shape, int[] strides) {
        int hi = offset;
        for (int d = 0; d < shape.length; d++) {
            if (shape[d] > 1 && strides[d] > 0) hi += strides[d] * (shape[d] - 1);
        }
        return hi;
    }

    private static void checkExtent(
            float[] data, int offset, int[] shape, int[] strides) {

        int lo = lowest(offset, shape, strides);
        int hi = highest(offset, shape, strides);
        Objects.checkIndex(lo, data.length);
        Objects.checkIndex(hi, data.length);
    }

    private static void checkAlias(
            NumJArray in, int[] inStrides,
            NumJArray dest, int[] shape, int[] destStrides) {

        if (in.data != dest.data) return;

        if (in.offset == dest.offset) {
            boolean same = true;
            for (int d = 0; d < shape.length; d++) {
                if (shape[d] > 1 && inStrides[d] != destStrides[d]) {
                    same = false;
                    break;
                }
            }
            if (same) return;
        }

        int inLo = lowest(in.offset, shape, inStrides);
        int inHi = highest(in.offset, shape, inStrides);
        int dLo = lowest(dest.offset, shape, destStrides);
        int dHi = highest(dest.offset, shape, destStrides);

        if (inHi < dLo || dHi < inLo) return;

        throw new IllegalArgumentException("Destination overlaps an input with a different memory layout");
    }
}