
package numj.core;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.BinaryOperator;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

public final class KernelScalars {

    private static final int ACT_RELU       = 0;
    private static final int ACT_RELU_D     = 1;
    private static final int ACT_LEAKY      = 2;
    private static final int ACT_LEAKY_D    = 3;
    private static final int ACT_SIGMOID    = 4;
    private static final int ACT_SIGMOID_D  = 5;
    private static final int ACT_TANH       = 6;
    private static final int ACT_TANH_D     = 7;
    private static final int ACT_ELU        = 8;
    private static final int ACT_ELU_D      = 9;
    private static final int ACT_SOFTPLUS   = 10;
    private static final int ACT_SOFTPLUS_D = 11;
    private static final int ACT_SILU       = 12;
    private static final int ACT_SILU_D     = 13;
    private static final int ACT_GELU       = 14;
    private static final int ACT_GELU_D     = 15;

    private static final VectorSpecies<Float> SPECIES =
            FloatVector.SPECIES_PREFERRED;

    private KernelScalars() {}


    private static NumJArray create(NumJArray a) {
        return new NumJArray(new float[a.size], a.shape, a.size);
    }

    private static void check(NumJArray a, NumJArray dest) {
        Objects.requireNonNull(a);
        Objects.requireNonNull(dest);

        if (!a.isContiguous() || !dest.isContiguous()
                || a.size != dest.size) {
            throw new IllegalArgumentException(
                    "Kernel requires contiguous arrays of equal size");
        }
    }

    private static void checkPair(
            NumJArray a, NumJArray b, NumJArray dest) {

        check(a, dest);

        if (!b.isContiguous() || b.size != a.size) {
            throw new IllegalArgumentException(
                    "Inputs must be contiguous arrays of equal size");
        }
    }

    private static void checkBounds(
            float[] src, int srcOff,
            float[] dst, int dstOff, int n) {

        Objects.checkFromIndexSize(srcOff, n, src.length);
        Objects.checkFromIndexSize(dstOff, n, dst.length);
    }


    public static NumJArray abs(NumJArray a) {
        NumJArray out = create(a);
        absi(a, out);
        return out;
    }

    public static NumJArray absi(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::absi);
        check(a, dest);

        float[] src = a.data;
        float[] dst = dest.data;

        int srcOff = a.offset;
        int dstOff = dest.offset;
        int n = a.size;

        checkBounds(src, srcOff, dst, dstOff, n);

        int i = 0;
        int bound = SPECIES.loopBound(n);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(
                    SPECIES, src, srcOff + i);

            v.abs().intoArray(dst, dstOff + i);
        }

        for (; i < n; i++) {
            dst[dstOff + i] = Math.abs(src[srcOff + i]);
        }

        return dest;
    }


    public static NumJArray neg(NumJArray a) {
        NumJArray out = create(a);
        negi(a, out);
        return out;
    }

    public static NumJArray negi(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::negi);
        check(a, dest);

        float[] src = a.data;
        float[] dst = dest.data;

        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        int i = 0;
        int bound = SPECIES.loopBound(n);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(SPECIES, src, so + i);
            v.neg().intoArray(dst, d + i);
        }

        for (; i < n; i++) {
            dst[d + i] = -src[so + i];
        }

        return dest;
    }


    public static NumJArray sqrt(NumJArray a) {
        NumJArray out = create(a);
        sqrti(a, out);
        return out;
    }

    public static NumJArray sqrti(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::sqrti);
        check(a, dest);

        float[] src = a.data;
        float[] dst = dest.data;

        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        int i = 0;
        int bound = SPECIES.loopBound(n);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(SPECIES, src, so + i);
            v.sqrt().intoArray(dst, d + i);
        }

        for (; i < n; i++) {
            dst[d + i] = (float) Math.sqrt(src[so + i]);
        }

        return dest;
    }

    public static NumJArray pow(NumJArray a, float power) {
        NumJArray out = create(a);
        powi(a, out, power);
        return out;
    }

    public static NumJArray powi(NumJArray a, NumJArray dest, float power) {

        if (strided(a, dest)) return viaTemp(a, dest, (src, dst) -> powi(src, dst, power));
        check(a, dest);

        float[] src = a.data;
        float[] dst = dest.data;

        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++) {
            dst[d + i] = (float) Math.pow(src[so + i], power);
        }

        return dest;
    }

    public static NumJArray exp(NumJArray a) {
        NumJArray out = create(a);
        expi(a, out);
        return out;
    }

    public static NumJArray expi(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::expi);
        check(a, dest);

        float[] src = a.data;
        float[] dst = dest.data;

        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++) {
            dst[d + i] = (float) Math.exp(src[so + i]);
        }

        return dest;
    }

    public static NumJArray log(NumJArray a) {
        NumJArray out = create(a);
        logi(a, out);
        return out;
    }

    public static NumJArray logi(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::logi);
        check(a, dest);

        float[] src = a.data;
        float[] dst = dest.data;

        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++) {
            dst[d + i] = (float) Math.log(src[so + i]);
        }

        return dest;
    }

    public static NumJArray floor(NumJArray a) {
        NumJArray out = create(a);
        floori(a, out);
        return out;
    }

    public static NumJArray floori(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::floori);
        check(a, dest);

        float[] src = a.data;
        float[] dst = dest.data;

        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++) {
            dst[d + i] = (float) Math.floor(src[so + i]);
        }

        return dest;
    }

    public static NumJArray ceil(NumJArray a) {
        NumJArray out = create(a);
        ceili(a, out);
        return out;
    }

    public static NumJArray ceili(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::ceili);
        check(a, dest);

        float[] src = a.data;
        float[] dst = dest.data;

        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++) {
            dst[d + i] = (float) Math.ceil(src[so + i]);
        }

        return dest;
    }

    public static NumJArray round(NumJArray a) {
        NumJArray out = create(a);
        roundi(a, out);
        return out;
    }

    public static NumJArray roundi(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::roundi);
        check(a, dest);

        float[] src = a.data;
        float[] dst = dest.data;

        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++) {
            dst[d + i] = (float) Math.rint(src[so + i]);
        }

        return dest;
    }

    public static NumJArray sign(NumJArray a) {
        NumJArray out = create(a);
        signi(a, out);
        return out;
    }

    public static NumJArray signi(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::signi);
        check(a, dest);

        float[] src = a.data;
        float[] dst = dest.data;

        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++) {
            dst[d + i] = Math.signum(src[so + i]);
        }

        return dest;
    }

    public static NumJArray sin(NumJArray a) {
        NumJArray out = create(a);
        sini(a, out);
        return out;
    }

    public static NumJArray sini(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::sini);
        check(a, dest);
        float[] src = a.data, dst = dest.data;
        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++)
            dst[d + i] = (float) Math.sin(src[so + i]);

        return dest;
    }

    public static NumJArray cos(NumJArray a) {
        NumJArray out = create(a);
        cosi(a, out);
        return out;
    }

    public static NumJArray cosi(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::cosi);
        check(a, dest);
        float[] src = a.data, dst = dest.data;
        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++)
            dst[d + i] = (float) Math.cos(src[so + i]);

        return dest;
    }

    public static NumJArray tan(NumJArray a) {
        NumJArray out = create(a);
        tani(a, out);
        return out;
    }

    public static NumJArray tani(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::tani);
        check(a, dest);
        float[] src = a.data, dst = dest.data;
        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++)
            dst[d + i] = (float) Math.tan(src[so + i]);

        return dest;
    }

    public static NumJArray asin(NumJArray a) {
        NumJArray out = create(a);
        asini(a, out);
        return out;
    }

    public static NumJArray asini(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::asini);
        check(a, dest);
        float[] src = a.data, dst = dest.data;
        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++)
            dst[d + i] = (float) Math.asin(src[so + i]);

        return dest;
    }

    public static NumJArray acos(NumJArray a) {
        NumJArray out = create(a);
        acosi(a, out);
        return out;
    }

    public static NumJArray acosi(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::acosi);
        check(a, dest);
        float[] src = a.data, dst = dest.data;
        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++)
            dst[d + i] = (float) Math.acos(src[so + i]);

        return dest;
    }

    public static NumJArray atan(NumJArray a) {
        NumJArray out = create(a);
        atani(a, out);
        return out;
    }

    public static NumJArray atani(NumJArray a, NumJArray dest) {
        if (strided(a, dest)) return viaTemp(a, dest, KernelScalars::atani);
        check(a, dest);
        float[] src = a.data, dst = dest.data;
        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++)
            dst[d + i] = (float) Math.atan(src[so + i]);

        return dest;
    }

    public static NumJArray atan2(NumJArray y, NumJArray x) {
        NumJArray out = create(y);
        atan2i(y, x, out);
        return out;
    }

    public static NumJArray atan2i(NumJArray y, NumJArray x, NumJArray dest) {

        if (!y.isContiguous() || !x.isContiguous() || !dest.isContiguous()) {
            Objects.requireNonNull(y);
            Objects.requireNonNull(x);
            Objects.requireNonNull(dest);
            if (y.size != dest.size || x.size != y.size) {
                throw new IllegalArgumentException("Arrays must have equal size");
            }
            NumJArray out = create(y);
            atan2i(compact(y), compact(x), out);
            scatter(out.data, dest);
            return dest;
        }
        checkPair(y, x, dest);

        float[] yd = y.data, xd = x.data, dst = dest.data;
        int yo = y.offset, xo = x.offset, d = dest.offset, n = y.size;

        checkBounds(yd, yo, dst, d, n);
        Objects.checkFromIndexSize(xo, n, xd.length);

        for (int i = 0; i < n; i++) {
            dst[d + i] = (float) Math.atan2(
                    yd[yo + i], xd[xo + i]);
        }

        return dest;
    }

    public static NumJArray add(NumJArray a, float b) {
        NumJArray out = create(a);
        addi(a, out, b);
        return out;
    }

    public static NumJArray addi(NumJArray a, NumJArray dest, float b) {

        return arithmeticAny(a, dest, b, 0);   // was: check(a, dest); return arithmetic(a, dest, b, 0);
    }

    public static NumJArray sub(NumJArray a, float b) {
        NumJArray out = create(a);
        subi(a, out, b);
        return out;
    }

    public static NumJArray subi(NumJArray a, NumJArray dest, float b) {

        return arithmeticAny(a, dest, b, 1);   // was: check(a, dest); return arithmetic(a, dest, b, 1);
    }

    public static NumJArray mul(NumJArray a, float b) {
        NumJArray out = create(a);
        muli(a, out, b);
        return out;
    }

    public static NumJArray muli(NumJArray a, NumJArray dest, float b) {

        return arithmeticAny(a, dest, b, 2);   // was: check(a, dest); return arithmetic(a, dest, b, 2);
    }

    public static NumJArray div(NumJArray a, float b) {
        NumJArray out = create(a);
        divi(a, out, b);
        return out;
    }

    public static NumJArray divi(NumJArray a, NumJArray dest, float b) {

        return arithmeticAny(a, dest, b, 3);   // was: check(a, dest); return arithmetic(a, dest, b, 3);
    }

    private static NumJArray arithmeticAny(NumJArray a, NumJArray dest, float b, int op) {

        if (a.isContiguous() && dest.isContiguous()) {
            check(a, dest);
            return arithmetic(a, dest, b, op);
        }

        Objects.requireNonNull(a);
        Objects.requireNonNull(dest);

        if (!Arrays.equals(a.shape, dest.shape)) {
            throw new IllegalArgumentException("Shapes must match for strided kernel");
        }

        // Element-wise in place (same buffer, same layout) is safe; other sharing is not.
        if (a != dest && a.data == dest.data
                && (a.offset != dest.offset
                    || !Arrays.equals(a.strides, dest.strides))) {
            throw new IllegalArgumentException(
                    "Destination overlaps input with a different layout");
        }

        return arithmeticStrided(a, dest, b, op);
    }

    private static NumJArray arithmeticStrided(
            NumJArray a, NumJArray dest, float b, int op) {

        int[] shape = a.shape;
        int nd = shape.length;
        int n = 1;
        for (int s : shape) n *= s;
        if (n == 0) return dest;

        float[] src = a.data, dst = dest.data;

        if (nd == 0) {
            float x = src[a.offset];
            dst[dest.offset] = switch (op) {
                case 0 -> x + b;
                case 1 -> x - b;
                case 2 -> x * b;
                case 3 -> x / b;
                default -> throw new AssertionError("Unknown operation");
            };
            return dest;
        }

        int last = nd - 1;
        int len = shape[last];
        int sa = a.strides[last], sd = dest.strides[last];
        int rows = n / len;
        int[] idx = new int[last];
        int pa = a.offset, pd = dest.offset;

        for (int r = 0; r < rows; r++) {
            stridedRow(op, src, pa, sa, dst, pd, sd, len, b);

            for (int d = last - 1; d >= 0; d--) {
                idx[d]++;
                pa += a.strides[d];
                pd += dest.strides[d];
                if (idx[d] < shape[d]) break;
                pa -= a.strides[d] * shape[d];
                pd -= dest.strides[d] * shape[d];
                idx[d] = 0;
            }
        }
        return dest;
    }

    private static void stridedRow(
            int op, float[] src, int so, int ss,
            float[] dst, int d, int sd, int n, float b) {

        switch (op) {
            case 0 -> { for (int i = 0; i < n; i++) dst[d + i * sd] = src[so + i * ss] + b; }
            case 1 -> { for (int i = 0; i < n; i++) dst[d + i * sd] = src[so + i * ss] - b; }
            case 2 -> { for (int i = 0; i < n; i++) dst[d + i * sd] = src[so + i * ss] * b; }
            case 3 -> { for (int i = 0; i < n; i++) dst[d + i * sd] = src[so + i * ss] / b; }
            default -> throw new AssertionError("Unknown operation");
        }
    }

    private static boolean strided(NumJArray a, NumJArray dest) {
        return !a.isContiguous() || !dest.isContiguous();
    }

    private static NumJArray compact(NumJArray a) {
        int n = a.size;
        float[] out = new float[n];

        if (n > 0) {
            int[] shape = a.shape, strides = a.strides;
            int nd = shape.length;
            int[] idx = new int[nd];
            int p = a.offset;

            for (int i = 0; i < n; i++) {
                out[i] = a.data[p];
                for (int d = nd - 1; d >= 0; d--) {
                    idx[d]++;
                    p += strides[d];
                    if (idx[d] < shape[d]) break;
                    p -= strides[d] * shape[d];
                    idx[d] = 0;
                }
            }
        }
        return new NumJArray(out, a.shape, n);
    }

    private static void scatter(float[] src, NumJArray dest) {
        int n = dest.size;
        if (n == 0) return;

        int[] shape = dest.shape, strides = dest.strides;
        int nd = shape.length;
        int[] idx = new int[nd];
        int p = dest.offset;

        for (int i = 0; i < n; i++) {
            dest.data[p] = src[i];
            for (int d = nd - 1; d >= 0; d--) {
                idx[d]++;
                p += strides[d];
                if (idx[d] < shape[d]) break;
                p -= strides[d] * shape[d];
                idx[d] = 0;
            }
        }
    }

    private static NumJArray viaTemp(
            NumJArray a, NumJArray dest, BinaryOperator<NumJArray> kernel) {

        Objects.requireNonNull(a);
        Objects.requireNonNull(dest);

        if (a.size != dest.size) {
            throw new IllegalArgumentException("Arrays must have equal size");
        }

        NumJArray in = compact(a);
        NumJArray out = create(a);
        kernel.apply(in, out);         
        scatter(out.data, dest);
        return dest;
    }

    private static NumJArray arithmetic(
            NumJArray a, NumJArray dest, float b, int operation) {

        float[] src = a.data, dst = dest.data;
        int so = a.offset, d = dest.offset, n = a.size;

        checkBounds(src, so, dst, d, n);

        int i = 0;
        int bound = SPECIES.loopBound(n);

        FloatVector scalar = FloatVector.broadcast(SPECIES, b);

        switch (operation) {
            case 0 -> {
                for (; i < bound; i += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, src, so + i);
                    v.add(scalar).intoArray(dst, d + i);
                }
                for (; i < n; i++) dst[d + i] = src[so + i] + b;
            }
            case 1 -> {
                for (; i < bound; i += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, src, so + i);
                    v.sub(scalar).intoArray(dst, d + i);
                }
                for (; i < n; i++) dst[d + i] = src[so + i] - b;
            }
            case 2 -> {
                for (; i < bound; i += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, src, so + i);
                    v.mul(scalar).intoArray(dst, d + i);
                }
                for (; i < n; i++) dst[d + i] = src[so + i] * b;
            }
            case 3 -> {
                for (; i < bound; i += SPECIES.length()) {
                    FloatVector v = FloatVector.fromArray(SPECIES, src, so + i);
                    v.div(scalar).intoArray(dst, d + i);
                }
                for (; i < n; i++) dst[d + i] = src[so + i] / b;
            }
            default -> throw new AssertionError("Unknown operation");
        }

        return dest;
    }


    public static NumJArray clip(
            NumJArray a, float min, float max) {

        NumJArray out = create(a);
        clipi(a, out, min, max);
        return out;
    }

    public static NumJArray clipi(
            NumJArray a, NumJArray dest, float min, float max) {

        if (min > max) {
            throw new IllegalArgumentException("min must be <= max");
        }

        check(a, dest);

        float[] src = a.data, dst = dest.data;
        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        int i = 0;
        int bound = SPECIES.loopBound(n);

        FloatVector vMin = FloatVector.broadcast(SPECIES, min);
        FloatVector vMax = FloatVector.broadcast(SPECIES, max);

        for (; i < bound; i += SPECIES.length()) {
            FloatVector v = FloatVector.fromArray(SPECIES, src, so + i);
            v.max(vMin).min(vMax).intoArray(dst, d + i);
        }

        for (; i < n; i++) {
            dst[d + i] = Math.max(min, Math.min(max, src[so + i]));
        }

        return dest;
    }

    public static NumJArray mod(NumJArray a, float b) {
        NumJArray out = create(a);
        modi(a, out, b);
        return out;
    }

    public static NumJArray modi(
            NumJArray a, NumJArray dest, float b) {

        check(a, dest);

        float[] src = a.data, dst = dest.data;
        int so = a.offset, d = dest.offset, n = a.size;
        checkBounds(src, so, dst, d, n);

        for (int i = 0; i < n; i++) {
            float x = src[so + i];
            float r = x % b;

            dst[d + i] = (r != 0 && (r < 0) != (b < 0))
                    ? r + b
                    : r;
        }

        return dest;
    }

    private static final float GELU_C = 0.7978845608f;
    private static final float GELU_K = 0.044715f;

    public static NumJArray relu(NumJArray a) {
        NumJArray out = create(a);
        relui(a, out);
        return out;
    }

    public static NumJArray relui(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_RELU, 0f);
    }

    public static NumJArray reluD(NumJArray a) {
        NumJArray out = create(a);
        reluDi(a, out);
        return out;
    }

    public static NumJArray reluDi(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_RELU_D, 0f);
    }

    public static NumJArray leakyRelu(NumJArray a, float alpha) {
        NumJArray out = create(a);
        leakyRelui(a, out, alpha);
        return out;
    }

    public static NumJArray leakyRelui(NumJArray a, NumJArray dest, float alpha) {
        return activationKernel(a, dest, ACT_LEAKY, alpha);
    }

    public static NumJArray leakyReluD(NumJArray a, float alpha) {
        NumJArray out = create(a);
        leakyReluDi(a, out, alpha);
        return out;
    }

    public static NumJArray leakyReluDi(
            NumJArray a, NumJArray dest, float alpha) {
        return activationKernel(a, dest, ACT_LEAKY_D, alpha);
    }

    public static NumJArray sigmoid(NumJArray a) {
        NumJArray out = create(a);
        sigmoidi(a, out);
        return out;
    }

    public static NumJArray sigmoidi(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_SIGMOID, 0f);
    }

    public static NumJArray sigmoidD(NumJArray a) {
        NumJArray out = create(a);
        sigmoidDi(a, out);
        return out;
    }

    public static NumJArray sigmoidDi(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_SIGMOID_D, 0f);
    }

    public static NumJArray tanh(NumJArray a) {
        NumJArray out = create(a);
        tanhi(a, out);
        return out;
    }

    public static NumJArray tanhi(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_TANH, 0f);
    }

    public static NumJArray tanhD(NumJArray a) {
        NumJArray out = create(a);
        tanhDi(a, out);
        return out;
    }

    public static NumJArray tanhDi(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_TANH_D, 0f);
    }

    public static NumJArray elu(NumJArray a, float alpha) {
        NumJArray out = create(a);
        elui(a, out, alpha);
        return out;
    }

    public static NumJArray elui(
            NumJArray a, NumJArray dest, float alpha) {
        return activationKernel(a, dest, ACT_ELU, alpha);
    }

    public static NumJArray eluD(NumJArray a, float alpha) {
        NumJArray out = create(a);
        eluDi(a, out, alpha);
        return out;
    }

    public static NumJArray eluDi(
            NumJArray a, NumJArray dest, float alpha) {
        return activationKernel(a, dest, ACT_ELU_D, alpha);
    }

    public static NumJArray softplus(NumJArray a) {
        NumJArray out = create(a);
        softplusi(a, out);
        return out;
    }

    public static NumJArray softplusi(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_SOFTPLUS, 0f);
    }

    public static NumJArray softplusD(NumJArray a) {
        NumJArray out = create(a);
        softplusDi(a, out);
        return out;
    }

    public static NumJArray softplusDi(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_SOFTPLUS_D, 0f);
    }

    public static NumJArray silu(NumJArray a) {
        NumJArray out = create(a);
        silui(a, out);
        return out;
    }

    public static NumJArray silui(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_SILU, 0f);
    }

    public static NumJArray siluD(NumJArray a) {
        NumJArray out = create(a);
        siluDi(a, out);
        return out;
    }

    public static NumJArray siluDi(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_SILU_D, 0f);
    }

    public static NumJArray gelu(NumJArray a) {
        NumJArray out = create(a);
        gelui(a, out);
        return out;
    }

    public static NumJArray gelui(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_GELU, 0f);
    }

    public static NumJArray geluD(NumJArray a) {
        NumJArray out = create(a);
        geluDi(a, out);
        return out;
    }

    public static NumJArray geluDi(NumJArray a, NumJArray dest) {
        return activationKernel(a, dest, ACT_GELU_D, 0f);
    }


    private static NumJArray activationKernel(NumJArray a, NumJArray dest, int operation, float alpha) {

        if (strided(a, dest)) {
            return viaTemp(a, dest,
                    (src, dst) -> activationKernel(
                            src, dst, operation, alpha));
        }

        check(a, dest);

        float[] src = a.data;
        float[] dst = dest.data;

        int so = a.offset;
        int d = dest.offset;
        int n = a.size;

        checkBounds(src, so, dst, d, n);

        int i = 0;
        int bound = SPECIES.loopBound(n);

        FloatVector zero = FloatVector.zero(SPECIES);
        FloatVector one = FloatVector.broadcast(SPECIES, 1f);
        FloatVector half = FloatVector.broadcast(SPECIES, 0.5f);
        FloatVector alphaV = FloatVector.broadcast(SPECIES, alpha);

        for (; i < bound; i += SPECIES.length()) {

            FloatVector x = FloatVector.fromArray(
                    SPECIES, src, so + i);

            FloatVector result;

            switch (operation) {

                case ACT_RELU -> {
                    result = x.max(zero);
                }

                case ACT_RELU_D -> {
                    result = zero.blend(
                    one,
                    x.compare(VectorOperators.GT, zero));
                }

                case ACT_LEAKY -> {
                    VectorMask<Float> positive =
                            x.compare(VectorOperators.GE, zero);

                    result = x.mul(alphaV).blend(x, positive);
                }

                case ACT_LEAKY_D -> {
                    VectorMask<Float> positive =
                            x.compare(VectorOperators.GE, zero);

                    result = alphaV.blend(one, positive);
                }

                case ACT_SIGMOID -> {
                    FloatVector e = x.abs().neg().lanewise(VectorOperators.EXP);

                    FloatVector denominator = one.add(e);

                    FloatVector positive = one.div(denominator);
                    FloatVector negative = e.div(denominator);

                    result = positive.blend(negative, x.compare(VectorOperators.LT, zero));
                }

                case ACT_SIGMOID_D -> {
                    FloatVector e = x.abs().neg().lanewise(VectorOperators.EXP);

                    FloatVector denominator = one.add(e);

                    FloatVector s = one.div(denominator).blend(
                                    e.div(denominator),
                                    x.compare(VectorOperators.LT, zero));

                    result = s.mul(one.sub(s));
                }

                case ACT_TANH -> {
                    result = x.lanewise(VectorOperators.TANH);
                }

                case ACT_TANH_D -> {
                    FloatVector t = x.lanewise(VectorOperators.TANH);
                    result = one.sub(t.mul(t));
                }

                case ACT_ELU -> {
                    FloatVector negative = x.lanewise(VectorOperators.EXPM1).mul(alphaV);

                    result = negative.blend(x, x.compare(VectorOperators.GE, zero));
                }

                case ACT_ELU_D -> {
                    FloatVector negative = x.lanewise(VectorOperators.EXP).mul(alphaV);

                    result = negative.blend(one, x.compare(VectorOperators.GE, zero));
                }

                case ACT_SOFTPLUS -> {
                    FloatVector e = x.abs().neg()
                            .lanewise(VectorOperators.EXP);

                    result = x.max(zero).add(
                            e.lanewise(VectorOperators.LOG1P));
                }

                case ACT_SOFTPLUS_D -> {
                    FloatVector e = x.abs().neg()
                            .lanewise(VectorOperators.EXP);

                    FloatVector denominator = one.add(e);

                    result = one.div(denominator)
                            .blend(
                                    e.div(denominator),
                                    x.compare(VectorOperators.LT, zero));
                }

                case ACT_SILU -> {
                    FloatVector e = x.abs().neg()
                            .lanewise(VectorOperators.EXP);

                    FloatVector denominator = one.add(e);

                    FloatVector s = one.div(denominator)
                            .blend(
                                    e.div(denominator),
                                    x.compare(VectorOperators.LT, zero));

                    result = x.mul(s);
                }

                case ACT_SILU_D -> {
                    FloatVector e = x.abs().neg()
                            .lanewise(VectorOperators.EXP);

                    FloatVector denominator = one.add(e);

                    FloatVector s = one.div(denominator)
                            .blend(
                                    e.div(denominator),
                                    x.compare(VectorOperators.LT, zero));

                    result = s.add(x.mul(s).mul(one.sub(s)));
                }

                case ACT_GELU -> {
                    FloatVector x2 = x.mul(x);
                    FloatVector x3 = x2.mul(x);

                    FloatVector inner = x.add(
                            x3.mul(GELU_K)).mul(GELU_C);

                    FloatVector t = inner.lanewise(
                            VectorOperators.TANH);

                    result = x.mul(half).mul(one.add(t));
                }

                case ACT_GELU_D -> {
                    FloatVector x2 = x.mul(x);
                    FloatVector x3 = x2.mul(x);

                    FloatVector inner = x.add(
                            x3.mul(GELU_K)).mul(GELU_C);

                    FloatVector t = inner.lanewise(
                            VectorOperators.TANH);

                    FloatVector du = one.add(
                            x2.mul(3f * GELU_K)).mul(GELU_C);

                    result = half.mul(one.add(t)).add(
                            half.mul(x).mul(
                                    one.sub(t.mul(t))).mul(du));
                }

                default -> throw new AssertionError(
                        "Unknown activation operation: " + operation);
            }

            result.intoArray(dst, d + i);
        }

        // Scalar tail
        for (; i < n; i++) {
            dst[d + i] = activationScalar(
                    src[so + i], operation, alpha);
        }

        return dest;
    }

    private static float activationScalar(
            float x, int operation, float alpha) {

        return switch (operation) {

            case ACT_RELU ->
                    Math.max(0f, x);

            case ACT_RELU_D ->
                    x > 0f ? 1f : 0f;

            case ACT_LEAKY ->
                    x >= 0f ? x : alpha * x;

            case ACT_LEAKY_D ->
                    x >= 0f ? 1f : alpha;

            case ACT_SIGMOID -> {
                if (x >= 0f) {
                    yield 1f / (1f + (float) Math.exp(-x));
                }

                float e = (float) Math.exp(x);
                yield e / (1f + e);
            }

            case ACT_SIGMOID_D -> {
                float s;
                if (x >= 0f) {
                    s = 1f / (1f + (float) Math.exp(-x));
                } else {
                    float e = (float) Math.exp(x);
                    s = e / (1f + e);
                }
                yield s * (1f - s);
            }

            case ACT_TANH -> (float) Math.tanh(x);

            case ACT_TANH_D -> {
                float t = (float) Math.tanh(x);
                yield 1f - t * t;
            }

            case ACT_ELU ->
                    x >= 0f ? x : alpha * (float) Math.expm1(x);

            case ACT_ELU_D ->
                    x >= 0f ? 1f : alpha * (float) Math.exp(x);

            case ACT_SOFTPLUS ->
                    Math.max(x, 0f)
                            + (float) Math.log1p(
                                    Math.exp(-Math.abs(x)));

            case ACT_SOFTPLUS_D -> {
                if (x >= 0f) {
                    yield 1f / (1f + (float) Math.exp(-x));
                }
                float e = (float) Math.exp(x);
                yield e / (1f + e);
            }

            case ACT_SILU -> {
                float s;
                if (x >= 0f) {
                    s = 1f / (1f + (float) Math.exp(-x));
                } else {
                    float e = (float) Math.exp(x);
                    s = e / (1f + e);
                }
                yield x * s;
            }

            case ACT_SILU_D -> {
                float s;
                if (x >= 0f) {
                    s = 1f / (1f + (float) Math.exp(-x));
                } else {
                    float e = (float) Math.exp(x);
                    s = e / (1f + e);
                }
                yield s + x * s * (1f - s);
            }

            case ACT_GELU -> {
                float x3 = x * x * x;
                float t = (float) Math.tanh(
                        GELU_C * (x + GELU_K * x3));
                yield 0.5f * x * (1f + t);
            }

            case ACT_GELU_D -> {
                float x2 = x * x;
                float x3 = x2 * x;

                float u = GELU_C * (x + GELU_K * x3);
                float t = (float) Math.tanh(u);

                float du = GELU_C * (1f + 3f * GELU_K * x2);

                yield 0.5f * (1f + t)
                        + 0.5f * x * (1f - t * t) * du;
            }

            default -> throw new AssertionError(
                    "Unknown activation operation: " + operation);
        };
    }

}
