package numj.math;

import java.util.Arrays;

import numj.core.NumJArray;
import numj.core.Validator;

public class arithmatic {

    public static NumJArray map(NumJArray a, FloatUnaryOperator f) {
        if (f == null) {
            throw new IllegalArgumentException("f must not be null");
        }

        float[] out = a.toContiguousData();
        for (int i = 0; i < out.length; i++) {
            out[i] = f.applyAsFloat(out[i]);
        }

        return new NumJArray(out, a.shape);
    }

    public static NumJArray mapi(NumJArray a, FloatUnaryOperator f) {
        if (f == null) {
            throw new IllegalArgumentException("f must not be null");
        }

        for (int p : a.positions()) {
            a.data[p] = f.applyAsFloat(a.data[p]);
        }

        return a;
    }

    public static NumJArray abs(NumJArray a) {
        return map(a, Math::abs);
    }

    public static NumJArray neg(NumJArray a) {
        return map(a, x -> -x);
    }

    public static NumJArray sqrt(NumJArray a) {
        return map(a, x -> (float) Math.sqrt(x));
    }

    public static NumJArray pow(NumJArray a, float power) {
        return map(a, x -> (float) Math.pow(x, power));
    }

    public static NumJArray exp(NumJArray a) {
        return map(a, x -> (float) Math.exp(x));
    }

    public static NumJArray log(NumJArray a) {
        return map(a, x -> (float) Math.log(x));
    }

    public static NumJArray floor(NumJArray a) {
        return map(a, x -> (float) Math.floor(x));
    }

    public static NumJArray ceil(NumJArray a) {
        return map(a, x -> (float) Math.ceil(x));
    }

    public static NumJArray round(NumJArray a) {
        return map(a, x -> (float) Math.rint(x));
    }

    public static NumJArray sign(NumJArray a) {
        return map(a, Math::signum);
    }

    public static NumJArray sin(NumJArray a) {
        return map(a, x -> (float) Math.sin(x));
    }

    public static NumJArray cos(NumJArray a) {
        return map(a, x -> (float) Math.cos(x));
    }

    public static NumJArray tan(NumJArray a) {
        return map(a, x -> (float) Math.tan(x));
    }

    public static NumJArray asin(NumJArray a) {
        return map(a, x -> (float) Math.asin(x));
    }

    public static NumJArray acos(NumJArray a) {
        return map(a, x -> (float) Math.acos(x));
    }

    public static NumJArray atan(NumJArray a) {
        return map(a, x -> (float) Math.atan(x));
    }

    public static NumJArray atan2(NumJArray a) {
        return map(a, x -> (float) Math.atan2(x));
    }

    public static NumJArray add(NumJArray a, float b) {
        return map(a, x -> x + b);
    }

    public static NumJArray sub(NumJArray a, float b) {
        return map(a, x -> x - b);
    }

    public static NumJArray mul(NumJArray a, float b) {
        return map(a, x -> x * b);
    }

    public static NumJArray div(NumJArray a, float b) {
        return map(a, x -> x / b);
    }

    public static NumJArray clip(NumJArray a, float min, float max) {
        if (min > max) {
            throw new IllegalArgumentException("min must be <= max");
        }
        return map(a, x -> Math.max(min, Math.min(max, x)));
    }

    public static NumJArray mod(NumJArray a, float b) {
        return map(a, x -> {float r = x % b;return (r != 0 && (r < 0) != (b < 0)) ? r + b : r;});
    }

    public static NumJArray add(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> x + y);
    }

    public static NumJArray sub(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> x - y);
    }

    public static NumJArray mul(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> x * y);
    }

    public static NumJArray div(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> x / y);
    }

    public static float flag(boolean condition) {
        return (float) (condition ? 1.0 : 0.0);
    }

    public static NumJArray equal(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> flag(x == y));
    }

    public static NumJArray equal(NumJArray a, float b) {
        return map(a, x -> flag(x == b));
    }

    public static NumJArray notEqual(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> flag(x != y));
    }

    public static NumJArray notEqual(NumJArray a, float b) {
        return map(a, x -> flag(x != b));
    }

    public static NumJArray greater(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> flag(x > y));
    }

    public static NumJArray greater(NumJArray a, float b) {
        return map(a, x -> flag(x > b));
    }

    public static NumJArray greaterEqual(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> flag(x >= y));
    }

    public static NumJArray greaterEqual(NumJArray a, float b) {
        return map(a, x -> flag(x >= b));
    }

    public static NumJArray less(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> flag(x < y));
    }

    public static NumJArray less(NumJArray a, float b) {
        return map(a, x -> flag(x < b));
    }

    public static NumJArray lessEqual(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> flag(x <= y));
    }

    public static NumJArray lessEqual(NumJArray a, float b) {
        return map(a, x -> flag(x <= b));
    }

    public static NumJArray logicalAnd(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> flag(x != 0 && y != 0));
    }

    public static NumJArray logicalOr(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> flag(x != 0 || y != 0));
    }

    public static NumJArray logicalXor(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, (x, y) -> flag((x != 0) != (y != 0)));
    }

    public static NumJArray logicalNot(NumJArray a) {
        return map(a, x -> flag(x == 0));
    }

    public static boolean all(NumJArray a) {
        for (float v : a.readOnlyData()) {
            if (v == 0) return false;
        }
        return true;
    }

    public static boolean any(NumJArray a) {
        for (float v : a.readOnlyData()) {
            if (v != 0) return true;
        }
        return false;
    }

    public static NumJArray allAxis(NumJArray a, int axis) {
        return a.reduceAxis(axis, (acc, x) -> flag(acc != 0 && x != 0), 1.0f);
    }

    public static NumJArray anyAxis(NumJArray a, int axis) {
        return a.reduceAxis(axis, (acc, x) -> flag(acc != 0 || x != 0), 0.0f);
    }

    public static final float DEFAULT_RTOL = 1e-5f;

    public static final float DEFAULT_ATOL = 1e-8f;

    public static NumJArray isClose(NumJArray a, NumJArray b) {
        return a.isClose(b, DEFAULT_RTOL, DEFAULT_ATOL);
    }

    public static NumJArray isClose(NumJArray a, NumJArray b, float rtol, float atol) {
        NumJArray.checkTolerances(rtol, atol);
        return a.broadcastOp(b, (x, y) -> flag(NumJArray.closeEnough(x, y, rtol, atol)));
    }

    public static NumJArray isClose(NumJArray a, float b) {
        return isClose(a, b, DEFAULT_RTOL, DEFAULT_ATOL);
    }

    public static NumJArray isClose(NumJArray a, float b, float rtol, float atol) {
        NumJArray.checkTolerances(rtol, atol);
        return map(a, x -> flag(NumJArray.closeEnough(x, b, rtol, atol)));
    }

    public static boolean allClose(NumJArray a, NumJArray b) {
        return isClose(a, b).all();
    }

    public static boolean allClose(NumJArray a, NumJArray b, float rtol, float atol) {
        return isClose(a, b, rtol, atol).all();
    }

    public static int countNonzero(NumJArray array) {
        int count = 0;
        for (float v : array.readOnlyData()) {
            if (v != 0) count++;
        }
        return count;
    }

    public static NumJArray countNonzero(NumJArray a, int axis) {
        return map(a, x -> flag(x != 0)).sumAxis(axis);
    }

    public static NumJArray isNaN(NumJArray a) {
        return map(a, x -> flag(Float.isNaN(x)));
    }

    public static NumJArray isInf(NumJArray a) {
        return map(a, x -> flag(Float.isInfinite(x)));
    }

    public static NumJArray isFinite(NumJArray a) {
        return map(a, x -> flag(Float.isFinite(x)));
    }

    public static NumJArray maximum(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, Math::max);
    }

    public static NumJArray maximum(NumJArray a, float b) {
        return map(a, x -> Math.max(x, b));
    }

    public static NumJArray minimum(NumJArray a, NumJArray b) {
        return a.broadcastOp(b, Math::min);
    }

    public static NumJArray minimum(NumJArray a, float b) {
        return map(a, x -> Math.min(x, b));
    }

    public static boolean closeEnough(float x, float y, float rtol, float atol) {
        if (x == y) return true;
        if (Float.isInfinite(x) || Float.isInfinite(y)) return false;
        return Math.abs(x - y) <= atol + rtol * Math.abs(y);
    }

    public static void checkTolerances(float rtol, float atol) {
        if (rtol < 0 || atol < 0) {
            throw new IllegalArgumentException("rtol and atol must be non-negative");
        }
    }

    public static NumJArray elementwise(NumJArray a, NumJArray b, FloatBinaryOperator op) {
        NumJArray.requireOperator(op);
        return a.broadcastOp(b, op);
    }

    public static NumJArray elementwise(NumJArray a, float b, FloatBinaryOperator op) {
        NumJArray.requireOperator(op);
        return map(a, x -> op.applyAsFloat(x, b));
    }

    public static NumJArray elementwise(NumJArray a, FloatUnaryOperator op) {
        return map(a, op);
    }

    public static NumJArray elementwisei(NumJArray a, FloatUnaryOperator op) {
        if (op == null) {
            throw new IllegalArgumentException("op must not be null");
        }
        for (int p : a.positions()) a.data[p] = op.applyAsFloat(a.data[p]);
        return a;
    }

    public static NumJArray elementwisei(NumJArray a, float b, FloatBinaryOperator op) {
        NumJArray.requireOperator(op);
        for (int p : a.positions()) a.data[p] = op.applyAsFloat(a.data[p], b);
        return a;
    }

    public static NumJArray elementwisei(NumJArray a, NumJArray b, FloatBinaryOperator op) {
        NumJArray.requireOperator(op);
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }
        float[] other = b.broadcastTo(a.shape).toContiguousData();
        int[] pos = a.positions();
        for (int i = 0; i < pos.length; i++) {
            a.data[pos[i]] = op.applyAsFloat(a.data[pos[i]], other[i]);
        }
        return a;
    }

    public static NumJArray addi(NumJArray a, float b) {
        return elementwisei(a, b, (x, y) -> x + y);
    }

    public static NumJArray addi(NumJArray a, NumJArray b) {
        return elementwisei(a, b, (x, y) -> x + y);
    }

    public static NumJArray subi(NumJArray a, float b) {
        return elementwisei(a, b, (x, y) -> x - y);
    }

    public static NumJArray subi(NumJArray a, NumJArray b) {
        return elementwisei(a, b, (x, y) -> x - y);
    }

    public static NumJArray muli(NumJArray a, float b) {
        return elementwisei(a, b, (x, y) -> x * y);
    }

    public static NumJArray muli(NumJArray a, NumJArray b) {
        return elementwisei(a, b, (x, y) -> x * y);
    }

    public static NumJArray divi(NumJArray a, float b) {
        return elementwisei(a, b, (x, y) -> x / y);
    }

    public static NumJArray divi(NumJArray a, NumJArray b) {
        return elementwisei(a, b, (x, y) -> x / y);
    }

    public static NumJArray sqrti(NumJArray a) {
        return elementwisei(a, x -> (float) Math.sqrt(x));
    }

    public static NumJArray powi(NumJArray a, float b) {
        return elementwisei(a, x -> (float) Math.pow(x, b));
    }

    public static NumJArray expi(NumJArray a) {
        return elementwisei(a, x -> (float) Math.exp(x));
    }

    public static NumJArray logi(NumJArray a) {
        return elementwisei(a, x -> (float) Math.log(x));
    }

    public static NumJArray modi(NumJArray a, float b) {
        return elementwisei(a, x -> x % b);
    }

    public static NumJArray putWhere(NumJArray array, NumJArray mask, float value) {
        if (mask == null) {
            throw new IllegalArgumentException("mask must not be null");
        }
        float[] m = mask.broadcastTo(array.shape).toContiguousData();
        int[] pos = array.positions();
        for (int i = 0; i < pos.length; i++) {
            if (m[i] != 0) array.data[pos[i]] = value;
        }
        return array;
    }

    public static NumJArray putWhere(NumJArray array, NumJArray mask, NumJArray values) {
        if (mask == null || values == null) {
            throw new IllegalArgumentException("mask and values must not be null");
        }
        float[] m = mask.broadcastTo(array.shape).toContiguousData();
        float[] v = values.broadcastTo(array.shape).toContiguousData();
        int[] pos = array.positions();
        for (int i = 0; i < pos.length; i++) {
            if (m[i] != 0) array.data[pos[i]] = v[i];
        }
        return array;
    }

    public static NumJArray conditionalPut(NumJArray array, FloatPredicate test, float value) {
        if (test == null) {
            throw new IllegalArgumentException("test must not be null");
        }
        for (int p : array.positions()) {
            if (test.test(array.data[p])) array.data[p] = value;
        }
        return array;
    }

    public static NumJArray where(NumJArray array, FloatPredicate test) {
        if (test == null) {
            throw new IllegalArgumentException("test must not be null");
        }
        return map(array, x -> flag(test.test(x)));
    }

    public static NumJArray where(NumJArray cond, NumJArray x, NumJArray y) {
        if (cond == null || x == null || y == null) {
            throw new IllegalArgumentException("cond, x and y must not be null");
        }
        int[] outShape = NumJArray.broadcastShape(NumJArray.broadcastShape(cond.shape, x.shape), y.shape);
        float[] c = cond.broadcastTo(outShape).toContiguousData();
        float[] xs = x.broadcastTo(outShape).toContiguousData();
        float[] ys = y.broadcastTo(outShape).toContiguousData();

        float[] out = new float[c.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = (c[i] != 0) ? xs[i] : ys[i];
        }
        return new NumJArray(out, outShape);
    }

    public static NumJArray where(NumJArray cond, float x, float y) {
        if (cond == null) {
            throw new IllegalArgumentException("cond must not be null");
        }
        return map(cond, c -> c != 0 ? x : y);
    }

    public static int[] broadcastShape(int[] a, int[] b) {
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

    public static NumJArray broadcastOp(NumJArray a, NumJArray b, FloatBinaryOperator op) {
        if (b == null) {
            throw new IllegalArgumentException("b must not be null");
        }

        int[] outShape = broadcastShape(a.shape, b.shape);
        int n = Validator.sizeOf(outShape);
        float[] result = new float[n];
        if (n == 0) return new NumJArray(result, outShape);

        int[] sa = NumJArray.broadcastStrides(a.shape, a.strides, outShape);
        int[] sb = NumJArray.broadcastStrides(b.shape, b.strides, outShape);

        int nd = outShape.length;
        int[] idx = new int[nd];
        int pa = a.offset;
        int pb = b.offset;

        for (int i = 0; i < n; i++) {
            result[i] = op.applyAsFloat(a.data[pa], b.data[pb]);
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

}
