package numj.core;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class NumJSIMD {

    // =========================================================================
    // OPERATION IDENTIFIERS
    // =========================================================================

    // Unary operation IDs
    public static final int ABS = 0, NEG = 1, SQRT = 2, EXP = 3, LOG = 4,
            FLOOR = 5, CEIL = 6, ROUND = 7, SIGN = 8,
            SIN = 9, COS = 10, TAN = 11, ASIN = 12, ACOS = 13, ATAN = 14,
            IS_NAN = 15, IS_INF = 16, IS_FINITE = 17, LOGICAL_NOT = 18;

    // Binary operation IDs
    public static final int ADD = 50, SUB = 51, MUL = 52, DIV = 53,
            POW = 54, MOD = 55, CLIP = 56,
            EQUAL = 57, NOT_EQUAL = 58, GREATER = 59, GREATER_EQUAL = 60,
            LESS = 61, LESS_EQUAL = 62, LOGICAL_AND = 63, LOGICAL_OR = 64,
            LOGICAL_XOR = 65, MAXIMUM = 66, MINIMUM = 67, IS_CLOSE = 68;

    private static final VectorSpecies<Float> SPECIES =
            FloatVector.SPECIES_PREFERRED;

    private NumJSIMD() {
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    private static FloatVector maskToFloat(VectorMask<Float> mask) {
        return FloatVector.zero(SPECIES).blend(1.0f, mask);
    }

    private static boolean truthy(float value) {
        return value != 0.0f;
    }

    private static void validateArray(NumJArray a) {
        if (a == null) {
            throw new IllegalArgumentException("Array must not be null");
        }
    }

    private static void validateExtra(int opId, float[] extra) {
        if (opId == CLIP && (extra == null || extra.length < 1)) {
            throw new IllegalArgumentException(
                    "CLIP requires a maximum value in extra[0]");
        }
    }

    // =========================================================================
    // UNARY OPERATIONS - IN PLACE
    // =========================================================================

    public static NumJArray elementwisei(NumJArray a, int opId) {
        validateArray(a);

        if (a.isContiguous() && supportsUnaryVector(opId)) {
            float[] data = a.data;
            int start = a.offset;
            int length = a.size;
            int i = 0;
            int upperBound = SPECIES.loopBound(length);

            for (; i < upperBound; i += SPECIES.length()) {
                int p = start + i;
                FloatVector va = FloatVector.fromArray(SPECIES, data, p);
                applyUnaryVector(va, opId).intoArray(data, p);
            }

            for (; i < length; i++) {
                data[start + i] = applyUnaryScalar(data[start + i], opId);
            }

            return a;
        }

        for (int p : a.positions()) {
            a.data[p] = applyUnaryScalar(a.data[p], opId);
        }

        return a;
    }

    // =========================================================================
    // UNARY OPERATIONS - OUT OF PLACE
    // =========================================================================

    public static NumJArray elementwise(NumJArray a, int opId) {
        validateArray(a);

        float[] inData = a.toContiguousData();
        float[] outData = new float[inData.length];

        if (supportsUnaryVector(opId)) {
            int i = 0;
            int upperBound = SPECIES.loopBound(inData.length);

            for (; i < upperBound; i += SPECIES.length()) {
                FloatVector va = FloatVector.fromArray(SPECIES, inData, i);
                applyUnaryVector(va, opId).intoArray(outData, i);
            }

            for (; i < inData.length; i++) {
                outData[i] = applyUnaryScalar(inData[i], opId);
            }
        } else {
            for (int i = 0; i < inData.length; i++) {
                outData[i] = applyUnaryScalar(inData[i], opId);
            }
        }

        return new NumJArray(outData, a.shape);
    }

    private static boolean supportsUnaryVector(int opId) {
        return switch (opId) {
            case ABS, NEG, SQRT, EXP, LOG,
                SIN, COS, TAN, ASIN, ACOS, ATAN,
                LOGICAL_NOT, IS_NAN -> true;
            default -> false;
        };
    }

    private static FloatVector applyUnaryVector(FloatVector va, int opId) {
        return switch (opId) {
            case ABS -> va.abs();
            case NEG -> va.neg();
            case SQRT -> va.lanewise(VectorOperators.SQRT);
            case EXP -> va.lanewise(VectorOperators.EXP);
            case LOG -> va.lanewise(VectorOperators.LOG);
            case SIN -> va.lanewise(VectorOperators.SIN);
            case COS -> va.lanewise(VectorOperators.COS);
            case TAN -> va.lanewise(VectorOperators.TAN);
            case ASIN -> va.lanewise(VectorOperators.ASIN);
            case ACOS -> va.lanewise(VectorOperators.ACOS);
            case ATAN -> va.lanewise(VectorOperators.ATAN);
            case LOGICAL_NOT ->
                    maskToFloat(va.compare(VectorOperators.EQ, 0.0f));
            case IS_NAN ->
                    maskToFloat(va.test(VectorOperators.IS_NAN));
            default -> throw new IllegalArgumentException(
                    "Unsupported vector unary operation: " + opId);
        };
    }

    private static float applyUnaryScalar(float v, int opId) {
        return switch (opId) {
            case ABS -> Math.abs(v);
            case NEG -> -v;
            case SQRT -> (float) Math.sqrt(v);
            case EXP -> (float) Math.exp(v);
            case LOG -> (float) Math.log(v);
            case FLOOR -> (float) Math.floor(v);
            case CEIL -> (float) Math.ceil(v);
            case ROUND -> (float) Math.round(v);
            case SIGN -> Math.signum(v);
            case SIN -> (float) Math.sin(v);
            case COS -> (float) Math.cos(v);
            case TAN -> (float) Math.tan(v);
            case ASIN -> (float) Math.asin(v);
            case ACOS -> (float) Math.acos(v);
            case ATAN -> (float) Math.atan(v);
            case IS_NAN -> Float.isNaN(v) ? 1.0f : 0.0f;
            case IS_INF -> Float.isInfinite(v) ? 1.0f : 0.0f;
            case IS_FINITE -> Float.isFinite(v) ? 1.0f : 0.0f;
            case LOGICAL_NOT -> !truthy(v) ? 1.0f : 0.0f;
            default -> throw new IllegalArgumentException(
                    "Unknown unary operation ID: " + opId);
        };
    }

    // =========================================================================
    // ARRAY - SCALAR OPERATIONS, IN PLACE
    // =========================================================================

    public static NumJArray elementwisei(
            NumJArray a, float b, int opId, float... extra) {

        validateArray(a);
        validateExtra(opId, extra);

        if (a.isContiguous() && supportsScalarVector(opId)) {
            float[] data = a.data;
            int start = a.offset;
            int length = a.size;
            int i = 0;
            int upperBound = SPECIES.loopBound(length);

            for (; i < upperBound; i += SPECIES.length()) {
                int p = start + i;
                FloatVector va = FloatVector.fromArray(SPECIES, data, p);
                applyBinaryScalarVector(va, b, opId, extra)
                        .intoArray(data, p);
            }

            for (; i < length; i++) {
                data[start + i] =
                        applyBinaryScalarScalar(data[start + i], b, opId, extra);
            }

            return a;
        }

        for (int p : a.positions()) {
            a.data[p] = applyBinaryScalarScalar(a.data[p], b, opId, extra);
        }

        return a;
    }

    // =========================================================================
    // ARRAY - SCALAR OPERATIONS, OUT OF PLACE
    // =========================================================================

    public static NumJArray elementwise(
            NumJArray a, float b, int opId, float... extra) {

        validateArray(a);
        validateExtra(opId, extra);

        float[] inData = a.toContiguousData();
        float[] outData = new float[inData.length];

        if (supportsScalarVector(opId)) {
            int i = 0;
            int upperBound = SPECIES.loopBound(inData.length);

            for (; i < upperBound; i += SPECIES.length()) {
                FloatVector va = FloatVector.fromArray(SPECIES, inData, i);
                applyBinaryScalarVector(va, b, opId, extra)
                        .intoArray(outData, i);
            }

            for (; i < inData.length; i++) {
                outData[i] =
                        applyBinaryScalarScalar(inData[i], b, opId, extra);
            }
        } else {
            for (int i = 0; i < inData.length; i++) {
                outData[i] =
                        applyBinaryScalarScalar(inData[i], b, opId, extra);
            }
        }

        return new NumJArray(outData, a.shape);
    }

    private static boolean supportsScalarVector(int opId) {
        return switch (opId) {
            case ADD, SUB, MUL, DIV, POW, CLIP,
                    EQUAL, NOT_EQUAL, GREATER, GREATER_EQUAL,
                    LESS, LESS_EQUAL -> true;
            default -> false;
        };
    }

    private static FloatVector applyBinaryScalarVector(
            FloatVector va, float b, int opId, float[] extra) {

        return switch (opId) {
            case ADD -> va.add(b);
            case SUB -> va.sub(b);
            case MUL -> va.mul(b);
            case DIV -> va.div(b);
            case POW -> va.lanewise(VectorOperators.POW, b);

            case CLIP ->
                    va.lanewise(VectorOperators.MAX, b)
                      .lanewise(VectorOperators.MIN, extra[0]);

            case EQUAL ->
                    maskToFloat(va.compare(VectorOperators.EQ, b));
            case NOT_EQUAL ->
                    maskToFloat(va.compare(VectorOperators.NE, b));
            case GREATER ->
                    maskToFloat(va.compare(VectorOperators.GT, b));
            case GREATER_EQUAL ->
                    maskToFloat(va.compare(VectorOperators.GE, b));
            case LESS ->
                    maskToFloat(va.compare(VectorOperators.LT, b));
            case LESS_EQUAL ->
                    maskToFloat(va.compare(VectorOperators.LE, b));

            default -> throw new IllegalArgumentException(
                    "Unsupported vector scalar operation: " + opId);
        };
    }

    private static float applyBinaryScalarScalar(
            float v, float b, int opId, float[] extra) {

        return switch (opId) {
            case ADD -> v + b;
            case SUB -> v - b;
            case MUL -> v * b;
            case DIV -> v / b;
            case POW -> (float) Math.pow(v, b);
            case MOD -> v % b;

            case CLIP -> Math.max(b, Math.min(v, extra[0]));

            case EQUAL -> v == b ? 1.0f : 0.0f;
            case NOT_EQUAL -> v != b ? 1.0f : 0.0f;
            case GREATER -> v > b ? 1.0f : 0.0f;
            case GREATER_EQUAL -> v >= b ? 1.0f : 0.0f;
            case LESS -> v < b ? 1.0f : 0.0f;
            case LESS_EQUAL -> v <= b ? 1.0f : 0.0f;

            case LOGICAL_AND ->
                    truthy(v) && truthy(b) ? 1.0f : 0.0f;
            case LOGICAL_OR ->
                    truthy(v) || truthy(b) ? 1.0f : 0.0f;
            case LOGICAL_XOR ->
                    truthy(v) != truthy(b) ? 1.0f : 0.0f;

            case MAXIMUM -> Math.max(v, b);
            case MINIMUM -> Math.min(v, b);

            default -> throw new IllegalArgumentException(
                    "Unknown binary scalar operation ID: " + opId);
        };
    }

    // =========================================================================
    // ARRAY - ARRAY OPERATIONS, IN PLACE
    // =========================================================================

    public static NumJArray elementwisei(
            NumJArray a, NumJArray b, int opId) {

        validateArray(a);
        validateArray(b);

        float[] other = b.broadcastTo(a.shape).toContiguousData();

        if (a.isContiguous() && supportsArrayVector(opId)) {
            float[] data = a.data;
            int start = a.offset;
            int length = a.size;
            int i = 0;
            int upperBound = SPECIES.loopBound(length);

            for (; i < upperBound; i += SPECIES.length()) {
                int p = start + i;
                FloatVector va = FloatVector.fromArray(SPECIES, data, p);
                FloatVector vb = FloatVector.fromArray(SPECIES, other, i);

                applyArrayVector(va, vb, opId).intoArray(data, p);
            }

            for (; i < length; i++) {
                data[start + i] =
                        applyArrayScalar(data[start + i], other[i], opId);
            }

            return a;
        }

        int[] positions = a.positions();

        for (int i = 0; i < positions.length; i++) {
            int p = positions[i];
            a.data[p] = applyArrayScalar(a.data[p], other[i], opId);
        }

        return a;
    }

    // =========================================================================
    // ARRAY - ARRAY OPERATIONS, OUT OF PLACE
    // =========================================================================

    public static NumJArray elementwise(
            NumJArray a, NumJArray b, int opId) {

        validateArray(a);
        validateArray(b);

        float[] flatA = a.toContiguousData();
        float[] flatB = b.broadcastTo(a.shape).toContiguousData();
        float[] outData = new float[flatA.length];

        if (supportsArrayVector(opId)) {
            int i = 0;
            int upperBound = SPECIES.loopBound(flatA.length);

            for (; i < upperBound; i += SPECIES.length()) {
                FloatVector va = FloatVector.fromArray(SPECIES, flatA, i);
                FloatVector vb = FloatVector.fromArray(SPECIES, flatB, i);

                applyArrayVector(va, vb, opId).intoArray(outData, i);
            }

            for (; i < flatA.length; i++) {
                outData[i] = applyArrayScalar(flatA[i], flatB[i], opId);
            }
        } else {
            for (int i = 0; i < flatA.length; i++) {
                outData[i] = applyArrayScalar(flatA[i], flatB[i], opId);
            }
        }

        return new NumJArray(outData, a.shape);
    }

    private static boolean supportsArrayVector(int opId) {
        return switch (opId) {
            case ADD, SUB, MUL, DIV,
                    MAXIMUM, MINIMUM,
                    EQUAL, NOT_EQUAL, GREATER, GREATER_EQUAL,
                    LESS, LESS_EQUAL -> true;
            default -> false;
        };
    }

    private static FloatVector applyArrayVector(
            FloatVector va, FloatVector vb, int opId) {

        return switch (opId) {
            case ADD -> va.add(vb);
            case SUB -> va.sub(vb);
            case MUL -> va.mul(vb);
            case DIV -> va.div(vb);
            case MAXIMUM -> va.lanewise(VectorOperators.MAX, vb);
            case MINIMUM -> va.lanewise(VectorOperators.MIN, vb);

            case EQUAL ->
                    maskToFloat(va.compare(VectorOperators.EQ, vb));
            case NOT_EQUAL ->
                    maskToFloat(va.compare(VectorOperators.NE, vb));
            case GREATER ->
                    maskToFloat(va.compare(VectorOperators.GT, vb));
            case GREATER_EQUAL ->
                    maskToFloat(va.compare(VectorOperators.GE, vb));
            case LESS ->
                    maskToFloat(va.compare(VectorOperators.LT, vb));
            case LESS_EQUAL ->
                    maskToFloat(va.compare(VectorOperators.LE, vb));

            default -> throw new IllegalArgumentException(
                    "Unsupported vector array operation: " + opId);
        };
    }

    private static float applyArrayScalar(
            float a, float b, int opId) {

        return switch (opId) {
            case ADD -> a + b;
            case SUB -> a - b;
            case MUL -> a * b;
            case DIV -> a / b;
            case POW -> (float) Math.pow(a, b);
            case MOD -> a % b;

            case MAXIMUM -> Math.max(a, b);
            case MINIMUM -> Math.min(a, b);

            case EQUAL -> a == b ? 1.0f : 0.0f;
            case NOT_EQUAL -> a != b ? 1.0f : 0.0f;
            case GREATER -> a > b ? 1.0f : 0.0f;
            case GREATER_EQUAL -> a >= b ? 1.0f : 0.0f;
            case LESS -> a < b ? 1.0f : 0.0f;
            case LESS_EQUAL -> a <= b ? 1.0f : 0.0f;

            case LOGICAL_AND ->
                    truthy(a) && truthy(b) ? 1.0f : 0.0f;
            case LOGICAL_OR ->
                    truthy(a) || truthy(b) ? 1.0f : 0.0f;
            case LOGICAL_XOR ->
                    truthy(a) != truthy(b) ? 1.0f : 0.0f;

            default -> throw new IllegalArgumentException(
                    "Unknown array operation ID: " + opId);
        };
    }
}
