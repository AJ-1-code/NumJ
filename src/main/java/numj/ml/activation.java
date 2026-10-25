package numj.ml;

import numj.core.NumJArray;
import numj.math.arithmatic;
import java.lang.Math;

public class activation {

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
    public static NumJArray relu(NumJArray array) {
        return arithmatic.map(array, (float x) -> Math.max(0f, x));
    }

    /**
     * Derivative of ReLU: 1 where {@code x > 0}, otherwise 0.
     *
     * @return a new array of 0s and 1s (input is the pre-activation value)
     */
    public static NumJArray reluD(NumJArray array) {
        return arithmatic.map(array, (float x) -> x > 0f ? 1.0f : 0.0f);
    }

    /**
     * Sigmoid activation applied to every element.
     *
     * @return a new array of the same shape with values in {@code (0, 1)}
     */
    public static NumJArray sigmoid(NumJArray array) {
        return arithmatic.map(array, activation::sig);
    }

    /**
     * Derivative of sigmoid, {@code s * (1 - s)} where {@code s = sigmoid(x)}.
     *
     * @return a new array (input is the pre-activation value, not the sigmoid output)
     */
    public static NumJArray sigmoidD(NumJArray array) {
        return arithmatic.map(array, (float x) -> {
            float s = sig(x);
            return s * (1.0f - s);
        });
    }

    /**
     * Tanh activation applied to every element.
     *
     * @return a new array of the same shape with values in {@code (-1, 1)}
     */
    public static NumJArray tanh(NumJArray array) {
        return arithmatic.map(array, (float x) -> (float) Math.tanh(x));
    }

    /**
     * Derivative of tanh, {@code 1 - tanh(x)^2}.
     *
     * @return a new array (input is the pre-activation value, not the tanh output)
     */
    public static NumJArray tanhD(NumJArray array) {
        return arithmatic.map(array, (float x) -> {
            float t = (float) Math.tanh(x);
            return 1.0f - t * t;
        });
    }

    /**
     * Softmax over all elements (they sum to 1 afterwards). Stable against overflow.
     *
     * @return a new array of the same shape
     */
    public static NumJArray softmax(NumJArray array) {
        float[] out = array.toContiguousData();
        if (out.length == 0) return new NumJArray(out, array.shape);

        float max = Float.NEGATIVE_INFINITY;
        for (float v : out) max = Math.max(max, v);

        float total = 0;
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) Math.exp(out[i] - max);
            total += out[i];
        }
        for (int i = 0; i < out.length; i++) out[i] /= total;

        return new NumJArray(out, array.shape);
    }

    /**
     * Softmax along one axis, e.g. {@code softmaxAxis(-1)} on a batch of scores gives
     * per-row probabilities. Stable against overflow.
     *
     * @param axis the axis that should sum to 1; negative values count from the end
     * @return a new array of the same shape
     * @throws IllegalArgumentException if the axis is out of range
     */
    public static NumJArray softmaxAxis(NumJArray array, int axis) {
        int ax = array.normalizeAxis(axis);
        NumJArray out = array.copy();
        int len = out.shape[ax];
        int st = out.strides[ax];

        for (int base : out.laneOffsets(ax)) {
            float max = Float.NEGATIVE_INFINITY;
            for (int k = 0; k < len; k++) max = Math.max(max, out.data[base + k * st]);

            float total = 0;
            for (int k = 0; k < len; k++) {
                int p = base + k * st;
                out.data[p] = (float) Math.exp(out.data[p] - max);
                total += out.data[p];
            }
            for (int k = 0; k < len; k++) out.data[base + k * st] /= total;
        }
        return out;
    }
}
