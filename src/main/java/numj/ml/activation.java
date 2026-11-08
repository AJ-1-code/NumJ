
package numj.ml;

import numj.core.NumJArray;
import numj.core.KernelScalars;

public class activation {

    public static float sig(float x) {
        if (x >= 0f) {
            return 1f / (1f + (float) Math.exp(-x));
        }

        float e = (float) Math.exp(x);
        return e / (1f + e);
    }

    public static float sigmoid(float x) {
        return sig(x);
    }

    public static float tn(float x) {
        return (float) Math.tanh(x);
    }

    public static float tanh(float x) {
        return tn(x);
    }

    public static float relu(float x) {
        return Math.max(0f, x);
    }

    public static float leakyRelu(float x, float alpha) {
        return x >= 0f ? x : alpha * x;
    }

    public static float elu(float x, float alpha) {
        return x >= 0f ? x : alpha * (float) Math.expm1(x);
    }

    public static float softplus(float x) {
        return Math.max(x, 0f)
                + (float) Math.log1p(Math.exp(-Math.abs(x)));
    }

    public static float silu(float x) {
        return x * sig(x);
    }

    public static float gelu(float x) {
        float x3 = x * x * x;
        float t = (float) Math.tanh(
                0.7978845608f * (x + 0.044715f * x3));

        return 0.5f * x * (1f + t);
    }

    public static NumJArray relu(NumJArray array) {
        return KernelScalars.relu(array);
    }

    public static NumJArray relui(NumJArray array) {
        return KernelScalars.relui(array, array);
    }

    public static NumJArray reluD(NumJArray array) {
        return KernelScalars.reluD(array);
    }

    public static NumJArray leakyRelu(NumJArray array, float alpha) {
        return KernelScalars.leakyRelu(array, alpha);
    }

    public static NumJArray leakyRelui(NumJArray array, float alpha) {
        return KernelScalars.leakyRelui(array, array, alpha);
    }

    public static NumJArray leakyReluD(
            NumJArray array, float alpha) {
        return KernelScalars.leakyReluD(array, alpha);
    }

    public static NumJArray sigmoid(NumJArray array) {
        return KernelScalars.sigmoid(array);
    }

    public static NumJArray sigmoidi(NumJArray array) {
        return KernelScalars.sigmoidi(array, array);
    }

    public static NumJArray sigmoidD(NumJArray array) {
        return KernelScalars.sigmoidD(array);
    }

    public static NumJArray tanh(NumJArray array) {
        return KernelScalars.tanh(array);
    }

    public static NumJArray tanhi(NumJArray array) {
        return KernelScalars.tanhi(array, array);
    }

    public static NumJArray tanhD(NumJArray array) {
        return KernelScalars.tanhD(array);
    }

    public static NumJArray elu(NumJArray array, float alpha) {
        return KernelScalars.elu(array, alpha);
    }

    public static NumJArray elui(NumJArray array, float alpha) {
        return KernelScalars.elui(array, array, alpha);
    }

    public static NumJArray eluD(
            NumJArray array, float alpha) {
        return KernelScalars.eluD(array, alpha);
    }

    public static NumJArray softplus(NumJArray array) {
        return KernelScalars.softplus(array);
    }

    public static NumJArray softplusi(NumJArray array) {
        return KernelScalars.softplusi(array, array);
    }

    public static NumJArray softplusD(NumJArray array) {
        return KernelScalars.softplusD(array);
    }


    public static NumJArray silu(NumJArray array) {
        return KernelScalars.silu(array);
    }

    public static NumJArray silui(NumJArray array) {
        return KernelScalars.silui(array, array);
    }

    public static NumJArray siluD(NumJArray array) {
        return KernelScalars.siluD(array);
    }

    public static NumJArray gelu(NumJArray array) {
        return KernelScalars.gelu(array);
    }

    public static NumJArray gelui(NumJArray array) {
        return KernelScalars.gelui(array, array);
    }

    public static NumJArray geluD(NumJArray array) {
        return KernelScalars.geluD(array);
    }

    public static NumJArray softmax(NumJArray array) {
        float[] out = array.toContiguousData();

        if (out.length == 0) {
            return new NumJArray(out, array.shape);
        }

        float max = Float.NEGATIVE_INFINITY;

        for (float v : out) {
            max = Math.max(max, v);
        }

        float total = 0f;

        for (int i = 0; i < out.length; i++) {
            out[i] = (float) Math.exp(out[i] - max);
            total += out[i];
        }

        for (int i = 0; i < out.length; i++) {
            out[i] /= total;
        }

        return new NumJArray(out, array.shape);
    }

    public static NumJArray softmaxAxis(NumJArray array, int axis) {

        int ax = array.normalizeAxis(axis);
        NumJArray out = array.copy();

        int len = out.shape[ax];
        int st = out.strides[ax];

        for (int base : out.laneOffsets(ax)) {

            float max = Float.NEGATIVE_INFINITY;

            for (int k = 0; k < len; k++) {
                max = Math.max(max, out.data[base + k * st]);
            }

            float total = 0f;

            for (int k = 0; k < len; k++) {
                int p = base + k * st;

                out.data[p] = (float) Math.exp(
                        out.data[p] - max);

                total += out.data[p];
            }

            for (int k = 0; k < len; k++) {
                out.data[base + k * st] /= total;
            }
        }

        return out;
    }
}
