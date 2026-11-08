package numj.ml;

import numj.core.NumJArray;

public final class loss {

    private static final float EPS = 1e-7f;

    private loss() {}

    private static void checkShapes(NumJArray a, NumJArray b, String name) {
        if (!java.util.Arrays.equals(a.shape, b.shape)) {
            throw new IllegalArgumentException(name + ": shape mismatch "+ java.util.Arrays.toString(a.shape) + " vs "+ java.util.Arrays.toString(b.shape));
        }
    }


    public static float mse(NumJArray y, NumJArray yHat) {
        checkShapes(y, yHat, "mse");
        return yHat.sub(y).pow(2f).mean();
    }

    public static NumJArray mseD(NumJArray y, NumJArray yHat) {
        checkShapes(y, yHat, "mseD");
        return yHat.sub(y).mul(2.0f / yHat.size());
    }

    public static float mae(NumJArray y, NumJArray yHat) {
        checkShapes(y, yHat, "mae");
        return yHat.sub(y).abs().mean();
    }

    public static NumJArray maeD(NumJArray y, NumJArray yHat) {
        checkShapes(y, yHat, "maeD");
        return yHat.sub(y).sign().div(yHat.size());
    }

    public static float huber(NumJArray y, NumJArray yHat, float delta) {
        if (delta <= 0) throw new IllegalArgumentException("huber: delta must be > 0");
        checkShapes(y, yHat, "huber");
        NumJArray error = yHat.sub(y).abs();
        NumJArray cond = error.lessEqual(delta);
        NumJArray quad = error.pow(2f).mul(0.5f);
        NumJArray lin = error.sub(0.5f * delta).mul(delta);
        return NumJArray.where(cond, quad, lin).mean();
    }

    public static NumJArray huberD(NumJArray y, NumJArray yHat, float delta) {
        if (delta <= 0) throw new IllegalArgumentException("huberD: delta must be > 0");
        checkShapes(y, yHat, "huberD");
        return yHat.sub(y).clip(-delta, delta).div(yHat.size());
    }

    public static float bce(NumJArray y, NumJArray yHat) {
        checkShapes(y, yHat, "bce");
        NumJArray p = yHat.clip(EPS, 1.0f - EPS);
        NumJArray left = p.log().mul(y);
        NumJArray right = p.neg().add(1.0f).log().mul(y.neg().add(1.0f));
        return -left.add(right).mean();
    }

    public static NumJArray bceD(NumJArray y, NumJArray yHat) {
        checkShapes(y, yHat, "bceD");
        NumJArray p = yHat.clip(EPS, 1.0f - EPS);
        // (p - y) / (p * (1 - p)) / N
        NumJArray denom = p.mul(p.neg().add(1.0f));
        return p.sub(y).div(denom).div(p.size());
    }

    public static float bcel(NumJArray y, NumJArray logits) {
        checkShapes(y, logits, "bcel");
        NumJArray maxZero = logits.relu();
        NumJArray interaction = logits.mul(y);
        NumJArray stableLog = logits.abs().neg().exp().add(1.0f).log();
        return maxZero.sub(interaction).add(stableLog).mean();
    }

    public static NumJArray bcelD(NumJArray y, NumJArray logits) {
        checkShapes(y, logits, "bcelD");
        return logits.sigmoid().sub(y).div(logits.size());
    }

    public static float cce(NumJArray y, NumJArray yHat) {
        checkShapes(y, yHat, "cce");
        NumJArray p = yHat.clip(EPS, 1.0f);
        return -p.log().mul(y).sumAxis(1).mean();
    }

    public static float smce(NumJArray y, NumJArray logits) {
        checkShapes(y, logits, "smce");
        
        if (logits.ndim() == 1) {
            logits = logits.reshape(1, logits.shape[0]);
            y = y.reshape(1, y.shape[0]);
        }
    
        int n = logits.shape[0];
        NumJArray max = logits.maxAxis(1).reshape(n, 1);
        NumJArray shifted = logits.sub(max);
        NumJArray logSumExp = shifted.exp().sumAxis(1).log().reshape(n, 1);
        NumJArray logProbs = shifted.sub(logSumExp);
    
        return -logProbs.mul(y).sumAxis(1).mean();
    }
    
    public static NumJArray smceD(NumJArray y, NumJArray logits) {
        checkShapes(y, logits, "smceD");
    
        if (logits.ndim() == 1) {
            return logits.softmax().sub(y);   // single sample, no batch division
        }
        return logits.softmaxAxis(1).sub(y).div(logits.shape[0]);
    }
}