package numj.math;

import numj.core.NumJArray;

public class trig {

    public static NumJArray sin(NumJArray a) {
        return arithmatic.map(a, x -> (float) Math.sin(x));
    }

    public static NumJArray cos(NumJArray a) {
        return arithmatic.map(a, x -> (float) Math.cos(x));
    }

    public static NumJArray tan(NumJArray a) {
        return arithmatic.map(a, x -> (float) Math.tan(x));
    }

    public static NumJArray sinh(NumJArray a) {
        return arithmatic.map(a, x -> (float) Math.sinh(x));
    }

    public static NumJArray cosh(NumJArray a) {
        return arithmatic.map(a, x -> (float) Math.cosh(x));
    }

    public static NumJArray tanh(NumJArray a) {
        return arithmatic.map(a, x -> (float) Math.tanh(x));
    }

    public static NumJArray asin(NumJArray a) {
        return arithmatic.map(a, x -> (float) Math.asin(x));
    }

    public static NumJArray acos(NumJArray a) {
        return arithmatic.map(a, x -> (float) Math.acos(x));
    }

    public static NumJArray atan(NumJArray a) {
        return arithmatic.map(a, x -> (float) Math.atan(x));
    }

    public static NumJArray atan2(NumJArray y, NumJArray x) {
        return arithmatic.elementwise(y, x, (a, b) -> (float) Math.atan2(a, b));
    }

    public static NumJArray sini(NumJArray a) {
        return arithmatic.mapi(a, x -> (float) Math.sin(x));
    }

    public static NumJArray cosi(NumJArray a) {
        return arithmatic.mapi(a, x -> (float) Math.cos(x));
    }

    public static NumJArray tani(NumJArray a) {
        return arithmatic.mapi(a, x -> (float) Math.tan(x));
    }

    public static NumJArray sinhi(NumJArray a) {
        return arithmatic.mapi(a, x -> (float) Math.sinh(x));
    }

    public static NumJArray coshi(NumJArray a) {
        return arithmatic.mapi(a, x -> (float) Math.cosh(x));
    }

    public static NumJArray tanhi(NumJArray a) {
        return arithmatic.mapi(a, x -> (float) Math.tanh(x));
    }

    public static NumJArray asini(NumJArray a) {
        return arithmatic.mapi(a, x -> (float) Math.asin(x));
    }

    public static NumJArray acosi(NumJArray a) {
        return arithmatic.mapi(a, x -> (float) Math.acos(x));
    }

    public static NumJArray atani(NumJArray a) {
        return arithmatic.mapi(a, x -> (float) Math.atan(x));
    }

    public static NumJArray atan2i(NumJArray y, NumJArray x) {
        return arithmatic.elementwisei(y, x, (a, b) -> (float) Math.atan2(a, b));
    }       

}
