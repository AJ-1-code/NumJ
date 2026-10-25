package numj.core;

import java.util.Arrays;

public class Validator {

    public static int[] checkShape(int[] shape) {
        if (shape == null) {
            throw new IllegalArgumentException("shape must not be null");
        }
        for (int dim : shape) {
            if (dim < 0) {
                throw new IllegalArgumentException("Dimensions must be non-negative, got " + Arrays.toString(shape));
            }
        }
        return shape;
    }

    public static int sizeOf(int[] shape) {
        try {
            int total = 1;
            for (int dim : shape) total = Math.multiplyExact(total, dim);
            return total;
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("Shape is too large: " + Arrays.toString(shape));
        }
    }

}
