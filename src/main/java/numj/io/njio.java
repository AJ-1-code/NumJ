package numj.io;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;

import numj.core.NumJArray;
import numj.core.Validator;

public class njio {

    /**
     * Saves this array in NumJ's binary format.
     *
     * @param filePath the destination .nj file
     */
    public static void save(NumJArray a, String filePath) {
        if (filePath == null ||
                !filePath.toLowerCase().endsWith(".nj")) {
            throw new IllegalArgumentException(
                "File type must be .nj"
            );
        }

        NumJArray array = a.copy();

        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(
                    new FileOutputStream(filePath)))) {

            out.writeInt(0x4E554D4A);
            out.writeInt(1);
            out.writeInt(array.shape.length);

            for (int dimension : array.shape) {
                out.writeInt(dimension);
            }

            for (float value : array.data) {
                out.writeFloat(value);
            }

        } catch (IOException e) {
            throw new UncheckedIOException(
                "Failed to save NumJ array: " + filePath, e
            );
        }
    }

    /**
     * Loads an array from NumJ's binary format.
     *
     * @param filePath the source .nj file
     * @return the loaded array
     */
    public static NumJArray load(String filePath) {
        if (filePath == null ||!filePath.toLowerCase().endsWith(".nj")) {
            throw new IllegalArgumentException("File type must be .nj");
        }

        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {

            if (in.readInt() != 0x4E554D4A) {
                throw new IOException("Invalid NumJ file");
            }

            if (in.readInt() != 1) {
                throw new IOException("Unsupported NumJ file version");
            }

            int ndim = in.readInt();

            if (ndim < 0 || ndim > 32) {
                throw new IOException("Invalid number of dimensions: " + ndim);
            }

            int[] shape = new int[ndim];

            for (int i = 0; i < ndim; i++) {
                shape[i] = in.readInt();

                if (shape[i] < 0) {
                    throw new IOException("Negative dimension");
                }
            }

            int size;

            try {
                size = Validator.sizeOf(shape);
            } catch (RuntimeException e) {
                throw new IOException("Invalid array shape", e);
            }

            float[] data = new float[size];

            for (int i = 0; i < size; i++) {
                data[i] = in.readFloat();
            }

            return new NumJArray(data, shape);

        } catch (IOException e) {
            throw new UncheckedIOException("Failed to load NumJ array: " + filePath, e);
        }
    }

}
