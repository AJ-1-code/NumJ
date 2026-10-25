package numj.io;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import numj.core.NumJArray;
import numj.core.Validator;

public class npio {

    /**
     * Saves this array in NumPy's .npy format.
     *
     * // SHOULD WORK, BUT HAVEN'T FULLY TESTED.
     *
     * @param filePath the destination .npy file
     */
    public static void saveNpy(NumJArray a, String filePath) {
        // SHOULD WORK, BUT HAVEN'T FULLY TESTED.
        if (filePath == null ||!filePath.toLowerCase().endsWith(".npy")) {
            throw new IllegalArgumentException("File type must be .npy");
        }

        NumJArray array = a.copy();

        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(filePath))) {

            out.write(new byte[] {(byte) 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0});

            StringBuilder shapeString = new StringBuilder("(");

            for (int i = 0; i < array.shape.length; i++) {
                if (i > 0) shapeString.append(", ");
                shapeString.append(array.shape[i]);
            }

            if (array.shape.length == 1) {
                shapeString.append(",");
            }

            shapeString.append(")");

            String header ="{'descr': '<f8', 'fortran_order': False, 'shape': "+ shapeString + ", }";
            int padding =(16 - ((10 + header.length() + 1) % 16)) % 16;
            String paddedHeader = header + " ".repeat(padding) + "\n";
            byte[] headerBytes = paddedHeader.getBytes(StandardCharsets.US_ASCII);

            if (headerBytes.length > 65535) {
                throw new IOException("NPY header is too large");
            }

            out.write(headerBytes.length & 0xFF);
            out.write((headerBytes.length >>> 8) & 0xFF);
            out.write(headerBytes);

            ByteBuffer buffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);

            for (float value : array.data) {
                buffer.clear();
                buffer.putFloat(value);
                out.write(buffer.array());
            }

        } catch (IOException e) {
            throw new UncheckedIOException("Failed to save NPY array: " + filePath, e);
        }
    }

    /**
     * Loads a C-order float64 array from a NumPy .npy file.
     *
     * // SHOULD WORK, BUT HAVEN'T FULLY TESTED.
     *
     * @param filePath the source .npy file
     * @return the loaded array
     */
    public static NumJArray loadNpy(String filePath) {
        // SHOULD WORK, BUT HAVEN'T FULLY TESTED.
        if (filePath == null || !filePath.toLowerCase().endsWith(".npy")) {
            throw new IllegalArgumentException("File type must be .npy"
            );
        }

        try (InputStream in = new BufferedInputStream(new FileInputStream(filePath))) {

            byte[] magic = in.readNBytes(6);

            if (magic.length != 6 || magic[0] != (byte) 0x93 || magic[1] != 'N' || magic[2] != 'U' || magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
                throw new IOException("Invalid NPY file");
            }

            int major = in.read();
            int minor = in.read();

            if (major < 1 || major > 3 || minor < 0) {
                throw new IOException("Unsupported NPY version");
            }

            int headerLength;

            if (major == 1) {
                int lo = in.read();
                int hi = in.read();

                if (lo < 0 || hi < 0) {
                    throw new EOFException("Incomplete NPY header");
                }

                headerLength = lo | (hi << 8);

            } else {
                byte[] lengthBytes = in.readNBytes(4);

                if (lengthBytes.length != 4) {
                    throw new EOFException("Incomplete NPY header");
                }

                headerLength = ByteBuffer.wrap(lengthBytes).order(ByteOrder.LITTLE_ENDIAN).getInt();

                if (headerLength < 0) {
                    throw new IOException("Invalid NPY header length");
                }
            }

            if (headerLength <= 0 || headerLength > 1_000_000) {
                throw new IOException("Invalid NPY header length");
            }

            byte[] headerBytes = in.readNBytes(headerLength);

            if (headerBytes.length != headerLength) {
                throw new EOFException("Incomplete NPY header");
            }

            String header = new String(headerBytes,major == 3 ? StandardCharsets.UTF_8 : StandardCharsets.US_ASCII);

            Matcher dtypeMatcher = Pattern.compile("['\"]descr['\"]\\s*:\\s*['\"]([<>=|][fiu]\\d+)['\"]").matcher(header);

            if (!dtypeMatcher.find()) {
                throw new IOException("Missing or unsupported NPY dtype");
            }

            String dtype = dtypeMatcher.group(1);

            if (!dtype.matches("[<>=|]f8")) {
                throw new IOException("Only float64 NPY arrays are supported: " + dtype);
            }

            Matcher fortranMatcher = Pattern.compile("['\"]fortran_order['\"]\\s*:\\s*(True|False)").matcher(header);

            if (!fortranMatcher.find() ||fortranMatcher.group(1).equals("True")) {
                throw new IOException("Only C-order NPY arrays are supported");
            }

            Matcher shapeMatcher = Pattern.compile( "['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)").matcher(header);

            if (!shapeMatcher.find()) {
                throw new IOException("Missing NPY shape");
            }

            String shapeText = shapeMatcher.group(1).trim();
            int[] shape;

            if (shapeText.isEmpty()) {
                shape = new int[0];
            } else {
                String[] dimensions = shapeText.split(",");
                java.util.ArrayList<Integer> dims = new java.util.ArrayList<>();

                for (String dimension : dimensions) {
                    dimension = dimension.trim();

                    if (!dimension.isEmpty()) {
                        dims.add(Integer.parseInt(dimension));
                    }
                }

                shape = new int[dims.size()];

                for (int i = 0; i < shape.length; i++) {
                    shape[i] = dims.get(i);

                    if (shape[i] < 0) {
                        throw new IOException("Negative NPY dimension");
                    }
                }
            }

            int size;

            try {
                size = Validator.sizeOf(shape);
            } catch (RuntimeException e) {
                throw new IOException("Invalid NPY shape", e);
            }

            ByteOrder order = dtype.charAt(0) == '>' ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN;

            float[] data = new float[size];
            byte[] valueBytes = new byte[8];
            ByteBuffer valueBuffer = ByteBuffer.wrap(valueBytes).order(order);

            for (int i = 0; i < size; i++) {
                int read = 0;

                while (read < 8) {
                    int count = in.read(valueBytes, read, 8 - read);

                    if (count < 0) {
                        throw new EOFException("Incomplete NPY data");
                    }

                    read += count;
                }

                valueBuffer.position(0);
                data[i] = valueBuffer.getFloat();
            }

            return new NumJArray(data, shape);

        } catch (IOException e) {
            throw new UncheckedIOException("Failed to load NPY array: " + filePath, e);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid NPY shape: " + filePath, e);
        }
    }

}
