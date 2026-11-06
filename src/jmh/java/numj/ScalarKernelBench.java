
package numj;

import numj.core.NumJArray;

import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)

@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 20, time = 1)

@Fork(value = 2, jvmArgsAppend = {
    "--add-modules=jdk.incubator.vector",

    "-XX:CompileCommand=inline,numj.core.NumJArray::add",
    "-XX:CompileCommand=inline,numj.core.NumJArray::addi",
    "-XX:CompileCommand=inline,numj.core.KernelScalars::add",
    "-XX:CompileCommand=inline,numj.core.KernelScalars::addi"
})
public class ScalarKernelBench {

    @Param({"100", "1000", "4096", "10000", "100000"})
    public int size;

    private static final float SCALAR = 2.5f;

    private NumJArray a;
    private NumJArray dest;

    @Setup(Level.Trial)
    public void setup() {
        a = NumJArray.ones(size);
        dest = NumJArray.zeros(size);
    }


    @Benchmark
    public NumJArray originalAdd() {

        NumJArray out = NumJArray.zeros(size);

        float[] src = a.data;
        float[] dst = out.data;

        int offset = a.offset;

        for (int i = 0; i < size; i++) {
            dst[i] = src[offset + i] + SCALAR;
        }

        return out;
    }


    @Benchmark
    public NumJArray simdAdd() {
        return a.add(SCALAR);
    }


    // Original scalar in-place addition
    @Benchmark
    public NumJArray originalAddInto() {
    
        float[] data = a.data;
        int offset = a.offset;
    
        for (int i = 0; i < size; i++) {
            data[offset + i] += SCALAR;
        }
    
        return a;
    }
    
    // SIMD in-place addition
    @Benchmark
    public NumJArray simdAddInto() {
        return a.addi(SCALAR);
    }
}
