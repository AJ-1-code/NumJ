
package numj;

import numj.core.NumJArray;

import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)

@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)

@Fork(value = 2, jvmArgsAppend = {
    "--add-modules=jdk.incubator.vector",

    "-XX:CompileCommand=inline,numj.core.NumJArray::matMul"
})
public class MatMulBench {

    @Param({"64", "256", "512", "1024"})
    public int size;

    private NumJArray a;
    private NumJArray b;

    @Setup(Level.Trial)
    public void setup() {
        a = NumJArray.ones(size, size);
        b = NumJArray.ones(size, size);
    }

    @Benchmark
    public NumJArray matMul() {
        return a.matMul(b);
    }
}
