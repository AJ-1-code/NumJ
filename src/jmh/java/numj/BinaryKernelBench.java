package numj;

import numj.core.NumJArray;

import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)

@Warmup(iterations = 10, time = 1)
@Measurement(iterations = 5, time = 1)

@Fork(value = 1, jvmArgsAppend = {
    "--add-modules=jdk.incubator.vector",

    "-XX:CompileCommand=inline,numj.core.NumJArray::add",
    "-XX:CompileCommand=inline,numj.core.NumJArray::sub",
    "-XX:CompileCommand=inline,numj.core.NumJArray::mul",
    "-XX:CompileCommand=inline,numj.core.NumJArray::div",

    "-XX:CompileCommand=inline,numj.core.NumJArray::addi",
    "-XX:CompileCommand=inline,numj.core.NumJArray::subi",
    "-XX:CompileCommand=inline,numj.core.NumJArray::muli",
    "-XX:CompileCommand=inline,numj.core.NumJArray::divi",

    "-XX:CompileCommand=inline,numj.core.NumJArray::addKernel",
    "-XX:CompileCommand=inline,numj.core.NumJArray::subKernel",
    "-XX:CompileCommand=inline,numj.core.NumJArray::mulKernel",
    "-XX:CompileCommand=inline,numj.core.NumJArray::divKernel",

    "-XX:CompileCommand=inline,numj.core.NumJArray::addInto",
    "-XX:CompileCommand=inline,numj.core.NumJArray::subInto",
    "-XX:CompileCommand=inline,numj.core.NumJArray::mulInto",
    "-XX:CompileCommand=inline,numj.core.NumJArray::divInto"
})

public class BinaryKernelBench {

    @Param({"100", "1000", "4096", "10000", "100000"})
    public int size;

    private NumJArray a;
    private NumJArray b;

    private NumJArray reusableDest;

    @Setup(Level.Iteration)
    public void setup() {
        a = NumJArray.ones(size);
        b = NumJArray.ones(size);

        reusableDest = NumJArray.zeros(size);
    }

    @Benchmark
    public NumJArray originalAdd() {
        return a.add(b);
    }

    @Benchmark
    public NumJArray originalAddi() {
        return a.addi(b);
    }

    @Benchmark
    public NumJArray kernelAdd() {
        return a.add(b);
    }

    @Benchmark
    public NumJArray kernelAddInto() {
        return a.addi(reusableDest);
    }

    @Benchmark
    public NumJArray originalSub() {
        return a.sub(b);
    }

    @Benchmark
    public NumJArray originalSubi() {
        return a.subi(b);
    }

    @Benchmark
    public NumJArray kernelSub() {
        return a.sub(b);
    }

    @Benchmark
    public NumJArray kernelSubInto() {
        return a.subi(reusableDest);
    }

    @Benchmark
    public NumJArray originalMul() {
        return a.mul(b);
    }

    @Benchmark
    public NumJArray originalMuli() {
        return a.muli(b);
    }

    @Benchmark
    public NumJArray kernelMul() {
        return a.mul(b);
    }

    @Benchmark
    public NumJArray kernelMulInto() {
        return a.muli(reusableDest);
    }

    @Benchmark
    public NumJArray originalDiv() {
        return a.div(b);
    }

    @Benchmark
    public NumJArray originalDivi() {
        return a.divi(b);
    }

    @Benchmark
    public NumJArray kernelDiv() {
        return a.div(b);
    }

    @Benchmark
    public NumJArray kernelDivInto() {
        return a.divi(reusableDest);
    }
}
