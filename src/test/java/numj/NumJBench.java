package numj;

import java.io.PrintWriter;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

import numj.core.NumJArray;

/**
 * Benchmark / profiling harness for NumJArray. Complements NumJTest (which checks correctness).
 *
 * Per method it reports: min / median / mean / p95 / stddev time, ns per element,
 * bytes allocated per call (by the calling thread), and how many GCs ran (and for how long)
 * during the measured phase.
 *
 * Args:
 *   --n=22          vectors have 2^n elements (default 22 = 4M floats = 16 MB). Matrices are ~2^(n-2).
 *   --only=text     only run cases whose name or section contains text (case-insensitive)
 *   --warmup=300    minimum warmup time per case in ms
 *   --measure=1000  minimum measuring time per case in ms
 *   --csv=out.csv   also write results to a CSV file
 *   --tests         run NumJTest (correctness) first
 *
 * Suggested JVM flags for steadier numbers:  -Xms2g -Xmx2g -XX:+AlwaysPreTouch
 * Add -Xlog:gc to see the JVM's own GC log interleaved with the output.
 */
public class NumJBench {

    // ---------------------------------------------------------------- config
    static int n = 22;
    static long warmupMs = 300, measureMs = 1000;
    static String only = null;
    static String csvPath = null;

    static final int MIN_WARM = 3, MIN_ITERS = 5, MAX_ITERS = 200;

    // ---------------------------------------------------------------- state
    static final com.sun.management.ThreadMXBean TMX =
            (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
    static final List<GarbageCollectorMXBean> GCS = ManagementFactory.getGarbageCollectorMXBeans();
    static volatile Object sink;                       // keeps the JIT from deleting our work
    static String section = "";
    static final List<Result> results = new ArrayList<>();

    record Result(String section, String name, long elems, int iters,
                  double minNs, double medNs, double meanNs, double p95Ns, double stdNs,
                  long allocBytes, long gcCount, long gcMs) {}

    // ---------------------------------------------------------------- fixtures
    static int VEC, DIM, MM;
    static NumJArray vec, vecPos, mat, matB, matT, mm1, mm2, batchA, batchB, small, mask, rowVec, colVec;

    static void buildFixtures() {
        VEC = 1 << n;
        DIM = 1 << ((n - 2) / 2);                  // DIM*DIM ~= VEC/4
        MM = Math.max(32, DIM / 4);                // matMul is O(n^3), so keep it smaller
        vec = NumJArray.normal(0f, 1f, VEC);
        vecPos = NumJArray.uniform(0.5f, 2f, VEC);
        mat = NumJArray.normal(0f, 1f, DIM, DIM);
        matB = NumJArray.normal(0f, 1f, DIM, DIM);
        matT = mat.transpose();                     // non-contiguous view
        mm1 = NumJArray.normal(0f, 1f, MM, MM);
        mm2 = NumJArray.normal(0f, 1f, MM, MM);
        int b = 8, bd = Math.max(16, MM / 4);
        batchA = NumJArray.normal(0f, 1f, b, bd, bd);
        batchB = NumJArray.normal(0f, 1f, b, bd, bd);
        small = NumJArray.normal(0f, 1f, 1000);
        mask = vec.greater(0f);
        rowVec = NumJArray.normal(0f, 1f, DIM);
        colVec = NumJArray.normal(0f, 1f, DIM, 1);
    }

    // ---------------------------------------------------------------- main
    public static void main(String[] args) throws Exception {
        for (String a : args) {
            if (a.startsWith("--n=")) n = Integer.parseInt(a.substring(4));
            else if (a.startsWith("--only=")) only = a.substring(7).toLowerCase();
            else if (a.startsWith("--warmup=")) warmupMs = Long.parseLong(a.substring(9));
            else if (a.startsWith("--measure=")) measureMs = Long.parseLong(a.substring(10));
            else if (a.startsWith("--csv=")) csvPath = a.substring(6);
            else if (a.equals("--tests")) NumJTest.main(new String[0]);
        }

        printHeader();
        buildFixtures();
        System.out.printf("Fixtures: vec=%,d floats | mat=%dx%d | matMul=%dx%d | batch=%s%n",
                VEC, DIM, DIM, MM, MM, Arrays.toString(batchA.shape()));

        creation();
        elementwise();
        arithmetic();
        views();
        reductions();
        linearAlgebra();
        shapeOps();
        sortingAndSets();
        comparisons();
        inPlace();
        fileIO();

        printSummary();
        if (csvPath != null) writeCsv(csvPath);
    }

    // ---------------------------------------------------------------- cases
    static void creation() {
        section("Creation");
        bench("zeros", VEC, () -> NumJArray.zeros(VEC));
        bench("ones", VEC, () -> NumJArray.ones(VEC));
        bench("full", VEC, () -> NumJArray.full(7f, VEC));
        bench("arange", VEC, () -> NumJArray.arange(VEC));
        bench("linspace", VEC, () -> NumJArray.linspace(0f, 1f, VEC));
        bench("normal", VEC, () -> NumJArray.normal(0f, 1f, VEC));
        bench("uniform", VEC, () -> NumJArray.uniform(0f, 1f, VEC));
        bench("identity", (long) DIM * DIM, () -> NumJArray.identity(DIM));
        bench("eye(n,m,k=1)", (long) DIM * DIM, () -> NumJArray.eye(DIM, DIM, 1));
        bench("copy", VEC, () -> vec.copy());
        bench("copy (view)", (long) DIM * DIM, () -> matT.copy());
    }

    static void elementwise() {
        section("Element-wise (unary)");
        bench("abs", VEC, () -> vec.abs());
        bench("neg", VEC, () -> vec.neg());
        bench("sqrt", VEC, () -> vecPos.sqrt());
        bench("exp", VEC, () -> small.tile(VEC / 1000).exp());
        bench("exp (no tile cost: vecPos)", VEC, () -> vecPos.exp());
        bench("log", VEC, () -> vecPos.log());
        bench("pow(2)", VEC, () -> vec.pow(2));
        bench("sin", VEC, () -> vec.sin());
        bench("cos", VEC, () -> vec.cos());
        bench("tan", VEC, () -> vec.tan());
        bench("tanh", VEC, () -> vec.tanh());
        bench("floor", VEC, () -> vec.floor());
        bench("round", VEC, () -> vec.round());
        bench("sign", VEC, () -> vec.sign());
        bench("clip", VEC, () -> vec.clip(-1f, 1f));
        bench("relu", VEC, () -> vec.relu());
        bench("reluD", VEC, () -> vec.reluD());
        bench("sigmoid", VEC, () -> vec.sigmoid());
        bench("sigmoidD", VEC, () -> vec.sigmoidD());
        bench("map(x*x)", VEC, () -> vec.map(x -> x * x));
        bench("elementwise(unary)", VEC, () -> vec.elementwise(x -> x + 1f));
        bench("isNaN", VEC, () -> vec.isNaN());
        bench("isFinite", VEC, () -> vec.isFinite());
    }

    static void arithmetic() {
        section("Arithmetic / broadcasting");
        bench("add(scalar)", VEC, () -> vec.add(5f));
        bench("mul(scalar)", VEC, () -> vec.mul(2f));
        bench("mod(scalar)", VEC, () -> vecPos.mod(3f));
        bench("add(array) same shape", VEC, () -> vec.add(vecPos));
        bench("mul(array) same shape", VEC, () -> vec.mul(vecPos));
        bench("div(array) same shape", VEC, () -> vec.div(vecPos));
        bench("add(row) broadcast", (long) DIM * DIM, () -> mat.add(rowVec));
        bench("sub(col) broadcast", (long) DIM * DIM, () -> mat.sub(colVec));
        bench("maximum(array)", VEC, () -> vec.maximum(vecPos));
        bench("maximum(scalar)", VEC, () -> vec.maximum(0f));
        bench("elementwise(array, op)", VEC, () -> vec.elementwise(vecPos, (x, y) -> x * y + 1f));
        bench("atan2", VEC, () -> vec.atan2(vecPos));
    }

    static void views() {
        section("Views (non-contiguous)");
        long m = (long) DIM * DIM;
        bench("transpose (view only)", m, () -> mat.transpose());
        bench("reshape (contiguous, view)", m, () -> mat.reshape(DIM * DIM));
        bench("reshape (non-contiguous)", m, () -> matT.reshape(DIM * DIM));
        bench("flatten (view)", m, () -> matT.flatten());
        bench("flip(0)", m, () -> mat.flip(0));
        bench("slice step 2", m / 2, () -> mat.slice(new int[]{0, DIM, 2}, new int[]{0, DIM, 1}));
        bench("add(scalar) on view", m, () -> matT.add(1f));
        bench("sum on view", m, () -> matT.sum());
        bench("sumAxis(0) on view", m, () -> matT.sumAxis(0));
        bench("add(array) view+contig", m, () -> matT.add(matB));
    }

    static void reductions() {
        section("Reductions");
        bench("sum", VEC, () -> vec.sum());
        bench("mean", VEC, () -> vec.mean());
        bench("min", VEC, () -> vec.min());
        bench("max", VEC, () -> vec.max());
        bench("prod", VEC, () -> vec.prod());
        bench("var", VEC, () -> vec.var());
        bench("std", VEC, () -> vec.std());
        bench("norm", VEC, () -> vec.norm());
        bench("argMax", VEC, () -> vec.argMax());
        bench("argMaxFlat", VEC, () -> vec.argMaxFlat());
        bench("reduce(custom)", VEC, () -> vec.reduce((x, y) -> x + y));
        bench("all", VEC, () -> vec.all());
        bench("any", VEC, () -> vec.any());
        bench("countNonzero", VEC, () -> vec.countNonzero());
        long m = (long) DIM * DIM;
        bench("sumAxis(0)", m, () -> mat.sumAxis(0));
        bench("sumAxis(1)", m, () -> mat.sumAxis(1));
        bench("meanAxis(1)", m, () -> mat.meanAxis(1));
        bench("maxAxis(0)", m, () -> mat.maxAxis(0));
        bench("argMaxAxis(1)", m, () -> mat.argMaxAxis(1));
        bench("varAxis(1)", m, () -> mat.varAxis(1));
        bench("reduceAxis(custom)", m, () -> mat.reduceAxis(1, (x, y) -> x + y));
        bench("sum(0,1) multi-axis", m, () -> mat.sum(0, 1));
        bench("cumSum (flat)", VEC, () -> vec.cumSum());
        bench("cumSumAxis(1)", m, () -> mat.cumSumAxis(1));
        bench("diff", VEC, () -> vec.diff());
        bench("softmax", VEC, () -> vec.softmax());
        bench("softmaxAxis(1)", m, () -> mat.softmaxAxis(1));
        bench("median", VEC, () -> vec.median());
        bench("quantile(0.9)", VEC, () -> vec.quantile(0.9f));
        bench("percentile(50) axis 1", m, () -> mat.percentile(50, 1));
    }

    static void linearAlgebra() {
        section("Linear algebra");
        long mmOps = (long) MM * MM * MM;
        bench("matMul NxN", mmOps, () -> mm1.matMul(mm2));
        bench("matMul transposed left", mmOps, () -> mm1.transpose().matMul(mm2));
        bench("matMul transposed right", mmOps, () -> mm1.matMul(mm2.transpose()));
        bench("dot 2D.2D", mmOps, () -> mm1.dot(mm2));
        bench("matMul 2D.1D", (long) MM * MM, () -> mm1.matMul(NumJArray.ones(MM)));
        bench("matMul batched", batchA.size() * batchA.shape()[1], () -> batchA.matMul(batchB));
        bench("dot 1D.1D", VEC, () -> vec.dot(vecPos));
        bench("inner", VEC, () -> vec.inner(vecPos));
        bench("outer (1000x1000)", 1_000_000, () -> small.outer(small));
        bench("trace", DIM, () -> mat.trace());
        bench("diag (extract)", DIM, () -> mat.diag());
        bench("tril", (long) DIM * DIM, () -> mat.tril());
        bench("triu", (long) DIM * DIM, () -> mat.triu());
    }

    static void shapeOps() {
        section("Shape manipulation");
        long m = (long) DIM * DIM;
        bench("concat axis 0", 2 * m, () -> mat.concat(matB, 0));
        bench("concat axis 1", 2 * m, () -> mat.concat(matB, 1));
        bench("stack", 2 * m, () -> mat.stack(matB, 0));
        bench("split(0, 4)", m, () -> mat.split(0, 4));
        bench("repeat(2, axis 0)", 2 * m, () -> mat.repeat(2, 0));
        bench("tile(2, 2)", 4 * m, () -> mat.tile(2, 2));
        bench("pad(1 each side)", m, () -> mat.pad(new int[][]{{1, 1}, {1, 1}}));
        bench("roll(1, axis 0)", m, () -> mat.roll(1, 0));
        bench("roll flat", m, () -> mat.roll(7));
        bench("select(0, 5)", DIM, () -> mat.select(0, 5));
        bench("expandDims", m, () -> mat.expandDims(0));
        bench("swapAxes", m, () -> mat.swapAxes(0, 1));
        bench("broadcastTo (view)", m, () -> rowVec.broadcastTo(DIM, DIM));
        int[] idx = new int[100_000];
        for (int i = 0; i < idx.length; i++) idx[i] = (int) ((i * 2654435761L) % VEC);
        bench("take(flat, 100k idx)", idx.length, () -> vec.take(idx));
        int[] rows = new int[DIM];
        for (int i = 0; i < DIM; i++) rows[i] = DIM - 1 - i;
        bench("take(axis 0, reversed)", m, () -> mat.take(0, rows));
    }

    static void sortingAndSets() {
        section("Sort / unique");
        long m = (long) DIM * DIM;
        bench("sort (1D)", VEC, () -> vec.sort());
        bench("sortFlat (2D)", m, () -> mat.sortFlat());
        bench("sort(axis 1)", m, () -> mat.sort(1));
        bench("argsort (1D)", VEC, () -> vec.argsort());
        bench("argsort(axis 1)", m, () -> mat.argsort(1));
        bench("unique", VEC, () -> vec.unique());
        bench("uniqueCounts", VEC, () -> vec.uniqueCounts());
    }

    static void comparisons() {
        section("Comparisons / logic / where");
        bench("greater(scalar)", VEC, () -> vec.greater(0f));
        bench("equal(array)", VEC, () -> vec.equal(vecPos));
        bench("less(array)", VEC, () -> vec.less(vecPos));
        bench("logicalAnd", VEC, () -> mask.logicalAnd(mask));
        bench("logicalNot", VEC, () -> mask.logicalNot());
        bench("isClose", VEC, () -> vec.isClose(vecPos));
        bench("allClose", VEC, () -> vec.allClose(vec));
        bench("nonzero", VEC, () -> mask.nonzero());
        bench("argwhere (2D)", (long) DIM * DIM, () -> mat.greater(0f).argwhere());
        bench("where(cond, x, y)", VEC, () -> NumJArray.where(mask, vec, vecPos));
        bench("where(predicate)", VEC, () -> vec.where(v -> v > 0));
        bench("equals (same data)", VEC, () -> vec.equals(vec.copy()));
        bench("hashCode", VEC, () -> vec.hashCode());
    }

    static void inPlace() {
        section("In-place mutators (setup copy not timed)");
        long m = (long) DIM * DIM;
        benchIn("addi(scalar)", VEC, () -> vec.copy(), x -> x.addi(1f));
        benchIn("muli(scalar)", VEC, () -> vec.copy(), x -> x.muli(2f));
        benchIn("addi(array)", VEC, () -> vec.copy(), x -> x.addi(vecPos));
        benchIn("addi(row) broadcast", m, () -> mat.copy(), x -> x.addi(rowVec));
        benchIn("mapi", VEC, () -> vec.copy(), x -> x.mapi(v -> v * 2f));
        benchIn("sqrti", VEC, () -> vecPos.copy(), x -> x.sqrti());
        benchIn("expi", VEC, () -> vecPos.copy(), x -> x.expi());
        benchIn("sini", VEC, () -> vec.copy(), x -> x.sini());
        benchIn("elementwisei(array, op)", VEC, () -> vec.copy(), x -> x.elementwisei(vecPos, (a, b) -> a + b));
        benchIn("addi on transposed view", m, () -> mat.copy(), x -> x.transpose().addi(1f));
        benchIn("assign(array)", m, () -> NumJArray.zeros(DIM, DIM), x -> x.assign(mat));
        benchIn("assign(array) into view", m, () -> NumJArray.zeros(DIM, DIM), x -> x.transpose().assign(mat));
        benchIn("fill(value)", VEC, () -> NumJArray.zeros(VEC), x -> x.fill(5f));
        benchIn("fill(coordinates)", m, () -> NumJArray.zeros(DIM, DIM), x -> x.fill(co -> co[0] * 10 + co[1]));
        benchIn("putWhere(mask, value)", VEC, () -> vec.copy(), x -> x.putWhere(mask, 0f));
        benchIn("conditionalPut", VEC, () -> vec.copy(), x -> x.conditionalPut(v -> v > 1f, 1f));
    }

    static void fileIO() {
        section("Save / load");
        try {
            Path nj = Files.createTempFile("numj-bench", ".nj");
            Path npy = Files.createTempFile("numj-bench", ".npy");
            try {
                long m = (long) DIM * DIM;
                bench("save (.nj)", m, () -> { mat.save(nj.toString()); return null; });
                bench("load (.nj)", m, () -> NumJArray.load(nj.toString()));
                bench("saveNpy", m, () -> { mat.saveNpy(npy.toString()); return null; });
                bench("loadNpy", m, () -> NumJArray.loadNpy(npy.toString()));
            } finally {
                Files.deleteIfExists(nj);
                Files.deleteIfExists(npy);
            }
        } catch (java.io.IOException e) {
            throw new RuntimeException("File IO benchmarks failed", e);
        }
    }

    // ---------------------------------------------------------------- harness
    static void section(String title) {
        section = title;
        if (only != null && !title.toLowerCase().contains(only)) return;
        System.out.println();
        System.out.println("=== " + title + " ===");
        System.out.printf("%-34s %10s %10s %10s %10s %9s %9s %11s %5s %7s %6s%n",
                "method", "min", "median", "mean", "p95", "stddev", "ns/elem", "alloc/op", "GCs", "GC ms", "iters");
    }

    /** Benchmark an operation with no per-iteration setup. */
    static void bench(String name, long elems, Supplier<Object> op) {
        run(name, elems, () -> null, ignored -> op.get());
    }

    /** Benchmark an in-place operation. setup runs before every iteration and is not measured. */
    static <T> void benchIn(String name, long elems, Supplier<T> setup, Consumer<T> op) {
        run(name, elems, setup, t -> { op.accept(t); return t; });
    }

    static <T> void run(String name, long elems, Supplier<T> setup, Function<T, Object> op) {
        if (only != null && !name.toLowerCase().contains(only) && !section.toLowerCase().contains(only)) return;

        settle();

        // ---- warmup: at least MIN_WARM runs and warmupMs of wall time ----
        long warmStart = System.nanoTime();
        int w = 0;
        while (w < MIN_WARM || (System.nanoTime() - warmStart) / 1_000_000 < warmupMs) {
            T state = setup.get();
            sink = op.apply(state);
            w++;
            if (w >= 1000) break;
        }

        settle();   // start the measured phase from a quiet heap so GC counts mean something

        // ---- measure ----
        long gc0 = gcCount(), gcT0 = gcTime();
        long[] times = new long[MAX_ITERS];
        long[] allocs = new long[MAX_ITERS];
        int iters = 0;
        long phaseStart = System.nanoTime();
        long tid = Thread.currentThread().threadId();
        while (iters < MIN_ITERS || ((System.nanoTime() - phaseStart) / 1_000_000 < measureMs && iters < MAX_ITERS)) {
            T state = setup.get();                      // not measured
            long a0 = TMX.getThreadAllocatedBytes(tid);
            long t0 = System.nanoTime();
            Object r = op.apply(state);
            long t1 = System.nanoTime();
            long a1 = TMX.getThreadAllocatedBytes(tid);
            sink = r;
            times[iters] = t1 - t0;
            allocs[iters] = a1 - a0;
            iters++;
        }
        long gcs = gcCount() - gc0, gcMs = gcTime() - gcT0;

        long[] t = Arrays.copyOf(times, iters);
        long[] a = Arrays.copyOf(allocs, iters);
        Arrays.sort(t);
        Arrays.sort(a);
        double mean = Arrays.stream(t).average().orElse(0);
        double var = Arrays.stream(t).mapToDouble(x -> (x - mean) * (x - mean)).sum() / iters;
        double med = t[iters / 2];
        double p95 = t[Math.min(iters - 1, (int) Math.ceil(iters * 0.95) - 1)];

        Result res = new Result(section, name, elems, iters, t[0], med, mean, p95, Math.sqrt(var), a[iters / 2], gcs, gcMs);
        results.add(res);

        System.out.printf("%-34s %10s %10s %10s %10s %9s %9s %11s %5d %7d %6d%n",
                trunc(name, 34), fmtTime(res.minNs), fmtTime(res.medNs), fmtTime(res.meanNs), fmtTime(res.p95Ns),
                fmtTime(res.stdNs), elems > 0 ? String.format("%.2f", res.medNs / elems) : "-",
                fmtBytes(res.allocBytes), gcs, gcMs, iters);
    }

    /** Ask for a GC and give it a moment, so one benchmark's garbage isn't billed to the next. */
    static void settle() {
        System.gc();
        try { Thread.sleep(30); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }

    static long gcCount() {
        long c = 0;
        for (GarbageCollectorMXBean b : GCS) if (b.getCollectionCount() >= 0) c += b.getCollectionCount();
        return c;
    }

    static long gcTime() {
        long t = 0;
        for (GarbageCollectorMXBean b : GCS) if (b.getCollectionTime() >= 0) t += b.getCollectionTime();
        return t;
    }

    // ---------------------------------------------------------------- reporting
    static void printHeader() {
        Runtime rt = Runtime.getRuntime();
        System.out.println("NumJ benchmark");
        System.out.printf("JVM:    %s %s (%s)%n", System.getProperty("java.vm.name"), System.getProperty("java.version"),
                System.getProperty("java.vendor"));
        System.out.printf("OS/CPU: %s %s, %d cores%n", System.getProperty("os.name"), System.getProperty("os.arch"),
                rt.availableProcessors());
        System.out.printf("Heap:   max %s%n", fmtBytes(rt.maxMemory()));
        StringBuilder gcs = new StringBuilder();
        for (GarbageCollectorMXBean b : GCS) gcs.append(b.getName()).append("  ");
        System.out.println("GC:     " + gcs.toString().trim());
        System.out.printf("Config: n=2^%d, warmup>=%dms, measure>=%dms%s%n", n, warmupMs, measureMs,
                only != null ? ", only=" + only : "");
        if (rt.maxMemory() < 1L << 30)
            System.out.println("WARNING: max heap < 1 GB. Large fixtures may OOM; try -Xmx2g or a smaller --n.");
        System.out.println("Notes:  alloc/op counts the calling thread only (work done on other threads isn't included).");
        System.out.println("        GC ms has coarse (millisecond) resolution; GCs > 0 means timings include collection pauses.");
    }

    static void printSummary() {
        if (results.isEmpty()) { System.out.println("\nNo cases matched."); return; }
        System.out.println("\n================ SUMMARY ================");
        System.out.printf("Cases run: %d | total GCs during measured phases: %d | total GC time: %d ms%n",
                results.size(), results.stream().mapToLong(r -> r.gcCount).sum(),
                results.stream().mapToLong(r -> r.gcMs).sum());

        System.out.println("\nSlowest 10 (median):");
        results.stream().sorted(Comparator.comparingDouble((Result r) -> r.medNs).reversed()).limit(10)
                .forEach(r -> System.out.printf("  %-34s %10s   [%s]%n", trunc(r.name, 34), fmtTime(r.medNs), r.section));

        System.out.println("\nHeaviest allocators (bytes/op):");
        results.stream().sorted(Comparator.comparingLong((Result r) -> r.allocBytes).reversed()).limit(10)
                .forEach(r -> System.out.printf("  %-34s %10s   [%s]%n", trunc(r.name, 34), fmtBytes(r.allocBytes), r.section));

        System.out.println("\nMost GC-triggering:");
        results.stream().filter(r -> r.gcCount > 0)
                .sorted(Comparator.comparingDouble((Result r) -> (double) r.gcCount / r.iters).reversed()).limit(10)
                .forEach(r -> System.out.printf("  %-34s %3d GCs / %3d iters (%d ms)   [%s]%n",
                        trunc(r.name, 34), r.gcCount, r.iters, r.gcMs, r.section));

        System.out.println("\nNoisy results (stddev > 25% of mean; treat with suspicion):");
        results.stream().filter(r -> r.meanNs > 0 && r.stdNs / r.meanNs > 0.25)
                .forEach(r -> System.out.printf("  %-34s mean %s  stddev %s%n", trunc(r.name, 34), fmtTime(r.meanNs), fmtTime(r.stdNs)));

        System.out.println("\nAllocation ~0 on a method that returns a new array usually means it returned a view (good), or the JIT eliminated it.");
    }

    static void writeCsv(String path) throws java.io.IOException {
        try (PrintWriter w = new PrintWriter(path)) {
            w.println("section,method,elements,iterations,min_ns,median_ns,mean_ns,p95_ns,stddev_ns,alloc_bytes_per_op,gc_count,gc_ms");
            for (Result r : results)
                w.printf("\"%s\",\"%s\",%d,%d,%.0f,%.0f,%.0f,%.0f,%.0f,%d,%d,%d%n", r.section, r.name, r.elems, r.iters,
                        r.minNs, r.medNs, r.meanNs, r.p95Ns, r.stdNs, r.allocBytes, r.gcCount, r.gcMs);
        }
        System.out.println("\nWrote " + path);
    }

    static String fmtTime(double ns) {
        if (ns < 1_000) return String.format("%.0f ns", ns);
        if (ns < 1_000_000) return String.format("%.2f us", ns / 1_000);
        if (ns < 1_000_000_000) return String.format("%.2f ms", ns / 1_000_000);
        return String.format("%.2f s", ns / 1_000_000_000);
    }

    static String fmtBytes(long b) {
        if (b < 1024) return b + " B";
        if (b < 1024 * 1024) return String.format("%.1f KB", b / 1024.0);
        if (b < 1024L * 1024 * 1024) return String.format("%.1f MB", b / (1024.0 * 1024));
        return String.format("%.2f GB", b / (1024.0 * 1024 * 1024));
    }

    static String trunc(String s, int max) {
        return s.length() <= max ? s : s.substring(0, max - 1) + "~";
    }
}