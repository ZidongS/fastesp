// Exact electrostatic potential from a Gaussian-basis density.
//
//   V(C) = sum_A Z_A/|C-R_A| - sum_pairs K * sum_k g_k * R^alpha_k * F_{m_k}(p R^2)
//
// The right-hand sum is the contracted McMurchie-Davidson table produced by
// espfast.esp; see that module for the derivation.  Interactions are exact (no
// real-space quadrature) -- the only approximation is the pair screening that
// the exporter already applied.
//
// Usage:  esp_gpu <pairs.bin> <out.txt> [--repeats N]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(e_));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Boys functions F_0..F_M(T), M <= 12
// ---------------------------------------------------------------------------
__device__ __forceinline__ void boys_all(double T, int M, double *F) {
    if (T < 1e-10) {
        for (int m = 0; m <= M; ++m) F[m] = 1.0 / (2 * m + 1);
        return;
    }
    if (T >= (double)M - 0.5) {
        // upward recursion, stable in this regime
        // erfc(sqrt(30)) = 1.4e-14, so above T=30 the erf factor is 1 to
        // double precision and the expensive call can be skipped
        double F0 = (T > 30.0) ? 0.5 * sqrt(M_PI / T)
                               : 0.5 * sqrt(M_PI / T) * erf(sqrt(T));
        F[0] = F0;
        double e = exp(-T);
        double inv2T = 0.5 / T;
        for (int m = 0; m < M; ++m)
            F[m + 1] = ((2 * m + 1) * F[m] - e) * inv2T;
    } else {
        // T small: seed F_M from the Taylor series, recurse downwards
        double s = 0.0, tk = 1.0;
        for (int k = 0; k < 80; ++k) {
            double add = tk / (2 * M + 2 * k + 1);
            s += add;
            if (fabs(add) <= 1e-18 * fabs(s)) break;
            tk *= -T / (k + 1);
        }
        F[M] = s;
        double e = exp(-T);
        for (int m = M; m > 0; --m)
            F[m - 1] = (2.0 * T * F[m] + e) / (2 * m - 1);
    }
}

// ---------------------------------------------------------------------------
struct PairTable {
    int n_pairs = 0, n_terms = 0, n_points = 0, n_atoms = 0, max_mono = 0;
    std::vector<double> pp_exp, pp_cx, pp_cy, pp_cz, pp_K;
    std::vector<int> pp_nt, pp_off, pp_nmono, pp_monooff, mo_parent, mo_axis,
        tm_aidx, tm_m;
    std::vector<double> tm_g, pts, Z, xyz;
};

static void *read_blob(FILE *f, size_t n) {
    void *p = malloc(n);
    if (fread(p, 1, n, f) != n) {
        fprintf(stderr, "short read\n");
        exit(1);
    }
    return p;
}

template <class T>
static std::vector<T> take(FILE *f, size_t n) {
    std::vector<T> v(n);
    if (n && fread(v.data(), sizeof(T), n, f) != n) {
        fprintf(stderr, "short read\n");
        exit(1);
    }
    return v;
}

static PairTable load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    int hdr[8];
    if (fread(hdr, sizeof(int), 8, f) != 8) {
        fprintf(stderr, "bad header\n");
        exit(1);
    }
    PairTable t;
    t.n_pairs = hdr[0];
    t.n_terms = hdr[1];
    t.n_points = hdr[2];
    t.n_atoms = hdr[3];
    t.max_mono = hdr[4];
    t.pp_exp = take<double>(f, t.n_pairs);
    t.pp_cx = take<double>(f, t.n_pairs);
    t.pp_cy = take<double>(f, t.n_pairs);
    t.pp_cz = take<double>(f, t.n_pairs);
    t.pp_K = take<double>(f, t.n_pairs);
    t.pp_nt = take<int>(f, t.n_pairs);
    t.pp_off = take<int>(f, t.n_pairs);
    t.pp_nmono = take<int>(f, t.n_pairs);
    t.pp_monooff = take<int>(f, t.n_pairs);
    int n_mono = t.pp_monooff.empty() ? 0
                                      : t.pp_monooff.back() + t.pp_nmono.back();
    t.mo_parent = take<int>(f, n_mono);
    t.mo_axis = take<int>(f, n_mono);
    t.tm_aidx = take<int>(f, t.n_terms);
    t.tm_m = take<int>(f, t.n_terms);
    t.tm_g = take<double>(f, t.n_terms);
    t.pts = take<double>(f, (size_t)3 * t.n_points);
    t.Z = take<double>(f, t.n_atoms);
    t.xyz = take<double>(f, (size_t)3 * t.n_atoms);
    fclose(f);
    return t;
}

// ---------------------------------------------------------------------------
struct Dev {
    double *pp_exp, *pp_cx, *pp_cy, *pp_cz, *pp_K, *tm_g;
    int *pp_nt, *pp_off, *pp_nmono, *pp_monooff, *mo_parent, *mo_axis, *tm_aidx, *tm_m;
    double *pts, *Z, *xyz;
    int *pp_mmax;
    int n_pairs, n_terms, n_points, n_atoms, max_mono;
};

template <class T>
static T *to_dev(const std::vector<T> &v) {
    T *p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, sizeof(T) * std::max<size_t>(v.size(), 1)));
    if (!v.empty())
        CUDA_CHECK(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
    return p;
}

// ---------------------------------------------------------------------------
// Kernel 1: each block covers TPB points of one pair chunk.
// ---------------------------------------------------------------------------
#define TPB 64
#define MAX_MONO 192
#define MAX_BOYS 16

__global__ void esp_pairs_kernel(
    const int n_pairs, const int n_points,
    const double *__restrict__ pp_exp, const double *__restrict__ pp_cx,
    const double *__restrict__ pp_cy, const double *__restrict__ pp_cz,
    const double *__restrict__ pp_K, const int *__restrict__ pp_nt,
    const int *__restrict__ pp_off, const int *__restrict__ pp_nmono,
    const int *__restrict__ pp_monooff, const int *__restrict__ mo_parent,
    const int *__restrict__ mo_axis, const int *__restrict__ tm_aidx,
    const int *__restrict__ tm_m, const double *__restrict__ tm_g,
    const int *__restrict__ pp_mmax,
    const double *__restrict__ pts, double *__restrict__ partial,
    const int chunk, const int n_chunk)
{
    const int pt = blockIdx.x * TPB + threadIdx.x;
    const int p0 = (int)(((long long)blockIdx.y * n_pairs) / n_chunk);
    const int p1 = (int)(((long long)(blockIdx.y + 1) * n_pairs) / n_chunk);

    double acc = 0.0;
    if (pt < n_points) {
        const double cx = pts[3 * pt], cy = pts[3 * pt + 1], cz = pts[3 * pt + 2];
        double R[3];
        for (int ip = p0; ip < p1; ++ip) {
            R[0] = cx - pp_cx[ip];
            R[1] = cy - pp_cy[ip];
            R[2] = cz - pp_cz[ip];
            const double p = pp_exp[ip];
            const double T = p * (R[0] * R[0] + R[1] * R[1] + R[2] * R[2]);
            const int nt = pp_nt[ip];
            const int off = pp_off[ip];
            const int mo = pp_monooff[ip];
            const int nm = pp_nmono[ip];

            double F[MAX_BOYS];
            boys_all(T, pp_mmax[ip], F);

            double mono[MAX_MONO];
            mono[0] = 1.0;
            for (int j = 1; j < nm; ++j) {
                const int a = mo_axis[mo + j];
                mono[j] = mono[mo_parent[mo + j]] * R[a];
            }
            double s = 0.0;
            for (int k = 0; k < nt; ++k)
                s += tm_g[off + k] * mono[tm_aidx[off + k]] * F[tm_m[off + k]];
            acc += pp_K[ip] * s;
        }
    }
    partial[(size_t)blockIdx.y * (gridDim.x * TPB) + pt] = acc;
}

// Kernel 2: reduce the chunk partials and add the nuclear term.
__global__ void esp_reduce_kernel(
    const int n_points, const int n_chunk, const int width,
    const double *__restrict__ partial, const int n_atoms,
    const double *__restrict__ Z, const double *__restrict__ xyz,
    const double *__restrict__ pts, double *__restrict__ out)
{
    const int pt = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= n_points) return;
    double acc = 0.0;
    for (int c = 0; c < n_chunk; ++c) acc += partial[(size_t)c * width + pt];
    double cx = pts[3 * pt], cy = pts[3 * pt + 1], cz = pts[3 * pt + 2];
    double nuc = 0.0;
    for (int a = 0; a < n_atoms; ++a) {
        double dx = cx - xyz[3 * a], dy = cy - xyz[3 * a + 1], dz = cz - xyz[3 * a + 2];
        nuc += Z[a] / sqrt(dx * dx + dy * dy + dz * dz);
    }
    out[pt] = nuc - acc;
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <pairs.bin> <out.txt> [--repeats N] [--nchunk N]\n", argv[0]);
        return 1;
    }
    int repeats = 1, n_chunk = 4096, quiet = 0;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nchunk") && i + 1 < argc) n_chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
    }

    PairTable t = load(argv[1]);
    const int n_points = t.n_points;
    const int width = ((n_points + TPB - 1) / TPB) * TPB;
    if (!quiet)
        printf("pairs=%d terms=%d points=%d atoms=%d max_mono=%d\n",
               t.n_pairs, t.n_terms, n_points, t.n_atoms, t.max_mono);

    Dev d;
    d.pp_exp = to_dev(t.pp_exp); d.pp_cx = to_dev(t.pp_cx);
    d.pp_cy = to_dev(t.pp_cy);   d.pp_cz = to_dev(t.pp_cz);
    d.pp_K = to_dev(t.pp_K);     d.tm_g = to_dev(t.tm_g);
    d.pp_nt = to_dev(t.pp_nt);   d.pp_off = to_dev(t.pp_off);
    d.pp_nmono = to_dev(t.pp_nmono); d.pp_monooff = to_dev(t.pp_monooff);
    d.mo_parent = to_dev(t.mo_parent); d.mo_axis = to_dev(t.mo_axis);
    d.tm_aidx = to_dev(t.tm_aidx); d.tm_m = to_dev(t.tm_m);
    d.pts = to_dev(t.pts); d.Z = to_dev(t.Z); d.xyz = to_dev(t.xyz);
    d.n_pairs = t.n_pairs; d.n_terms = t.n_terms; d.n_points = n_points;
    d.n_atoms = t.n_atoms; d.max_mono = t.max_mono;

    std::vector<int> h_mmax(t.n_pairs, 0);
    for (int ip = 0; ip < t.n_pairs; ++ip) {
        int off = t.pp_off[ip], nt = t.pp_nt[ip], m = 0;
        for (int k = 0; k < nt; ++k) m = std::max(m, t.tm_m[off + k]);
        h_mmax[ip] = m;
    }
    if (!h_mmax.empty() && *std::max_element(h_mmax.begin(), h_mmax.end()) >= MAX_BOYS) {
        fprintf(stderr, "Boys order exceeds MAX_BOYS\n");
        return 1;
    }
    if (t.max_mono >= MAX_MONO) {
        fprintf(stderr, "monomial count %d exceeds MAX_MONO\n", t.max_mono);
        return 1;
    }
    d.pp_mmax = to_dev(h_mmax);

    double *d_partial = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(double) * (size_t)n_chunk * width));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * n_points));

    dim3 grid1((n_points + TPB - 1) / TPB, n_chunk);
    dim3 blk1(TPB);
    int rblk = 128;
    int rgrid = (n_points + rblk - 1) / rblk;

    // warm-up: the first launch pays for context creation and module load,
    // which must not be charged to the measured solve
    esp_pairs_kernel<<<grid1, blk1>>>(
        d.n_pairs, n_points, d.pp_exp, d.pp_cx, d.pp_cy, d.pp_cz, d.pp_K,
        d.pp_nt, d.pp_off, d.pp_nmono, d.pp_monooff, d.mo_parent, d.mo_axis,
        d.tm_aidx, d.tm_m, d.tm_g, d.pp_mmax, d.pts, d_partial, 0, n_chunk);
    esp_reduce_kernel<<<rgrid, rblk>>>(n_points, n_chunk, width, d_partial,
                                       d.n_atoms, d.Z, d.xyz, d.pts, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        esp_pairs_kernel<<<grid1, blk1>>>(
            d.n_pairs, n_points, d.pp_exp, d.pp_cx, d.pp_cy, d.pp_cz, d.pp_K,
            d.pp_nt, d.pp_off, d.pp_nmono, d.pp_monooff, d.mo_parent, d.mo_axis,
            d.tm_aidx, d.tm_m, d.tm_g, d.pp_mmax, d.pts, d_partial, 0, n_chunk);
        CUDA_CHECK(cudaGetLastError());
        esp_reduce_kernel<<<rgrid, rblk>>>(n_points, n_chunk, width, d_partial,
                                           d.n_atoms, d.Z, d.xyz, d.pts, d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!quiet) printf("run %d: %.3f ms\n", r, ms);
    }

    std::vector<double> host(n_points);
    CUDA_CHECK(cudaMemcpy(host.data(), d_out, sizeof(double) * n_points,
                          cudaMemcpyDeviceToHost));
    FILE *fo = fopen(argv[2], "w");
    fprintf(fo, "%d\n", n_points);
    for (int i = 0; i < n_points; ++i)
        fprintf(fo, " %20.12f %20.12f %20.12f %22.14e\n", t.pts[3 * i],
                t.pts[3 * i + 1], t.pts[3 * i + 2], host[i]);
    fclose(fo);
    if (!quiet) printf("wrote %s\n", argv[2]);
    return 0;
}
