// Contracted McMurchie-Davidson pair table builder.
//
// Reads the Cartesian basis + density matrix + ESP points (written by
// scripts/dump_basis.py) and emits exactly the flat table esp_gpu consumes.
//
// For every shell pair and every primitive pair we form
//
//   I_pair(C) = K * sum_{alpha,m} G[alpha,m] * R^alpha * F_m(p R^2),  R = C - P
//
// with G obtained by contracting the Hermite E-coefficients, the (-1)^|t|
// derivative signs and the (2p)^m factors.  All of that is independent of C, so
// it is done once here; the GPU then only evaluates the Chebyshev-like sum.
//
// Build:  g++ -O3 -fopenmp -o bin/build_pairs src/build_pairs.cpp

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <vector>
#include <array>
#include <algorithm>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Basis {
    int n_func = 0, n_prim = 0, n_points = 0, n_atoms = 0, max_prim = 0;
    std::vector<int> fn_atom, fn_nprim, fn_off, fn_shell;
    std::vector<int> fn_lx, fn_ly, fn_lz;
    std::vector<double> fc;                             // 3*n_func
    std::vector<double> pexp, pcoef;
    std::vector<double> dens;                           // n_func^2
    std::vector<double> pts;                            // 3*n_points
    std::vector<double> Z, xyz;
};

template <class T>
static std::vector<T> rd(FILE *f, size_t n) {
    std::vector<T> v(n);
    if (n && fread(v.data(), sizeof(T), n, f) != n) { fprintf(stderr, "short read\n"); exit(1); }
    return v;
}

static Basis load_basis(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    int h[8]; if (fread(h, sizeof(int), 8, f) != 8) { fprintf(stderr, "bad hdr\n"); exit(1); }
    Basis b;
    b.n_func = h[0]; b.n_prim = h[1]; b.n_points = h[2]; b.n_atoms = h[3]; b.max_prim = h[4];
    b.fn_atom = rd<int>(f, b.n_func);
    b.fn_nprim = rd<int>(f, b.n_func);
    b.fn_off = rd<int>(f, b.n_func);
    b.fn_lx = rd<int>(f, b.n_func);
    b.fn_ly = rd<int>(f, b.n_func);
    b.fn_lz = rd<int>(f, b.n_func);
    b.fn_shell = rd<int>(f, b.n_func);
    b.fc = rd<double>(f, (size_t)3 * b.n_func);
    b.pexp = rd<double>(f, b.n_prim);
    b.pcoef = rd<double>(f, b.n_prim);
    b.dens = rd<double>(f, (size_t)b.n_func * b.n_func);
    b.pts = rd<double>(f, (size_t)3 * b.n_points);
    b.Z = rd<double>(f, b.n_atoms);
    b.xyz = rd<double>(f, (size_t)3 * b.n_atoms);
    fclose(f);
    return b;
}

// ---------------------------------------------------------------------------
// a-table:  d^t/dx^t F_0 = sum_j a[t][j] (2p)^(t-j) x^(t-2j) F_{t-j}
// ---------------------------------------------------------------------------
static void a_table(int lmax, std::vector<std::vector<double>> &A) {
    A.assign(lmax + 1, std::vector<double>(lmax / 2 + 2, 0.0));
    A[0][0] = 1.0;
    for (int t = 0; t < lmax; ++t)
        for (size_t j = 0; j < A[t].size(); ++j) {
            double v = -A[t][j];
            if (j >= 1) v += (t - 2 * (int)j + 2) * A[t][j - 1];
            A[t + 1][j] = v;
        }
}

// ---------------------------------------------------------------------------
// Per-L contraction:  etab(t) -> G(alpha, m)
// ---------------------------------------------------------------------------
struct LTable {
    int L = 0, T = 1;
    std::vector<std::array<int, 4>> keys;         // (ax, ay, az, m)
    std::vector<std::vector<int>> rows;           // rows[k] = list of t flat idx
    std::vector<std::vector<double>> vals;        // matching coefficients
    std::vector<double> wmom;                     // prod of int|He_n| moments
    std::vector<int> abs_t;
};

static const double M_ABS[16] = {2.5066282746310002, 2.0, 1.5957691216057308, 4.0,
                                 10.212920779565312, 33.63284766793385, 136.5963625280984,
                                 665.1416330443616, 3768.863619586618, 25506.330595911374,
                                 202331.81842358326, 1872154.2572862155, 2.3e7, 3.1e8,
                                 4.7e9, 8.0e10};

static void make_LTable(int L, LTable &lt) {
    lt.L = L; lt.T = L + 1;
    const int T = lt.T;
    std::vector<std::vector<double>> A;
    a_table(L, A);
    std::map<std::array<int, 4>, std::vector<double>> acc;
    for (int ax = 0; ax < T; ++ax)
        for (int ay = 0; ay < T; ++ay)
            for (int az = 0; az < T; ++az) {
                int ltot = ax + ay + az;
                if (ltot > L) continue;
                int col = (ax * T + ay) * T + az;
                double sgn = ((ltot & 1) ? -1.0 : 1.0);
                for (int jx = 0; jx <= ax / 2; ++jx) {
                    double a1 = A[ax][jx]; if (a1 == 0) continue;
                    for (int jy = 0; jy <= ay / 2; ++jy) {
                        double a2 = A[ay][jy]; if (a2 == 0) continue;
                        for (int jz = 0; jz <= az / 2; ++jz) {
                            double a3 = A[az][jz]; if (a3 == 0) continue;
                            std::array<int, 4> k = {ax - 2 * jx, ay - 2 * jy,
                                                    az - 2 * jz, ltot - jx - jy - jz};
                            auto it = acc.find(k);
                            if (it == acc.end())
                                it = acc.emplace(k, std::vector<double>(T * T * T, 0.0)).first;
                            it->second[col] += sgn * a1 * a2 * a3;
                        }
                    }
                }
            }
    lt.keys.clear(); lt.rows.clear(); lt.vals.clear();
    for (auto &kv : acc) {
        lt.keys.push_back(kv.first);
        std::vector<int> r; std::vector<double> v;
        for (int c = 0; c < T * T * T; ++c)
            if (kv.second[c] != 0.0) { r.push_back(c); v.push_back(kv.second[c]); }
        lt.rows.push_back(std::move(r));
        lt.vals.push_back(std::move(v));
    }
    lt.wmom.assign(T * T * T, 0.0);
    lt.abs_t.assign(T * T * T, 0);
    for (int ax = 0; ax < T; ++ax)
        for (int ay = 0; ay < T; ++ay)
            for (int az = 0; az < T; ++az) {
                int c = (ax * T + ay) * T + az;
                lt.wmom[c] = M_ABS[ax] * M_ABS[ay] * M_ABS[az];
                lt.abs_t[c] = ax + ay + az;
            }
}

// ---------------------------------------------------------------------------
struct OutPair {
    double p, P[3], K;
    int nt, off, nmono, monooff;
};

struct Term { int aidx, m; double g; };
struct Mono { int parent, axis; };

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <basis.bin> <pairs.bin> [tol]\n", argv[0]);
        return 1;
    }
    double tol = (argc > 3) ? atof(argv[3]) : 1e-12;
    Basis b = load_basis(argv[1]);

    // group basis functions by shell (contiguous runs with identical centre+prim list)
    struct Shell { int atom, l, fn0, nfn, prim0, nprim; double c[3]; };
    std::vector<Shell> sh;
    for (int i = 0; i < b.n_func;) {
        int j = i;
        while (j < b.n_func && b.fn_shell[j] == b.fn_shell[i]) ++j;
        Shell s;
        s.atom = b.fn_atom[i]; s.l = b.fn_lx[i] + b.fn_ly[i] + b.fn_lz[i];
        s.fn0 = i; s.nfn = j - i; s.prim0 = b.fn_off[i]; s.nprim = b.fn_nprim[i];
        s.c[0] = b.fc[3 * i]; s.c[1] = b.fc[3 * i + 1]; s.c[2] = b.fc[3 * i + 2];
        sh.push_back(s);
        i = j;
    }
    int nsh = (int)sh.size();
    fprintf(stderr, "shells %d  functions %d\n", nsh, b.n_func);

    std::map<int, LTable> Ltabs;
    for (int L = 0; L <= 8; ++L) { LTable t; make_LTable(L, t); Ltabs[L] = std::move(t); }

    std::vector<OutPair> pairs;
    std::vector<Term> terms;
    std::vector<Mono> monos;
    double nelec = 0.0;      // sum rule: Tr(P S) must equal the MOLDEN electron count

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<OutPair> l_pairs;
        std::vector<Term> l_terms;
        std::vector<Mono> l_monos;
        double l_nelec = 0.0;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
        for (int si = 0; si < nsh; ++si) {
            const Shell &A = sh[si];
            for (int sj = si; sj < nsh; ++sj) {
                const Shell &B = sh[sj];
                // density block
                bool zero = true;
                double dtot = 0.0;
                for (int mi = 0; mi < A.nfn && zero; ++mi)
                    for (int mj = 0; mj < B.nfn; ++mj) {
                        double d = b.dens[(size_t)(A.fn0 + mi) * b.n_func + (B.fn0 + mj)];
                        if (sj != si) d += b.dens[(size_t)(B.fn0 + mj) * b.n_func + (A.fn0 + mi)];
                        if (fabs(d) > 1e-15) { zero = false; break; }
                        dtot += fabs(d);
                    }
                if (zero) continue;
                if (dtot > 0) { /* screening handled per primitive pair */ }

                const LTable &LT = Ltabs[A.l + B.l];
                const int T = LT.T;
                const int nterm = (int)LT.keys.size();
                std::vector<double> etab(T * T * T);
                std::vector<double> G(nterm);
                std::vector<double> wgt(A.nfn * B.nfn);
                const double dx = A.c[0] - B.c[0], dy = A.c[1] - B.c[1], dz = A.c[2] - B.c[2];
                const double ab2 = dx * dx + dy * dy + dz * dz;
                for (int mi = 0; mi < A.nfn; ++mi)
                    for (int mj = 0; mj < B.nfn; ++mj) {
                        double d = b.dens[(size_t)(A.fn0 + mi) * b.n_func + (B.fn0 + mj)];
                        if (sj != si) d += b.dens[(size_t)(B.fn0 + mj) * b.n_func + (A.fn0 + mi)];
                        wgt[mi * B.nfn + mj] = d;
                    }

                for (int ii = 0; ii < A.nprim; ++ii)
                    for (int jj = 0; jj < B.nprim; ++jj) {
                        double al = b.pexp[A.prim0 + ii], be = b.pexp[B.prim0 + jj];
                        double p = al + be;
                        double cc = b.pcoef[A.prim0 + ii] * b.pcoef[B.prim0 + jj];
                        double K = exp(-al * be / p * ab2);
                        if (cc == 0.0 || K < 1e-30) continue;
                        double P[3];
                        for (int k = 0; k < 3; ++k)
                            P[k] = (al * A.c[k] + be * B.c[k]) / p;
                        double xa[3], xb[3];
                        for (int k = 0; k < 3; ++k) { xa[k] = P[k] - A.c[k]; xb[k] = P[k] - B.c[k]; }

                        // 1-D E coefficients per axis
                        const int l1 = A.l, l2 = B.l, tmax = A.l + B.l;
                        int dims[3] = {l1 + 1, l2 + 1, tmax + 1};
                        std::vector<double> E[3];
                        for (int ax = 0; ax < 3; ++ax) {
                            std::vector<double> &e = E[ax];
                            e.assign(dims[0] * dims[1] * dims[2], 0.0);
                            auto at = [&](int i, int j, int t) -> double& {
                                return e[(i * dims[1] + j) * dims[2] + t];
                            };
                            at(0, 0, 0) = 1.0;
                            for (int i = 0; i <= l1; ++i)
                                for (int j = 0; j <= l2; ++j) {
                                    if (i == 0 && j == 0) continue;
                                    int pi = (i > 0) ? i - 1 : i, pj = (i > 0) ? j : j - 1;
                                    double x = (i > 0) ? xa[ax] : xb[ax];
                                    for (int t = 0; t <= tmax; ++t) {
                                        double v = x * at(pi, pj, t);
                                        if (t >= 1) v += at(pi, pj, t - 1) / (2.0 * p);
                                        if (t + 1 <= tmax) v += (t + 1) * at(pi, pj, t + 1);
                                        at(i, j, t) = v;
                                    }
                                }
                        }
                        // etab = sum_{mu nu} w * Ex[t_x] Ey[t_y] Ez[t_z]
                        std::fill(etab.begin(), etab.end(), 0.0);
                        for (int mi = 0; mi < A.nfn; ++mi)
                            for (int mj = 0; mj < B.nfn; ++mj) {
                                double w = wgt[mi * B.nfn + mj] * cc;
                                if (w == 0.0) continue;
                                int lx = b.fn_lx[A.fn0 + mi], ly = b.fn_ly[A.fn0 + mi], lz = b.fn_lz[A.fn0 + mi];
                                int mx = b.fn_lx[B.fn0 + mj], my = b.fn_ly[B.fn0 + mj], mz = b.fn_lz[B.fn0 + mj];
                                const double *ex = &E[0][(lx * dims[1] + mx) * dims[2]];
                                const double *ey = &E[1][(ly * dims[1] + my) * dims[2]];
                                const double *ez = &E[2][(lz * dims[1] + mz) * dims[2]];
                                for (int tx = 0; tx < T; ++tx) {
                                    double cx = ex[tx]; if (cx == 0) continue;
                                    for (int ty = 0; ty < T; ++ty) {
                                        double cxy = cx * ey[ty]; if (cxy == 0) continue;
                                        double *dst = &etab[(tx * T + ty) * T];
                                        for (int tz = 0; tz < T; ++tz) dst[tz] += w * cxy * ez[tz];
                                    }
                                }
                            }
                        // sum rule: int rho = sum_{mu nu} P_{mu nu} <mu|nu>
                        // Hermite Gaussians integrate to delta_{t,0} (pi/p)^{3/2}
                        l_nelec += K * pow(M_PI / p, 1.5) * etab[0];

                        // G[alpha,m] = (2 pi/p) (2p)^m * sum_t M0 (alpha,m) t * etab[t]
                        bool anyg = false;
                        for (int k = 0; k < nterm; ++k) {
                            double s = 0.0;
                            const auto &r = LT.rows[k]; const auto &v = LT.vals[k];
                            for (size_t q = 0; q < r.size(); ++q) s += v[q] * etab[r[q]];
                            double cf = 2.0 * M_PI / p;
                            double g = cf * pow(2.0 * p, LT.keys[k][3]) * s;
                            G[k] = g;
                            if (g != 0.0) anyg = true;
                        }
                        if (!anyg) continue;

                        // screening bound = int |rho_pair| du  (times K/d at runtime)
                        double bound = 0.0;
                        for (int c = 0; c < T * T * T; ++c)
                            bound += fabs(etab[c]) * LT.wmom[c] * pow(2.0 * p, 0.5 * LT.abs_t[c]);
                        bound *= pow(2.0 * p, -1.5);
                        double reff = 1.6 * sqrt((A.l + B.l + 2.0) / p);

                        // global screen over the point set
                        double dmin2 = 1e30;
                        for (int iq = 0; iq < b.n_points; ++iq) {
                            double qx = b.pts[3 * iq] - P[0], qy = b.pts[3 * iq + 1] - P[1],
                                   qz = b.pts[3 * iq + 2] - P[2];
                            double d2 = qx * qx + qy * qy + qz * qz;
                            if (d2 < dmin2) dmin2 = d2;
                        }
                        double dd = sqrt(dmin2) - reff;
                        if (dd < 0.05) dd = 0.05;
                        if (K * bound / dd < tol) continue;

                        // monomial schedule (closure under decrement)
                        std::vector<std::array<int, 3>> alphas;
                        for (auto &k : LT.keys) alphas.push_back({k[0], k[1], k[2]});
                        std::sort(alphas.begin(), alphas.end());
                        alphas.erase(std::unique(alphas.begin(), alphas.end()), alphas.end());
                        std::vector<std::array<int, 3>> grown;
                        for (auto &a : alphas) {
                            std::vector<std::array<int, 3>> stack = {a};
                            while (!stack.empty()) {
                                auto u = stack.back(); stack.pop_back();
                                if (std::find(grown.begin(), grown.end(), u) == grown.end()) {
                                    grown.push_back(u);
                                    for (int k = 0; k < 3; ++k)
                                        if (u[k] > 0) { auto w2 = u; w2[k] -= 1; stack.push_back(w2); }
                                }
                            }
                        }
                        std::sort(grown.begin(), grown.end());
                        std::map<std::array<int, 3>, int> gidx;
                        for (size_t q = 0; q < grown.size(); ++q) gidx[grown[q]] = (int)q;

                        OutPair op;
                        op.p = p; op.P[0] = P[0]; op.P[1] = P[1]; op.P[2] = P[2]; op.K = K;
                        op.nt = nterm;
                        op.off = (int)l_terms.size();
                        op.nmono = (int)grown.size();
                        op.monooff = (int)l_monos.size();
                        for (int k = 0; k < nterm; ++k) {
                            auto key = std::array<int, 3>{LT.keys[k][0], LT.keys[k][1], LT.keys[k][2]};
                            l_terms.push_back({gidx[key], LT.keys[k][3], G[k]});
                        }
                        for (size_t q = 0; q < grown.size(); ++q) {
                            int par = 0, ax_ = 0;
                            for (int k = 0; k < 3; ++k)
                                if (grown[q][k] > 0) {
                                    auto w2 = grown[q]; w2[k] -= 1;
                                    par = gidx[w2]; ax_ = k; break;
                                }
                            l_monos.push_back({par, ax_});
                        }
                        l_pairs.push_back(op);
                    }
            }
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            nelec += l_nelec;
            int poff = (int)pairs.size(), toff = (int)terms.size(), moff = (int)monos.size();
            for (auto &op : l_pairs) { op.off += toff; op.monooff += moff; pairs.push_back(op); }
            terms.insert(terms.end(), l_terms.begin(), l_terms.end());
            monos.insert(monos.end(), l_monos.begin(), l_monos.end());
        }
    }

    // ---- write pairs.bin ------------------------------------------------
    int hdr[8] = {(int)pairs.size(), (int)terms.size(), b.n_points, b.n_atoms, 0, 0, 0, 0};
    for (auto &m : monos) hdr[4] = std::max(hdr[4], 0);
    int maxmono = 0;
    for (auto &op : pairs) maxmono = std::max(maxmono, op.nmono);
    hdr[4] = maxmono;

    FILE *f = fopen(argv[2], "wb");
    fwrite(hdr, sizeof(int), 8, f);
    std::vector<double> d1(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) d1[i] = pairs[i].p;
    fwrite(d1.data(), 8, d1.size(), f);
    for (int k = 0; k < 3; ++k) {
        for (size_t i = 0; i < pairs.size(); ++i) d1[i] = pairs[i].P[k];
        fwrite(d1.data(), 8, d1.size(), f);
    }
    for (size_t i = 0; i < pairs.size(); ++i) d1[i] = pairs[i].K;
    fwrite(d1.data(), 8, d1.size(), f);
    std::vector<int> i1(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) i1[i] = pairs[i].nt;      fwrite(i1.data(), 4, i1.size(), f);
    for (size_t i = 0; i < pairs.size(); ++i) i1[i] = pairs[i].off;     fwrite(i1.data(), 4, i1.size(), f);
    for (size_t i = 0; i < pairs.size(); ++i) i1[i] = pairs[i].nmono;   fwrite(i1.data(), 4, i1.size(), f);
    for (size_t i = 0; i < pairs.size(); ++i) i1[i] = pairs[i].monooff; fwrite(i1.data(), 4, i1.size(), f);
    i1.resize(monos.size());
    for (size_t i = 0; i < monos.size(); ++i) i1[i] = monos[i].parent;   fwrite(i1.data(), 4, i1.size(), f);
    for (size_t i = 0; i < monos.size(); ++i) i1[i] = monos[i].axis;     fwrite(i1.data(), 4, i1.size(), f);
    i1.resize(terms.size());
    for (size_t i = 0; i < terms.size(); ++i) i1[i] = terms[i].aidx;     fwrite(i1.data(), 4, i1.size(), f);
    for (size_t i = 0; i < terms.size(); ++i) i1[i] = terms[i].m;        fwrite(i1.data(), 4, i1.size(), f);
    std::vector<double> g(terms.size());
    for (size_t i = 0; i < terms.size(); ++i) g[i] = terms[i].g;
    fwrite(g.data(), 8, g.size(), f);
    fwrite(b.pts.data(), 8, b.pts.size(), f);
    fwrite(b.Z.data(), 8, b.Z.size(), f);
    fwrite(b.xyz.data(), 8, b.xyz.size(), f);
    fclose(f);
    fprintf(stderr, "pairs %zu terms %zu mono %zu\n", pairs.size(), terms.size(), monos.size());
    fprintf(stderr, "sum rule Tr(P S) = %.9f  (must equal the MOLDEN electron count)\n", nelec);
    return 0;
}
