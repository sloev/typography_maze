/* Randomised serpentine error diffusion for tools/halftone.py.
 *
 * Floyd-Steinberg diffusion (7/16, 3/16, 5/16, 1/16) run in serpentine
 * order, with two sources of randomness that break up the patterns plain
 * error diffusion leaves in flat tones: the threshold is jittered for every
 * pixel, and the four weights are randomly perturbed (and renormalised) for
 * every pixel. The result is grain placed irregularly at all tone levels,
 * without grid, lattice or "worm" artefacts.
 *
 * in:  h*w doubles, ink coverage 0..1 (overwritten as the error buffer)
 * out: h*w bytes, 1 = ink
 */
#include <stdint.h>

static uint64_t s;
static double rnd(void) {            /* xorshift64*, uniform in [0,1) */
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return (double)((s * 2685821657736338717ULL) >> 11) / 9007199254740992.0;
}

void diffuse(double *in, uint8_t *out, int h, int w,
             double tjit, double wjit, uint64_t seed) {
    s = seed ? seed : 88172645463325252ULL;
    for (int y = 0; y < h; y++) {
        int ltr = (y % 2 == 0);
        for (int k = 0; k < w; k++) {
            int x = ltr ? k : w - 1 - k;
            int d = ltr ? 1 : -1;
            double v = in[(long)y * w + x];
            double t = 0.5 + tjit * (rnd() - 0.5);
            int ink = v > t;
            out[(long)y * w + x] = (uint8_t)ink;
            double e = v - (ink ? 1.0 : 0.0);
            double a = 7 * (1 + wjit * (rnd() - 0.5));
            double b = 3 * (1 + wjit * (rnd() - 0.5));
            double c = 5 * (1 + wjit * (rnd() - 0.5));
            double f = 1 * (1 + wjit * (rnd() - 0.5));
            double n = a + b + c + f;
            int xn = x + d, xp = x - d;
            if (xn >= 0 && xn < w) in[(long)y * w + xn] += e * a / n;
            if (y + 1 < h) {
                long r = (long)(y + 1) * w;
                if (xp >= 0 && xp < w) in[r + xp] += e * b / n;
                in[r + x] += e * c / n;
                if (xn >= 0 && xn < w) in[r + xn] += e * f / n;
            }
        }
    }
}

/* Stochastic clustered-dot ("analog newspaper") threshold field.
 *
 * Dot centres are thrown at random with a minimum spacing (dart throwing),
 * so they are evenly spread but follow no grid, row or lattice. Every pixel
 * gets the distance to its nearest centre, scaled by a random size factor
 * for that centre. Ranked into a uniform 0..1 threshold (in Python), the
 * field makes dots that grow from each centre as the tone darkens, merge in
 * the three-quarter tones and leave irregular white holes in the shadows,
 * as an analog halftone does, but with chaotic placement.
 *
 * T:       h*w floats, output distance field
 * s:       mean dot spacing in pixels
 * rmin:    minimum distance between centres, as a fraction of s
 * wjit:    dot-size randomness (0.2 = +/-10 %)
 */
#include <stdlib.h>
#include <math.h>

#define CAP 12
void dot_field(float *T, int h, int w, double sp, double rmin, double wjit, uint64_t seed) {
    s = seed ? seed : 88172645463325252ULL;
    int gw = (int)(w / sp) + 3, gh = (int)(h / sp) + 3;
    float *px = malloc(sizeof(float) * (size_t)gw * gh * CAP);
    float *py = malloc(sizeof(float) * (size_t)gw * gh * CAP);
    float *pk = malloc(sizeof(float) * (size_t)gw * gh * CAP);
    unsigned char *cnt = calloc((size_t)gw * gh, 1);
    double r2 = (rmin * sp) * (rmin * sp);
    long tries = (long)(12.0 * (double)w * h / (sp * sp));
    for (long n = 0; n < tries; n++) {
        double x = rnd() * (w + 2 * sp) - sp, y = rnd() * (h + 2 * sp) - sp;
        int bx = (int)((x + sp) / sp), by = (int)((y + sp) / sp);
        if (bx < 0 || by < 0 || bx >= gw || by >= gh) continue;
        int ok = 1;
        for (int j = by - 1; j <= by + 1 && ok; j++)
            for (int i = bx - 1; i <= bx + 1 && ok; i++) {
                if (i < 0 || j < 0 || i >= gw || j >= gh) continue;
                long b = (long)j * gw + i;
                for (int k = 0; k < cnt[b]; k++) {
                    double dx = px[b * CAP + k] - x, dy = py[b * CAP + k] - y;
                    if (dx * dx + dy * dy < r2) { ok = 0; break; }
                }
            }
        long b = (long)by * gw + bx;
        if (ok && cnt[b] < CAP) {
            px[b * CAP + cnt[b]] = (float)x; py[b * CAP + cnt[b]] = (float)y;
            pk[b * CAP + cnt[b]] = (float)(1.0 + wjit * (rnd() - 0.5));
            cnt[b]++;
        }
    }
    for (int y = 0; y < h; y++) {
        int by = (int)((y + sp) / sp);
        for (int x = 0; x < w; x++) {
            int bx = (int)((x + sp) / sp);
            double best = 1e30;
            for (int j = by - 2; j <= by + 2; j++)
                for (int i = bx - 2; i <= bx + 2; i++) {
                    if (i < 0 || j < 0 || i >= gw || j >= gh) continue;
                    long b = (long)j * gw + i;
                    for (int k = 0; k < cnt[b]; k++) {
                        double dx = px[b * CAP + k] - x, dy = py[b * CAP + k] - y;
                        double d = sqrt(dx * dx + dy * dy) * pk[b * CAP + k];
                        if (d < best) best = d;
                    }
                }
            T[(long)y * w + x] = (float)best;
        }
    }
    free(px); free(py); free(pk); free(cnt);
}
