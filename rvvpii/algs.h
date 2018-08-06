#ifndef ALGS
#define ALGS
typedef struct {
    int n;
    // male_prefs[i*n+j] = r mean male i ranks female j rth
    int* male_prefs;
    int* female_prefs;
} problem;
int* pii(problem* p, int* match, int* reverse_match, int* output_size);
int* pii2(problem* p, int* match, int* reverse_match, int* output_size);
int* pii3(problem* p, int* match, int* reverse_match, int* output_size);
int* pii4(problem* p, int* match, int* reverse_match, int* output_size);
int* dummy(problem* p, int* match, int* reverse_match, int* output_size);
#endif
