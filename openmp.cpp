#include "common.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>

double bin_size;

std::vector<std::vector<particle_t*>> bins;
int num_bins;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor, bool boundary) {
    
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    if (!boundary) {
        particle.ax += coef * dx;
        particle.ay += coef * dy;

        neighbor.ax -= coef * dx;
        neighbor.ay -= coef * dy;
    } else {
        #pragma omp atomic
        particle.ax += coef * dx;

        #pragma omp atomic
        particle.ay += coef * dy;

        #pragma omp atomic
        neighbor.ax -= coef * dx;

        #pragma omp atomic
        neighbor.ay -= coef * dy;
    }
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    int bin_x = floor(p.x/bin_size);
    int bin_y = floor(p.y/bin_size);
    int orig_bin = bin_x*num_bins+bin_y;

    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
    
    //rebin
    #pragma omp critical
    {
        bin_x = floor(p.x/bin_size);
        bin_y = floor(p.y/bin_size);
        int new_bin = bin_x*num_bins+bin_y;
        if (orig_bin != new_bin) {
            for(int i = 0; i < bins[orig_bin].size(); ++i){
                if(bins[orig_bin][i] == &p){
                    bins[orig_bin].erase(bins[orig_bin].begin() + i);
                    break;
                }
            }
            bins[new_bin].push_back(&p);
        }
    }
    // bin_x = floor(p.x/bin_size);
    // bin_y = floor(p.y/bin_size);
    // int new_bin = bin_x*num_bins+bin_y;
    // if (orig_bin != new_bin) {
    //     for(int i = 0; i < bins[orig_bin].size(); ++i){
    //         if(bins[orig_bin][i] == &p){
    //             bins[orig_bin].erase(bins[orig_bin].begin() + i);
    //             break;
    //         }
    //     }
    //     bins[new_bin].push_back(&p);
    // }
}


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    
    //initializes bins
    bin_size = fmax(cutoff+0.0001, 0.0005*size);
    num_bins = floor(size/bin_size) + 1;
    bins = std::vector<std::vector<particle_t*>>(num_bins * num_bins);
    for (int i = 0; i < num_parts; ++i) {
        int bin_x = floor(parts[i].x/bin_size);
        int bin_y = floor(parts[i].y/bin_size);
        bins[bin_x*num_bins+bin_y].push_back(&parts[i]);
    }
}

void simulate_bins(std::vector<particle_t*> &bin_parts, std::vector<particle_t*> &neighbors, bool boundary) {
    for (int i = 0; i < bin_parts.size(); ++i) {
        particle_t* part = bin_parts[i];
        for (int j = 0; j < neighbors.size(); ++j) {
            apply_force(*part, *neighbors[j], boundary);
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    
    int id, nthreads;
    id = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    int extra_parts = num_parts % nthreads;

    #pragma omp for
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
    }
    #pragma omp barrier
        
    int bins_per_thread = (num_bins*num_bins) / nthreads;
    int extra_bins = (num_bins*num_bins) % nthreads;
    int start_bin = id*bins_per_thread + std::min(id, extra_bins);
    //threads with lower ID may have to handle extra bin
    int end_bin = (id+1)*bins_per_thread + std::min(id+1, extra_bins);
    //iterate through every bin
    for (int bin = start_bin; bin < end_bin; ++bin) {
        int bin_x = bin/num_bins;
        int bin_y = bin - bin_x*num_bins;
        std::vector<particle_t*> &bin_parts = bins[bin];
        int n_pts = bin_parts.size();
        if (n_pts == 0) continue;

        bool boundary = (bin - start_bin <= num_bins+1) || (end_bin - bin <= num_bins+1);
        //Apply forces for all bins (x', y') such that (x' >= x and y' > y) or x' > x
        //same bin
        for (int i = 0; i < n_pts; ++i) {
            particle_t* part = bin_parts[i];
            for (int j = i + 1; j < n_pts; ++j) {
                apply_force(*part, *bin_parts[j], boundary);
            }
        }
        
        if (bin_x < num_bins - 1) {
            simulate_bins(bin_parts, bins[bin + num_bins], boundary);
            if (bin_y < num_bins - 1) {
                simulate_bins(bin_parts, bins[bin + num_bins + 1], boundary);
            }
            if (bin_y > 0) {
                simulate_bins(bin_parts, bins[bin + num_bins - 1], boundary);
            }
        }
        if (bin_y < num_bins - 1) {
            simulate_bins(bin_parts, bins[bin + 1], boundary);
            
        }
    }
    #pragma omp barrier
    #pragma omp for
    // {   if (omp_get_thread_num() == 0) {
    //     // Move Particles
        for (int i = 0; i < num_parts; ++i) {
            move(parts[i], size);
        }
    //     }
    // }
    #pragma omp barrier
}