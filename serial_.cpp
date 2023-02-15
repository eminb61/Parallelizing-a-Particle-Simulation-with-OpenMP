#include "common.h"
#include <cmath>
#include <vector>
#include <algorithm>

double bin_size;

std::vector<std::vector<particle_t*>> bins;
int num_bins;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    
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
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void apply_force2(particle_t& particle, particle_t& neighbor) {
    
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
    particle.ax += coef * dx;
    particle.ay += coef * dy;

    neighbor.ax -= coef * dx;
    neighbor.ay -= coef * dy;
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


void init_simulation(particle_t* parts, int num_parts, double size) {
	// You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
    
    //initializes bins
    bin_size = fmax(cutoff+0.0001, 0.005*size);
    num_bins = floor(size/bin_size) + 1;
    bins = std::vector<std::vector<particle_t*>>(num_bins * num_bins);
    for (int i = 0; i < num_parts; ++i) {
        int bin_x = floor(parts[i].x/bin_size);
        int bin_y = floor(parts[i].y/bin_size);
        bins[bin_x*num_bins+bin_y].push_back(&parts[i]);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    for (int i = 0; i < num_parts; ++i) {
        parts[i].ax = parts[i].ay = 0;
    }

    for (int i = 0; i < num_parts; ++i) {

        int bin_x = floor(parts[i].x/bin_size);
        int bin_y = floor(parts[i].y/bin_size);
        
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                int nx = bin_x + dx;
                int ny = bin_y + dy;
                if (!(nx >= 0 && nx < num_bins && ny >= 0 && ny < num_bins) || nx * num_bins + ny < bin_x * num_bins + bin_y) continue;
                std::vector<particle_t*> &neighbors = bins[nx * num_bins + ny];
                int num_neighors = neighbors.size();
                if (nx * num_bins + ny > bin_x * num_bins + bin_y) {
                    for (int j = 0; j < num_neighors; ++j) {
                        apply_force2(parts[i], *bins[nx * num_bins + ny][j]);
                    }
                    continue;
                }
                for (int j = 0; j < num_neighors; ++j) {
                    apply_force(parts[i], *bins[nx * num_bins + ny][j]);
                }
            }
        } 
    }

    // Move Particles
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}
