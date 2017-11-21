/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//number of particles defined
	num_particles = 100;

	std::default_random_engine gen;

	//noise added to sensed approximate position
	std::normal_distribution<double> N_x(x, std[0]);
	std::normal_distribution<double> N_y(y, std[1]);
	std::normal_distribution<double> N_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++)
	{
		//new particle initialized
		Particle particle;
		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 1;

		//new particle added to particles array
		particles.push_back(particle);
		weights.push_back(1);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;

	for (int i = 0; i < num_particles; i++)
	{
		double new_x;
		double new_y;
		double new_theta;

		//new position and orientation values are calculated for each particle
		if (yaw_rate == 0)
		{
			new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
			new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
			new_theta = particles[i].theta;
		}
		else
		{
			new_x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			new_y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			new_theta = particles[i].theta + yaw_rate * delta_t;
		}

		//noise added into newly calculated values
		std::normal_distribution<double> N_x(new_x, std_pos[0]);
		std::normal_distribution<double> N_y(new_y, std_pos[1]);
		std::normal_distribution<double> N_theta(new_theta, std_pos[2]);

		//particle updated with new values
		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int obs=0; obs<observations.size(); obs++)
	{
		double min_dist = 10000000.0;
		double calc_dist;
		int obs_id;
		for(int pred=0; pred<predicted.size(); pred++)
		{
			calc_dist = dist(predicted[pred].x, predicted[pred].y, observations[obs].x, observations[obs].y);
			if(calc_dist < min_dist)
			{
				min_dist = calc_dist;
				obs_id = pred;
			}
		}
		observations[obs].id = obs_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {

	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for(int i=0; i<num_particles; i++){
	        double current_x = particles[i].x;
	        double current_y = particles[i].y;
	        double current_theta = particles[i].theta;

	        vector<LandmarkObs> predicted_landmarks;
	        for(int l=0; l<map_landmarks.landmark_list.size(); l++){
	            int l_id = map_landmarks.landmark_list[l].id_i;
	            double l_x = map_landmarks.landmark_list[l].x_f;
	            double l_y = map_landmarks.landmark_list[l].y_f;

	            double delta_x = l_x - current_x;
	            double delta_y = l_y - current_y;

	            double distance = sqrt(pow(delta_x, 2.0) + pow(delta_y, 2.0));
	            if(distance<=sensor_range){
	                l_x = delta_x * cos(current_theta) + delta_y * sin(current_theta);
	                l_y = delta_y * cos(current_theta) - delta_x * sin(current_theta);
	                LandmarkObs landmark_in_range = {l_id, l_x, l_y};
	                predicted_landmarks.push_back(landmark_in_range);
	            }
	        }

	        dataAssociation(predicted_landmarks, observations);

	        double new_weight = 1.0;
	        for(int obs=0; obs<observations.size(); obs++) {
	            int l_id = observations[obs].id;
	            double obs_x = observations[obs].x;
	            double obs_y = observations[obs].y;

	            double delta_x = obs_x - predicted_landmarks[l_id].x;
	            double delta_y = obs_y - predicted_landmarks[l_id].y;

	            double numerator = exp(- 0.5 * (pow(delta_x,2.0)*std_landmark[0] + pow(delta_y,2.0)*std_landmark[1] ));
	            double denominator = sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
	            new_weight = new_weight * numerator/denominator;
	        }
	        weights[i] = new_weight;
	        particles[i].weight = new_weight;

	    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;

	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> resample_particles;

	for(int i = 0; i < num_particles; i++)
		resample_particles.push_back(particles[distribution(gen)]);

	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
