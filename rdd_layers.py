from numpy import std, sum, logical_or, abs, exp, array, flipud, zeros, ones, newaxis, logical_and, dot, squeeze, \
    concatenate

from shared_hyperparams import *

def kappa(x):
    return (exp(-x/(tau_L/dt)) - exp(-x/(tau_s/dt)))/((tau_L/dt) - (tau_s/dt))

def get_kappas(n=mem):
    return array([kappa(i+1) for i in range(n)])

kappas = flipud(get_kappas(mem))[:, newaxis] # initialize kappas array

class SpikingFA:
    def __init__(self, size, f_input_size=None, b_input_size=None):
        self.size         = size
        self.f_input_size = f_input_size
        self.b_input_size = b_input_size

        if self.f_input_size is not None:
            self.weight = zeros((self.size, self.f_input_size))

        self.RDD = self.b_input_size is not None

        self.weight        = None
        self.fb_weight     = None
        self.fb_weight_std = None

        if self.RDD:
            self.RDD_params = zeros((self.size, 4, self.b_input_size))
            self.beta       = zeros((self.size, self.b_input_size))
        
        self.reset()

    def reset(self):
        self.v = v_reset*ones((self.size, 1))
        self.B = zeros((self.size, 1))

        self.fired                = zeros((self.size, 1), dtype=bool)
        self.refractory_time_left = zeros((self.size, 1), dtype=int)
        self.spike_hist           = zeros((self.size, mem), dtype=int)

        if self.RDD:
            self.fb_weight      = zeros((self.size, self.b_input_size))
            self.fb_weight_orig = zeros((self.size, self.b_input_size))

            self.u = v_reset*ones((self.size, 1))

            self.RDD_time_left = zeros((self.size, 1))
            self.R             = zeros((self.size, self.b_input_size))
            self.R_pre         = zeros((self.size, self.b_input_size))
            self.n_spikes      = zeros((self.size, 1))
            self.max_u         = zeros((self.size, 1))
    
    def set_weights(self, weight=None, bias=None, fb_weight=None):
        if weight is not None:
            self.weight_orig = weight.copy()
            self.weight = 0.2*self.f_input_size*weight/std(weight)

        if fb_weight is not None:
            self.fb_weight = fb_weight.copy()

            if self.fb_weight_std is None:
                self.fb_weight_std = std(self.fb_weight)

    def update(self, f_input=None, b_input=None, driving_input=None):
        if self.RDD:
            # determine which neurons are just ending their RDD Data integration window
            self.RDD_window_ending_mask = self.RDD_time_left == 1

            # update maximum input drives for units in their RDD Data integration window
            self.max_u[logical_and(self.RDD_time_left > 0, self.u > self.max_u)] = self.u[logical_and(self.RDD_time_left > 0, self.u > self.max_u)]

        # update refractory period timesteps remaining for each neuron
        self.refractory_time_left[self.refractory_time_left > 0] -= 1

        # regression equation is Y_i = alpha + beta*x_i + rho*D + eta
        # alpha is a predefined weight in hyperparams, x_i is the current input, D is the output function (so 1 or 0

        # calculate basal potential
        if driving_input is not None:
            p = dot(driving_input, kappas)
            self.B = alpha*p
        elif f_input is not None:
            p = dot(f_input, kappas)
            self.B = dot(self.weight, p)
        else:
            self.B *= 0

        if self.RDD:
            # calculate apical potential
            q = dot(b_input, kappas)

        # calculate changes in voltages and input drives, and update both
        self.dv_dt = -g_L*self.v + g_D*(self.B - self.v)
        self.v    += dt*self.dv_dt

        if self.RDD:
            self.du_dt = -g_L*self.u + g_D*(self.B - self.u)
            self.u    += dt*self.du_dt

        if self.RDD:
            # update rewards for units in their RDD Data integration window
            self.R[squeeze(self.RDD_time_left > 0)] += q[:, 0]
            self.R_pre[squeeze(self.RDD_time_left == 0)] = q[:, 0]

        # determine which neurons are in a refractory period
        refractory_mask = self.refractory_time_left > 0

        # determine which neurons are above spiking threshold
        threshold_mask = self.v >= spike_threshold

        # neurons above threshold that are not in their refractory period will spike
        self.fired *= False
        self.fired[logical_and(threshold_mask, refractory_mask == False)] = True

        # reset voltages of neurons that spiked
        self.v[self.fired] = v_reset

        # update refractory period timesteps remaining for each neuron
        self.refractory_time_left[self.fired] = refractory_time

        if self.RDD:
            if driving_input is not None:
                # update RDD Data estimates (only neurons whose RDD Data integration window has ended will update their estimate)
                self.update_RDD_estimate()

            # decrement time left in RDD Data integration windows
            self.RDD_time_left[self.RDD_time_left > 0] -= 1

            # reset the input drive to match the voltage, for neurons that are not in an RDD Data integration window
            self.u[self.RDD_time_left == 0] = self.v[self.RDD_time_left == 0]

        if self.RDD:
            # determine which neurons are starting a new RDD Data integration window
            self.new_RDD_window_mask = logical_and(abs(spike_threshold - self.u) <= RDD_init_window, self.RDD_time_left == 0)

            self.RDD_time_left[self.new_RDD_window_mask] = RDD_window

            # update number of spikes that have occurred during RDD Data integration windows
            self.n_spikes[logical_and(self.RDD_time_left > 0, self.fired)] += 1

            # reset RDD Data variables for neurons whose RDD Data integration window has ended
            self.n_spikes[self.RDD_window_ending_mask]      = 0
            self.R[squeeze(self.RDD_window_ending_mask)] = 0
            self.max_u[self.RDD_window_ending_mask]         = 0

        # update spike histories
        self.spike_hist = concatenate([self.spike_hist[:, 1:], self.fired], axis=1)

    def update_RDD_estimate(self):
        # figure out which neurons are at the end of their RDD Data integration window, and either just spiked or almost spiked
        just_spiked_mask   = logical_and(self.RDD_window_ending_mask, logical_and(abs(self.max_u - spike_threshold) <= u_window, self.n_spikes >= 1))[:, 0]
        almost_spiked_mask = logical_and(self.RDD_window_ending_mask, logical_and(abs(self.max_u - spike_threshold) <= u_window, self.n_spikes < 1))[:, 0]

        # update RDD Data estimates for neurons that just spiked or almost spiked
        if sum(just_spiked_mask) > 0:
            self.R[just_spiked_mask] /= RDD_window
            err = self.RDD_params[just_spiked_mask, 2]*self.max_u[just_spiked_mask] + self.RDD_params[just_spiked_mask, 0] - (self.R[just_spiked_mask] - self.R_pre[just_spiked_mask])

            self.RDD_params[just_spiked_mask, 2] -= RDD_eta*err*self.max_u[just_spiked_mask]
            self.RDD_params[just_spiked_mask, 0] -= RDD_eta*err
        if sum(almost_spiked_mask) > 0:
            self.R[almost_spiked_mask] /= RDD_window
            err = self.RDD_params[almost_spiked_mask, 3]*self.max_u[almost_spiked_mask] + self.RDD_params[almost_spiked_mask, 1] - (self.R[almost_spiked_mask] - self.R_pre[almost_spiked_mask])

            self.RDD_params[almost_spiked_mask, 3] -= RDD_eta*err*self.max_u[almost_spiked_mask]
            self.RDD_params[almost_spiked_mask, 1] -= RDD_eta*err

        end_mask = logical_or(just_spiked_mask, almost_spiked_mask)

        self.beta[end_mask] = self.RDD_params[end_mask, 2]*spike_threshold + self.RDD_params[end_mask, 0] - (self.RDD_params[end_mask, 3]*spike_threshold + self.RDD_params[end_mask, 1])
        gamma = self.RDD_params[end_mask, 0] + self.RDD_params[end_mask, 1]
        self.beta[end_mask] += gamma
        self.R[end_mask] = 0

    def update_fb_weights(self):
        mask = self.beta != 0
        if sum(mask) > 0:
            self.fb_weight[mask] = self.beta[mask]*self.fb_weight_std/std(self.beta[mask])
