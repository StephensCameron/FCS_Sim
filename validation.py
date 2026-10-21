import ctypes
from importlib.resources import path
import os
from time import time
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import pickle



##NUM_PARTICLES = 100
#NUM_PARTICLES = ctypes.c_int(5000) # number of particles in the simulation
NUM_PARTICLES = ctypes.c_int(200) # number of particles in the simulation
# for now max number of particles is 1024 until I change number of threads per block in the CUDA kernel
BOUNDARY = ctypes.c_float(10.0) # width of the simulation box in um
D = ctypes.c_float(500.0) # diffusion coefficient in um^2/s
#DELTA_T = ctypes.c_float(0.000_000_1) # time step size in seconds
DELTA_T = ctypes.c_float(1e-5) # time step size in seconds
#DELTA_T = ctypes.c_float(1e-6) # time step size in seconds
#DELTA_T = ctypes.c_float(0.000_01) # time step size in seconds
#DELTA_T = ctypes.c_float(0.0001) # time step size in seconds
#NUM_TIMESTEPS = ctypes.c_int(50_000_000) # number of time steps to simulate
NUM_TIMESTEPS = ctypes.c_int(10_000_00) # number of time steps to simulate
NUM_TIMESTEPS = ctypes.c_int(7_500_000) # number of time steps to simulate
#NUM_TIMESTEPS = ctypes.c_int(7_500_00) # number of time steps to simulate
# #NUM_TIMESTEPS = ctypes.c_int(50_000_00) # number of time steps to simulate
# COUNTS_PER_SECOND = ctypes.c_int(10_000) # molecular brightness of the particles in counts per second
MAX_LAG_SECONDS = 1.0  # int, see the note on autocorrelate()

W0 = ctypes.c_float(1.0)
z0 = ctypes.c_float(2.0)


TOTAL_TIME_LENGTH = ctypes.c_float(DELTA_T.value * NUM_TIMESTEPS.value)

# particle types: arbitrary number of species, each with its own diffusion
# coefficient and molecular brightness. Particles are assigned to types
num_particle_types = ctypes.c_int(2) # number of particle types in the simulation
particle_type_Ds = (ctypes.c_float * num_particle_types.value)(500.0, 800.0) # diffusion coefficients for each particle type in um^2/s
particle_type_counts_per_second = (ctypes.c_float * num_particle_types.value)(10_000.0, 10_000.0) # molecular brightness for each particle type in counts per second

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64")



# class TimeTrace:
#     _fields_ = [("time", ctypes.POINTER(ctypes.c_double)),
#                 ("intensities", ctypes.POINTER(ctypes.c_double)),
#                 ("delta_t", ctypes.c_double),
#                 ("num_timesteps", ctypes.c_int),
#                 ("counts_per_second", ctypes.c_int)]
class TimeTrace(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("time", ctypes.POINTER(ctypes.c_double)),
        ("intensities", ctypes.POINTER(ctypes.c_double)),
        ("delta_t", ctypes.c_double),
        ("num_timesteps", ctypes.c_int),
        ("counts_per_second", ctypes.c_int)
    ]


# mirrors experiment_params_t in src_cuda_enabled/experiment_params.h
class ExperimentParams(ctypes.Structure):
    _fields_ = [
        ("num_particles", ctypes.c_int),
        ("boundary", ctypes.c_float),
        ("delta_t", ctypes.c_float),
        ("num_timesteps", ctypes.c_int),
        ("w0", ctypes.c_float),
        ("z0", ctypes.c_float),
        ("num_particle_types", ctypes.c_int),
        ("particle_type_Ds", ctypes.POINTER(ctypes.c_float)),
        ("particle_type_counts_per_second", ctypes.POINTER(ctypes.c_float)),
    ]


def build_experiment_params(num_particles, boundary, delta_t, num_timesteps, w0, z0,
                             num_particle_types, particle_type_Ds, particle_type_counts_per_second):
    '''
    Builds an ExperimentParams struct to pass to gpu_gen_timetrace. particle_type_Ds and
    particle_type_counts_per_second must be ctypes (c_float * num_particle_types) arrays.
    '''
    return ExperimentParams(
        num_particles=num_particles,
        boundary=boundary,
        delta_t=delta_t,
        num_timesteps=num_timesteps,
        w0=w0,
        z0=z0,
        num_particle_types=num_particle_types,
        particle_type_Ds=ctypes.cast(particle_type_Ds, ctypes.POINTER(ctypes.c_float)),
        particle_type_counts_per_second=ctypes.cast(particle_type_counts_per_second, ctypes.POINTER(ctypes.c_float)),
    )


def get_trace_from_struct(trace):
    I_t = np.zeros(trace.contents.num_timesteps, dtype=np.float64)
    t = np.zeros(trace.contents.num_timesteps, dtype=np.float64)
    for i in range(trace.contents.num_timesteps):
        I_t[i] = trace.contents.intensities[i]
        t[i] = trace.contents.time[i]
    return I_t, t

def get_trace_only_intensity_from_struct(trace):
    I_t = np.zeros(trace.contents.num_timesteps, dtype=np.float64)
    #t = np.zeros(trace.contents.num_timesteps, dtype=np.float64)
    for i in range(trace.contents.num_timesteps):
        I_t[i] = trace.contents.intensities[i]
        #t[i] = trace.contents.time[i]
    return I_t#, t

def plot_trace(I_t, t):
    plt.figure(figsize=(10, 6))
    plt.plot(t, I_t, label="Intensity")
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity (counts/s)")
    plt.title("Intensity vs Time")
    plt.legend()
    plt.show()

def Diffusion_model(tau, N, tau_D, G_inf):
    #alpha = 2.0
    #return (1 / N) * ((1 + (tau / tau_D)) * (1 + (alpha**(2))*(tau / tau_D))) ** -0.5 + 1
    kappa = 2.0
    return (1/N) * ((1 + tau / tau_D) * (1 + (kappa ** -2) * (tau / tau_D)) ** 0.5) ** -1 + G_inf

def Diffusion_model_multi_species(tau, n_species, alpha, G_0, tau_D, G_inf):
    kappa = 2.0
    G = G_0 * np.prod([alpha[i] * ((1 + tau / tau_D[i]) * (1 + (kappa ** -2) * (tau / tau_D[i])) ** 0.5) ** -1 for i in range(n_species)])
    G += G_inf
    return G


def fit_ACF_multi_species(lags, corr, n_species):
    # Initial guesses for the parameters
    alpha_guess = np.ones(n_species)
    G_0_guess = 0.1
    tau_D_guess = np.ones(n_species)
    G_inf_guess = 1.0

    p0 = np.concatenate([alpha_guess, [G_0_guess], tau_D_guess, [G_inf_guess]])

    def model_to_fit(lags, *params):
        alpha = params[:n_species]
        G_0 = params[n_species]
        tau_D = params[n_species+1:2*n_species+1]
        G_inf = params[-1]
        return Diffusion_model_multi_species(lags, n_species, alpha, G_0, tau_D, G_inf)

    popt, pcov = scipy.optimize.curve_fit(model_to_fit, lags, corr, p0=p0)
    return popt, pcov

def print_fit_results_multi_species(popt, n_species):
    alpha = popt[:n_species]
    G_0 = popt[n_species]
    tau_D = popt[n_species+1:2*n_species+1]
    G_inf = popt[-1]

    print("Fitted parameters:")
    for i in range(n_species):
        print(f"Species {i+1}: alpha = {alpha[i]}, tau_D = {tau_D[i]}")
    print(f"G_0 = {G_0}, G_inf = {G_inf}")


def call_my_cublas_autocorrelate(intensities, delta_t, max_lag_seconds):

    lib = ctypes.CDLL(r"DLLs_cuda_enabled/gpu_autocorrelate.dll")

    cuda_autocorrelate = lib.my_cublas_autocorrelate
    cuda_autocorrelate.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    cuda_autocorrelate.restype = ctypes.c_int


    lags = np.arange(0, max_lag_seconds.value, delta_t.value)
    corr = np.zeros(len(lags), dtype=np.float64)
    max_lag_indx = len(lags)

    intensities_f32 = intensities.astype(np.float32)
    corr_f32 = np.zeros(max_lag_indx, dtype=np.float32)

    ret = cuda_autocorrelate(
        intensities_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        corr_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(len(intensities_f32)),
        ctypes.c_int(max_lag_indx)
    )
    if ret != 0:
        raise RuntimeError(f"CUDA autocorrelate failed with error code {ret}")


    n = len(intensities)
    for i in range(len(corr)): # temp until this is implemented in the c code
        corr[i] = corr_f32[i] / n
        n -= 1

    corr[:] = corr / np.mean(intensities) ** 2
    return lags, corr

def fit_ACF(lags, corr):
    '''
    Fits autocorrelation function to a single-component diffusion model. Returns the fitted parameters.
    '''
    p0 = [0.1, 0.0001, 1.0]  # Initial guess for N, tau_D, G_inf
    bounds = ([1.0e-3, 1e-12, 0], [5, 1e-1, 5])  # bounds for N, tau_D, G_inf
    #weights = corr**2/(np.dot(corr, corr)) # temp, need to extimate uncertainty in the autocorrelation and deweight large taus
    weights = np.ones_like(corr)  # temporary uniform weights
    #weights = corr**2
    popt, pcov = scipy.optimize.curve_fit(Diffusion_model, lags, corr, p0=p0, bounds=bounds, sigma=1/weights)
    return popt, pcov

def calculate_N_from_settings(num_particles, boundary, w0, z0):
    '''
    Calculates the expected number of particles in the confocal volume based on the simulation settings.
    '''
    volume = (np.pi**(3/2) * w0**2 * z0)  # confocal volume in um^3
    box_volume = (2*boundary)**3  # simulation box volume in um^3
    concentration = num_particles / box_volume  # particles per um^3
    N = concentration * volume  # expected number of particles in the confocal volume
    return N

def calculate_N_from_ACF(corr):
    '''
    Calculates the number of particles in the confocal volume from the autocorrelation function.
    '''
    N = 1 / corr[0]  # N = 1/G(0)
    return N

def calculate_D_from_ACF(tau_D_fitted, w0, z0):
    '''
    Calculates the diffusion coefficient from the fitted characteristic diffusion time and the confocal volume dimensions.
    '''
    #D = w0**2 / (4 * tau_D_fitted * (z0 / w0))  # assuming 3D Gaussian beam profile
    D = w0**2 / (4 * tau_D_fitted)  # assuming 3D Gaussian beam profile
    return D

def calculate_tau_D_from_D(D, w0, z0):
    '''
    Calculates the characteristic diffusion time from the diffusion coefficient and the confocal volume dimensions.
    '''
    #tau_D = w0**2 / (4 * D * (z0 / w0))  # assuming 3D Gaussian beam profile
    tau_D = w0**2 / (4 * D)  # assuming 3D Gaussian beam profile
    return tau_D

def calculate_D_from_tau_D(tau_D, w0, z0):
    '''
    Calculates the diffusion coefficient from the characteristic diffusion time and the confocal volume dimensions.
    '''
    #D = w0**2 / (4 * tau_D * (z0 / w0))  # assuming 3D Gaussian beam profile
    D = w0**2 / (4 * tau_D)  # assuming 3D Gaussian beam profile
    return D







def main_cuda_enabled():

    path = os.getcwd()
    gpu_timetrace_lib = ctypes.CDLL(r"C:\Documents\CodingProjects\Diffusion_FCS_Sim\DLLs_cuda_enabled\gpu_generate_timetrace.dll")
#int NUM_PARTICLES, float BOUNDARY,  float D, float DELTA_T, int NUM_TIMESTEPS, int COUNTS_PER_SECOND, float W0, float Z0
    gpu_gen_timetrace = gpu_timetrace_lib.gpu_gen_timetrace
    gpu_gen_timetrace.argtypes = [ctypes.POINTER(ExperimentParams)]
    #returns a pointer to a float array containing the intensities of length NUM_TIMESTEPS
    gpu_gen_timetrace.restype = ctypes.POINTER(TimeTrace)
    params = build_experiment_params(
        NUM_PARTICLES.value, BOUNDARY.value, DELTA_T.value, NUM_TIMESTEPS.value, W0.value, z0.value,
        num_particle_types.value, particle_type_Ds, particle_type_counts_per_second
    )
    start_time = time.time()
    trace = gpu_gen_timetrace(ctypes.byref(params))
    end_time = time.time()
    #I_t = np.ctypeslib.as_array(trace, shape=(NUM_TIMESTEPS.value,))
    #print(f"Trace generation time: {end_time - start_time} seconds")
    #trace = gen_timetrace()
    #print(f"Trace generated with {trace.contents.num_timesteps} timesteps, delta_t {trace.contents.delta_t}, counts per second {trace.contents.counts_per_second}")
    #I_t, t = get_trace_from_struct(trace)
    I_t = get_trace_only_intensity_from_struct(trace)
    t = np.arange(0, NUM_TIMESTEPS.value*DELTA_T.value, DELTA_T.value)
    #print(f"First 10 intensities: {I_t[:10]}")
    #print(I_t[:10], t[:10])  # print the first 10 values for verification

    # plt.close()
    # plot_trace(I_t, t)

    
    print(f"Trace generation time: {end_time - start_time} seconds")
    # time each autocorrelation calculation
    # start_time = time.time()
    # lags, corr = my_autocorrelate(I_t, DELTA_T, MAX_LAG_SECONDS)
    # end_time = time.time()
    # print(f"Autocorrelation calculation time: {end_time - start_time} seconds")

    # start_time = time.time()
    # lags, corr = my_autocorrelate_cuda(I_t, DELTA_T, MAX_LAG_SECONDS)
    # end_time = time.time()
    # print(f"GPU (naive) autocorrelation calculation time: {end_time - start_time} seconds")


    start_time = time.time()
    lags, corr = call_my_cublas_autocorrelate(I_t, DELTA_T, MAX_LAG_SECONDS)
    end_time = time.time()
    print(f"GPU (cuBLAS) autocorrelation calculation time: {end_time - start_time} seconds")

    #scipy_FCS_correlation(intensities, DELTA_T, MAX_LAG_SECONDS)


    print(f"First 10 intensities: {I_t[:10]}")
    print(f"First 10 time points: {t[:10]}")
    print(f"simulation length: {t[-1]} seconds")
    print(f"num_timesteps*delta_t: {NUM_TIMESTEPS.value*DELTA_T.value} seconds")

    print(f"\n\nintensities: {I_t[:]}\n\n")
    print(f"average intensity: {np.mean(I_t)}")


    print(f"number of intensities until intenties are near zero: {np.sum(I_t > 1e-26)}")

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(t, I_t)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Intensity (counts/s)")
    ax1.set_title("Intensity vs Time")

    ax2.plot(lags, corr)
    ax2.set_xscale("log")
    ax2.set_xlabel("Time lag (s)")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title("Autocorrelation")

    fig.tight_layout()
    plt.show()


    # compare concentration from simulation settings and from autocorrelation function
    N_from_settings = calculate_N_from_settings(NUM_PARTICLES.value, BOUNDARY.value, W0.value, z0.value)
    N_from_ACF = fit_ACF(lags, corr)[0][0]  # N is the first parameter of the fit

    D_from_settings = D.value
    D_from_ACF = calculate_D_from_ACF(fit_ACF(lags, corr)[0][1], W0.value, z0.value)  # tau_D is the second parameter of the fit

    print(f"Expected diffusion coefficient from settings: {D_from_settings}")
    print(f"Estimated diffusion coefficient from autocorrelation function: {D_from_ACF}")
    print(f"percent error to expected diffusion coefficient: {(D_from_ACF / D_from_settings - 1) * 100}%")

    print(f"Expected number of particles in confocal volume from settings: {N_from_settings}")
    print(f"Estimated number of particles in confocal volume from autocorrelation function: {N_from_ACF}")
    print(f"percent error to expected number of particles: {(N_from_ACF / N_from_settings - 1) * 100}%")

    popt, pcov = fit_ACF(lags, corr)

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(t, I_t)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Intensity (counts/s)")
    ax1.set_title("Intensity vs Time")

    ax2.plot(lags, corr, label="Autocorrelation")
    ax2.plot(lags, Diffusion_model(lags, *popt), label=f"ACF Fit: N={popt[0]:.6f}, D={calculate_D_from_ACF(popt[1], W0.value, z0.value):.2f}")
    ax2.set_xscale("log")
    ax2.set_xlabel("Time lag (s)")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title("Autocorrelation")
    ax2.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(path, f"results\\autocorrelation_{NUM_TIMESTEPS.value}timesteps_{DELTA_T.value}dt_{NUM_PARTICLES.value}particles_GPU.png"))
    plt.show()

    # Example usage of multi-species fitting
    n_species = 2  # for example, fitting two species
    popt_multi, pcov_multi = fit_ACF_multi_species(lags, corr, n_species)
    print_fit_results_multi_species(popt_multi, n_species)
    plt.figure()
    plt.plot(lags, corr, label="Autocorrelation")
    plt.plot(lags, Diffusion_model_multi_species(lags, n_species, popt_multi[:n_species], popt_multi[n_species], popt_multi[n_species+1:2*n_species+1], popt_multi[-1]), label="Multi-species ACF Fit")
    plt.xscale("log")
    plt.xlabel("Time lag (s)")
    plt.ylabel("Autocorrelation")
    plt.title("Multi-species Autocorrelation Fit")
    plt.legend()
    plt.show()


def run_validation_old():
    '''
    This run the simulation at 3 different diffusion coefficients and 3 different concentrations, then
    compares the results from the autocorrelation analysis with the expected values from the simulation settings and plot the comparison
    alongside with the residuals.
    '''

    D_values = [10, 100, 500]
    N_values = [20, 250, 1000]

    timetrace_folder = os.getcwd() + "\\validation_timetraces"
    autocorrelation_folder = os.getcwd() + "\\validation_autocorrelations"

    gpu_timetrace_lib = ctypes.CDLL(r"C:\Documents\CodingProjects\Diffusion_FCS_Sim\DLLs_cuda_enabled\gpu_generate_timetrace.dll")
#int NUM_PARTICLES, float BOUNDARY,  float D, float DELTA_T, int NUM_TIMESTEPS, int COUNTS_PER_SECOND, float W0, float Z0
    gpu_gen_timetrace = gpu_timetrace_lib.gpu_gen_timetrace
    gpu_gen_timetrace.argtypes = [ctypes.POINTER(ExperimentParams)]
    #returns a pointer to a float array containing the intensities of length NUM_TIMESTEPS
    gpu_gen_timetrace.restype = ctypes.POINTER(TimeTrace)
    num_particle_types = ctypes.c_int(1) # number of particle types in the simulation
    particle_type_Ds = (ctypes.c_float * num_particle_types.value)(500.0) # diffusion coefficients for each particle type in um^2/s
    particle_type_counts_per_second = (ctypes.c_float * num_particle_types.value)(10_000.0) # molecular brightness for each particle type in counts per second
    NUM_TIMESTEPS = ctypes.c_int(7_500_0000)
    
    DELTA_T = ctypes.c_float(1e-6) # time step size in seconds
    MAX_LAG_SECONDS = ctypes.c_float(1.0) # maximum lag time for autocorrelation in seconds

    for N in N_values:
        for D in D_values:
            particle_type_Ds = (ctypes.c_float * num_particle_types.value)(D) # diffusion coefficients for each particle type in um^2/s
            NUM_PARTICLES = ctypes.c_int(N) # number of particles in the simulation
            params = build_experiment_params(
                        NUM_PARTICLES.value, BOUNDARY.value, DELTA_T.value, NUM_TIMESTEPS.value, W0.value, z0.value,
                        num_particle_types.value, particle_type_Ds, particle_type_counts_per_second
            )
            if not os.path.exists(timetrace_folder):
                os.makedirs(timetrace_folder)
            if not os.path.exists(timetrace_folder + f"\\D_{D}_N_{N}.pkl"):

                trace = gpu_gen_timetrace(ctypes.byref(params))
                I_t = get_trace_only_intensity_from_struct(trace)
                t = np.arange(0, NUM_TIMESTEPS.value*DELTA_T.value, DELTA_T.value)
                with open(timetrace_folder + f"\\D_{D}_N_{N}.pkl", "wb") as f:
                    pickle.dump((I_t, t), f)
                print(f"Generated and saved timetrace for D={D}, N={N}")
            else:
                 I_t, t = pickle.load(open(timetrace_folder + f"\\D_{D}_N_{N}.pkl", "rb"))
            print(f"Loaded timetrace for D={D}, N={N}")


            if not os.path.exists(autocorrelation_folder):
                os.makedirs(autocorrelation_folder)
            autocorrelation_path = autocorrelation_folder + f"\\D_{D}_N_{N}.pkl"
            # check if autocorrelation already exists, if not use my_cublas_autocorrelate
            if os.path.exists(autocorrelation_path):
                with open(autocorrelation_path, "rb") as f:
                    data = pickle.load(f)
                    lags = data["lags"]
                    corr = data["corr"]
                print(f"Loaded autocorrelation for D={D}, N={N}")
            else:
                lags, corr = call_my_cublas_autocorrelate(I_t, DELTA_T, MAX_LAG_SECONDS)
                with open(autocorrelation_path, "wb") as f:
                    pickle.dump({"lags": lags, "corr": corr}, f)
                print(f"Generated and saved autocorrelation for D={D}, N={N}")

            # Optionally, you can also plot the loaded timetrace for verification
            # plt.figure()
            # plt.plot(t, I_t)
            # plt.xlabel("Time (s)")
            # plt.ylabel("Intensity")
            # plt.title(f"Timetrace for D={D}, N={N}")
            # plt.show()

            # bin timetrace by milisecond
            bin_size = int(1e-3 / DELTA_T.value)  # number of timesteps per millisecond
            I_t_binned = np.bincount(np.arange(len(I_t)) // bin_size, weights=I_t) / np.bincount(np.arange(len(I_t)) // bin_size)
            t_binned = np.bincount(np.arange(len(t)) // bin_size, weights=t) / np.bincount(np.arange(len(t)) // bin_size)


            ## plot the generated or loaded timetrace vs fitting t to a model function with the expected diffusion behavior
            plt.figure()
            plt.plot(t_binned, I_t_binned, label="Simulated Time Trace")
            plt.xlabel("Time (s)")
            plt.ylabel("Intensity (counts per ms)")
            plt.title(f"Timetrace for D={D}, N={N}")
            plt.legend()
            #plt.show()



            # compare autocorrelation to model
            #lags = np.logspace(-6, np.log10(MAX_LAG_SECONDS), num=1000)
            corr_model = Diffusion_model(lags, (calculate_N_from_settings(N, BOUNDARY.value, W0.value, z0.value)), calculate_tau_D_from_D(D, W0.value, z0.value), 1)

            plt.figure()
            plt.xscale("log")
            plt.plot(lags, corr, label="Autocorrelation")
            plt.plot(lags, corr_model, label="Model")
            plt.xlabel("Lag Time (s)")
            plt.ylabel("Autocorrelation")
            plt.title(f"Autocorrelation for D={D}, N={N}")
            plt.legend()
            #plt.savefig(f"autocorrelation_D={D}_N={N}.png")
            #plt.show()

            




    pass

def run_validation():
    '''
    This run the simulation at 3 different diffusion coefficients and 3 different concentrations, then
    compares the results from the autocorrelation analysis with the expected values from the simulation settings and plot the comparison
    alongside with the residuals.
    '''

    D_values = [10, 100, 500]
    N_values = [20, 250, 500]

    timetrace_folder = os.getcwd() + "\\validation_timetraces"
    autocorrelation_folder = os.getcwd() + "\\validation_autocorrelations"

    gpu_timetrace_lib = ctypes.CDLL(r"C:\Documents\CodingProjects\Diffusion_FCS_Sim\DLLs_cuda_enabled\gpu_generate_timetrace.dll")
#int NUM_PARTICLES, float BOUNDARY,  float D, float DELTA_T, int NUM_TIMESTEPS, int COUNTS_PER_SECOND, float W0, float Z0
    gpu_gen_timetrace = gpu_timetrace_lib.gpu_gen_timetrace
    gpu_gen_timetrace.argtypes = [ctypes.POINTER(ExperimentParams)]
    #returns a pointer to a float array containing the intensities of length NUM_TIMESTEPS
    gpu_gen_timetrace.restype = ctypes.POINTER(TimeTrace)
    num_particle_types = ctypes.c_int(1) # number of particle types in the simulation
    particle_type_Ds = (ctypes.c_float * num_particle_types.value)(500.0) # diffusion coefficients for each particle type in um^2/s
    particle_type_counts_per_second = (ctypes.c_float * num_particle_types.value)(10_000.0) # molecular brightness for each particle type in counts per second
    NUM_TIMESTEPS = ctypes.c_int(7_500_0000)
    
    DELTA_T = ctypes.c_float(1e-6) # time step size in seconds
    MAX_LAG_SECONDS = ctypes.c_float(1.0) # maximum lag time for autocorrelation in seconds



    # plot FCS autocorrelations with one subplot per N value, each followed by its binned timetraces
    fig, axes = plt.subplots(2, len(N_values), figsize=(7 * len(N_values), 6),
                             gridspec_kw={"height_ratios": [2, 1]}, squeeze=False)
    acf_data = {}

    for i, N in enumerate(N_values):
        ax, ax_trace = axes[0, i], axes[1, i]
        for D in D_values:
            particle_type_Ds = (ctypes.c_float * num_particle_types.value)(D) # diffusion coefficients for each particle type in um^2/s
            NUM_PARTICLES = ctypes.c_int(N) # number of particles in the simulation
            params = build_experiment_params(
                        NUM_PARTICLES.value, BOUNDARY.value, DELTA_T.value, NUM_TIMESTEPS.value, W0.value, z0.value,
                        num_particle_types.value, particle_type_Ds, particle_type_counts_per_second
            )
            if not os.path.exists(timetrace_folder):
                os.makedirs(timetrace_folder)
            if not os.path.exists(timetrace_folder + f"\\D_{D}_N_{N}.pkl"):

                trace = gpu_gen_timetrace(ctypes.byref(params))
                I_t = get_trace_only_intensity_from_struct(trace)
                t = np.arange(0, NUM_TIMESTEPS.value*DELTA_T.value, DELTA_T.value)
                with open(timetrace_folder + f"\\D_{D}_N_{N}.pkl", "wb") as f:
                    pickle.dump((I_t, t), f)
                print(f"Generated and saved timetrace for D={D}, N={N}")
            else:
                 I_t, t = pickle.load(open(timetrace_folder + f"\\D_{D}_N_{N}.pkl", "rb"))
            print(f"Loaded timetrace for D={D}, N={N}")


            if not os.path.exists(autocorrelation_folder):
                os.makedirs(autocorrelation_folder)
            autocorrelation_path = autocorrelation_folder + f"\\D_{D}_N_{N}.pkl"
            # check if autocorrelation already exists, if not use my_cublas_autocorrelate
            if os.path.exists(autocorrelation_path):
                with open(autocorrelation_path, "rb") as f:
                    data = pickle.load(f)
                    lags = data["lags"]
                    corr = data["corr"]
                print(f"Loaded autocorrelation for D={D}, N={N}")
            else:
                lags, corr = call_my_cublas_autocorrelate(I_t, DELTA_T, MAX_LAG_SECONDS)
                with open(autocorrelation_path, "wb") as f:
                    pickle.dump({"lags": lags, "corr": corr}, f)
                print(f"Generated and saved autocorrelation for D={D}, N={N}")

            # Optionally, you can also plot the loaded timetrace for verification
            # plt.figure()
            # plt.plot(t, I_t)
            # plt.xlabel("Time (s)")
            # plt.ylabel("Intensity")
            # plt.title(f"Timetrace for D={D}, N={N}")
            # plt.show()

            # bin timetrace by milisecond
            bin_size = int(1e-3 / DELTA_T.value)  # number of timesteps per millisecond
            I_t_binned = np.bincount(np.arange(len(I_t)) // bin_size, weights=I_t) / np.bincount(np.arange(len(I_t)) // bin_size)
            t_binned = np.bincount(np.arange(len(t)) // bin_size, weights=t) / np.bincount(np.arange(len(t)) // bin_size)


            ## plot the generated or loaded timetrace vs fitting t to a model function with the expected diffusion behavior
            # plt.figure()
            # plt.plot(t_binned, I_t_binned, label="Simulated Time Trace")
            # plt.xlabel("Time (s)")
            # plt.ylabel("Intensity (counts per ms)")
            # plt.title(f"Timetrace for D={D}, N={N}")
            # plt.legend()
            #plt.show()



            # compare autocorrelation to model
            #lags = np.logspace(-6, np.log10(MAX_LAG_SECONDS), num=1000)
            corr_model = Diffusion_model(lags, (calculate_N_from_settings(N, BOUNDARY.value, W0.value, z0.value)), calculate_tau_D_from_D(D, W0.value, z0.value), 1)

            # plt.figure()
            # plt.xscale("log")
            # plt.plot(lags, corr, label="Autocorrelation")
            # plt.plot(lags, corr_model, label="Model")
            # plt.xlabel("Lag Time (s)")
            # plt.ylabel("Autocorrelation")
            # plt.title(f"Autocorrelation for D={D}, N={N}")
            # plt.legend()
            #plt.savefig(f"autocorrelation_D={D}_N={N}.png")
            #plt.show()

            acf_data[(N, D)] = (lags, corr)
            ax.plot(lags, corr, label=f"Autocorrelation_D={D}_N={N}")
            ax_trace.plot(t_binned, I_t_binned, linewidth=0.5, alpha=0.7, label=f"D={D}")
            #ax.plot(lags, corr_model, label=f"Model_D={D}_N={N}")
            if N == 1000:
                print(f"Autocorrelation for D={D}, N={N}")
                print(corr)
                print(f"timetrace for D={D}, N={N}")
                print(I_t)
                print(f"avg timetrace for D={D}, N={N}")
                print(np.mean(I_t))
                print(f"binned timetrace for D={D}, N={N}")
                print(I_t_binned)
                print(f"avg binned timetrace for D={D}, N={N}")
                print(np.mean(I_t_binned))

        ax.set_xscale("log")
        ax.set_xlabel("Lag Time (s)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Autocorrelation for N={N}")
        ax.legend()
        ax_trace.set_xlabel("Time (s)")
        ax_trace.set_ylabel("Intensity (1 ms bins)")
        ax_trace.set_title(f"Binned timetraces for N={N}")
        ax_trace.legend(fontsize=7)
    fig.tight_layout()
    figures_folder = os.getcwd() + "\\validation_figures"
    os.makedirs(figures_folder, exist_ok=True)
    fig.savefig(figures_folder + "\\autocorrelations_and_timetraces.png", dpi=200)
    plt.show()

    # fit each dataset to a single-species model and compare with the expected curve from the settings
    fig, axes = plt.subplots(1, len(N_values), figsize=(7 * len(N_values), 5), squeeze=False)
    axes = axes[0]
    print(f"{'N_set':>8} {'D_true':>8} {'D_fit':>10} {'N_exp':>9} {'N_fit':>10}")
    single_fit_rows = []
    for ax, N in zip(axes, N_values):
        N_expected = calculate_N_from_settings(N, BOUNDARY.value, W0.value, z0.value)
        for D in D_values:
            lags, corr = acf_data[(N, D)]
            popt, pcov = fit_ACF(lags, corr)
            D_fit = calculate_D_from_ACF(popt[1], W0.value, z0.value)
            print(f"{N:>8} {D:>8.1f} {D_fit:>10.2f} {N_expected:>9.4f} {popt[0]:>10.4f}")
            single_fit_rows.append((N, D, D_fit, N_expected, popt[0]))

            corr_model = Diffusion_model(lags, N_expected, calculate_tau_D_from_D(D, W0.value, z0.value), 1)
            line, = ax.plot(lags, corr, linewidth=3, alpha=0.35, label=f"Data, D={D}")
            ax.plot(lags, Diffusion_model(lags, *popt), linewidth=1, color=line.get_color(),
                    label=f"Fit, D={D} (D_fit={D_fit:.1f}, N_fit={popt[0]:.3f})")
            ax.plot(lags, corr_model, linewidth=0.8, linestyle="--", color="k", alpha=0.6,
                    label="Expected" if D == D_values[0] else None)
        ax.set_xscale("log")
        ax.set_xlabel("Lag Time (s)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Single-Species Fit vs Expected for N={N}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_folder + "\\single_species_fits.png", dpi=200)
    plt.show()

    with open(figures_folder + "\\single_species_fit_table.csv", "w") as f:
        f.write("N_set,D_true,D_fit,N_exp,N_fit\n")
        for row in single_fit_rows:
            f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.6f},{row[4]:.6f}\n")

    labels = [f"N={r[0]}\nD={r[1]}" for r in single_fit_rows]
    x = np.arange(len(single_fit_rows))
    fig, (ax_D, ax_N) = plt.subplots(1, 2, figsize=(14, 5))
    ax_D.bar(x - 0.2, [r[1] for r in single_fit_rows], width=0.4, label="True D")
    ax_D.bar(x + 0.2, [r[2] for r in single_fit_rows], width=0.4, label="Fitted D")
    ax_D.set_ylabel("D (um^2/s)")
    ax_D.set_title("True vs Fitted D")
    ax_N.bar(x - 0.2, [r[3] for r in single_fit_rows], width=0.4, label="Expected N")
    ax_N.bar(x + 0.2, [r[4] for r in single_fit_rows], width=0.4, label="Fitted N")
    ax_N.set_ylabel("N")
    ax_N.set_title("Expected vs Fitted N")
    for ax_i in (ax_D, ax_N):
        ax_i.set_xticks(x)
        ax_i.set_xticklabels(labels, fontsize=7)
        ax_i.legend()
    fig.tight_layout()
    fig.savefig(figures_folder + "\\single_species_fit_bars.png", dpi=200)
    plt.show()

def two_species_diffusion_model(tau, N1, tau_D1, N2, tau_D2, G_inf):
    """
    This model is assuming that each species has the same molecular brightness.
    """
    kappa = 2.0
    tau_D2 = 2.5e-3
    #note: this 'true N' value is incorrect
    #N1 = 0.1740
    #N2 = 0.1740
    return (1/N1) * ((1 + tau / tau_D1) * (1 + (kappa ** -2) * (tau / tau_D1)) ** 0.5) ** -1 + \
            (1/N2) * ((1 + tau / tau_D2) * (1 + (kappa ** -2) * (tau / tau_D2)) ** 0.5) ** -1 + G_inf

def fit_two_species_autocorrelation(lags, corr):
    '''
    Fits the two-species autocorrelation data to a model.

    Returns fit parameters for the two-species model:
    [D1, D2, N1, N2]
    '''

    bounds_lower = (0, 0, 0, 0, 0)
    bounds_upper = (np.inf, 1, np.inf, 1, 2.0)
    p0 = [0.15, 1e-3, 0.15, 2e-3, 1]
    
    popt, pcov = scipy.optimize.curve_fit(two_species_diffusion_model, lags, corr, p0=p0, bounds=(bounds_lower, bounds_upper))
    N1_fit, tau_D1_fit, N2_fit, tau_D2_fit, G_inf_fit = popt
    D1_fit = calculate_D_from_tau_D(tau_D1_fit, W0.value, z0.value)
    D2_fit = calculate_D_from_tau_D(tau_D2_fit, W0.value, z0.value)
    return [D1_fit, D2_fit, N1_fit, N2_fit], popt, pcov


def two_species_validation():
    '''
    Runs a two-species simulation with a fixed second species (D2=100) and a varying
    first species diffusion coefficient, then compares the resulting autocorrelation
    curves. N1=N2=125, so particles are assigned round-robin between the two types.
    '''

    D1_values = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    D1_values = [150, 200, 250, 300, 350, 400, 450, 500]
    D2 = 100
    N1 = 125
    N2 = 125
    NUM_PARTICLES_local = N1 + N2

    data_folder = os.getcwd() + "\\two_species_data"

    gpu_timetrace_lib = ctypes.CDLL(r"C:\Documents\CodingProjects\Diffusion_FCS_Sim\DLLs_cuda_enabled\gpu_generate_timetrace.dll")
    gpu_gen_timetrace = gpu_timetrace_lib.gpu_gen_timetrace
    gpu_gen_timetrace.argtypes = [ctypes.POINTER(ExperimentParams)]
    gpu_gen_timetrace.restype = ctypes.POINTER(TimeTrace)

    num_particle_types_local = ctypes.c_int(2)
    particle_type_counts_per_second_local = (ctypes.c_float * num_particle_types_local.value)(10_000.0, 10_000.0)

    NUM_TIMESTEPS_local = ctypes.c_int(7_500_000)
    DELTA_T_local = ctypes.c_float(1e-5)
    MAX_LAG_SECONDS_local = ctypes.c_float(1.0)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    fig, ax = plt.subplots(figsize=(10, 6))
    acf_data = {}

    for D1 in D1_values:
        particle_type_Ds_local = (ctypes.c_float * num_particle_types_local.value)(D1, D2)
        NUM_PARTICLES_ctype = ctypes.c_int(NUM_PARTICLES_local)
        params = build_experiment_params(
                    NUM_PARTICLES_ctype.value, BOUNDARY.value, DELTA_T_local.value, NUM_TIMESTEPS_local.value, W0.value, z0.value,
                    num_particle_types_local.value, particle_type_Ds_local, particle_type_counts_per_second_local
        )

        timetrace_path = data_folder + f"\\timetrace_D1_{D1}_D2_{D2}.pkl"
        if not os.path.exists(timetrace_path):
            trace = gpu_gen_timetrace(ctypes.byref(params))
            I_t = get_trace_only_intensity_from_struct(trace)
            t = np.arange(0, NUM_TIMESTEPS_local.value * DELTA_T_local.value, DELTA_T_local.value)
            with open(timetrace_path, "wb") as f:
                pickle.dump((I_t, t), f)
            print(f"Generated and saved timetrace for D1={D1}, D2={D2}")
        else:
            I_t, t = pickle.load(open(timetrace_path, "rb"))
            print(f"Loaded timetrace for D1={D1}, D2={D2}")

        autocorrelation_path = data_folder + f"\\autocorr_D1_{D1}_D2_{D2}.pkl"
        if os.path.exists(autocorrelation_path):
            with open(autocorrelation_path, "rb") as f:
                data = pickle.load(f)
                lags = data["lags"]
                corr = data["corr"]
            print(f"Loaded autocorrelation for D1={D1}, D2={D2}")
        else:
            lags, corr = call_my_cublas_autocorrelate(I_t, DELTA_T_local, MAX_LAG_SECONDS_local)
            with open(autocorrelation_path, "wb") as f:
                pickle.dump({"lags": lags, "corr": corr}, f)
            print(f"Generated and saved autocorrelation for D1={D1}, D2={D2}")

        acf_data[D1] = (lags, corr)
        ax.plot(lags, corr, label=f"D1={D1}_D2={D2}")

    ax.set_xscale("log")
    ax.set_xlabel("Lag Time (s)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Two-Species Autocorrelation for Varying D1")
    ax.legend()
    plt.show()

        # check error in the fitted diffusion coefficients compared to the true values for each dataset
    # also plot the fitted autocorrelation against the true autocorrelation for each dataset
    plt.close('all')
    plt.figure()
    for D1 in D1_values:
        D_fits, popt, pcov = fit_two_species_autocorrelation(lags, corr)
        # plot the fitted autocorrelation against the true autocorrelation
        fitted_corr = two_species_diffusion_model(lags, *popt)
        plt.plot(lags, corr, label=f"True, D1={D1}, D2={D2}")
        plt.plot(lags, fitted_corr, label=f"Fitted, D1={D1}, D2={D2}", linestyle="--")
    plt.xscale("log")
    plt.xlabel("Lag Time (s)")
    plt.ylabel("Autocorrelation")
    plt.title(f"Autocorrelation for D1={D1}")
    plt.legend()
    plt.show()

    


    # check error in the fitted diffusion coefficients compared to the true values for each dataset
    # also plot the fitted autocorrelation against the true autocorrelation for each dataset
    plt.close('all')
    plt.figure()
    N_true = calculate_N_from_settings(N1, BOUNDARY.value, W0.value, z0.value)
    ratios = []
    errors = {"D1": [], "D2": [], "N1": [], "N2": []}
    fit_rows = []
    for D1 in D1_values:
        lags, corr = acf_data[D1]
        D_fits, popt, pcov = fit_two_species_autocorrelation(lags, corr)
        # plot the fitted autocorrelation against the true autocorrelation
        fitted_corr = two_species_diffusion_model(lags, *popt)
        plt.plot(lags, corr, label=f"True, D1={D1}, D2={D2}")
        plt.plot(lags, fitted_corr, label=f"Fitted, D1={D1}, D2={D2}", linestyle="--")

        D1_fit, D2_fit, N1_fit, N2_fit = D_fits
        ratios.append(D1 / D2)
        fit_rows.append((D1, D1_fit, D2, D2_fit, N_true, N1_fit, N_true, N2_fit))
        errors["D1"].append((D1_fit / D1 - 1) * 100)
        errors["D2"].append((D2_fit / D2 - 1) * 100)
        errors["N1"].append((N1_fit / N_true - 1) * 100)
        errors["N2"].append((N2_fit / N_true - 1) * 100)
    plt.xscale("log")
    plt.xlabel("Lag Time (s)")
    plt.ylabel("Autocorrelation")
    plt.title("Two-Species Fits vs True Autocorrelation")
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax_i, (name, err) in zip(axes.flat, errors.items()):
        ax_i.plot(ratios, err, marker="o")
        ax_i.axhline(0, color="k", linewidth=0.5)
        ax_i.set_xlabel("D1 / D2")
        ax_i.set_ylabel("Percent error")
        ax_i.set_title(f"{name} error")
    fig.suptitle("Two-Species Fit Error vs Diffusion Coefficient Ratio")
    fig.tight_layout()
    plt.show()

    # Euclidean norms of the percent-error vectors, separately for D and N
    D_error_magnitude = np.linalg.norm(np.array([errors["D1"], errors["D2"]]), axis=0)
    N_error_magnitude = np.linalg.norm(np.array([errors["N1"], errors["N2"]]), axis=0)
    fig, (ax_D, ax_N) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    ax_D.plot(ratios, D_error_magnitude, marker="o")
    ax_D.set_title("D error magnitude")
    ax_N.plot(ratios, N_error_magnitude, marker="o")
    ax_N.set_title("N error magnitude")
    for ax_i in (ax_D, ax_N):
        ax_i.set_xlabel("D1 / D2")
        ax_i.set_ylabel("Error vector magnitude (percent)")
    fig.suptitle("Two-Species Total Fit Error vs Diffusion Coefficient Ratio")
    fig.tight_layout()
    plt.show()

    print(f"{'D1_true':>8} {'D1_fit':>10} {'D2_true':>8} {'D2_fit':>10} {'N1_true':>9} {'N1_fit':>10} {'N2_true':>9} {'N2_fit':>10}")
    for row in fit_rows:
        print(f"{row[0]:>8.1f} {row[1]:>10.2f} {row[2]:>8.1f} {row[3]:>10.2f} {row[4]:>9.4f} {row[5]:>10.4f} {row[6]:>9.4f} {row[7]:>10.4f}")

    



if __name__ == "__main__":
    run_validation()
    #two_species_validation()
    
