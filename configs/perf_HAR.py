"""Hyperparameters for HAR / WISDM (the hyperparameter tables of the paper).

Matched to the ~100k-parameter budget of [30]: hidden layers [128, 176, 176] over
T = 200, the last of which is a plain LIF (`no_recurrence_in_last_layer`), so only
two layers carry delays. `delay_std_init` is the variable swept in Section 2.4.
"""

from spikingjelly.activation_based import neuron, surrogate
from delrec.utils import Triangle

SURROGATE_GAMMA = 0.6  # triangle surrogate width, matching neuroseqbench's HAR alpha=0.6

def triangle_surrogate(x):
    # Triangle surrogate, matching neuroseqbench's HAR alpha
    return Triangle.apply(x, SURROGATE_GAMMA)

class Config():

    ### Dataset ###

    dataset = 'HAR'
    datasets_path = 'Datasets/HAR'
    seed = 1234

    # WISDM windowing (matching neuroseqbench: 200-step windows, 50% overlap)
    window_size = 200
    overlap = 100
    time_window = 200   # T (number of timesteps per window)
    # HAR data is a fully in-memory TensorDataset (cached .npy), so dataloader
    # workers add only fork/IPC overhead. Keep at 0, and crucially, forking
    # workers from the Triton process (large CUDA/JIT VM) triggers os.fork()
    # ENOMEM under the SLURM cgroup memory limit.
    num_workers = 0

    ### General ###

    epochs = 100
    batch_size = 256

    bias = True
    use_batch_norm = False

    results_dir = ''

    ### Model architechture ###

    hidden_layers = [128, 176, 176]

    input_size = 3     # tri-axial gyro (X, Y, Z)
    output_size = 18   # WISDM activities

    recurrent_dropout_rate = 0.2 #0. #0.2
    feedforward_dropout_rate = 0.

    init_ff_weights = 'default' # 'default' == PyTorch nn.Linear init == neuroseqbench feedforward init
    init_dcls_weights = 'default' # 'kaiming' or 'normal' or 'default'

    no_delay_in_first_layer = False
    no_delay_in_last_layer = False

    # neuroseqbench HAR makes the last hidden layer a plain (non-recurrent) LIF:
    # only the first len(hidden_layers)-1 layers carry recurrent connections/delays.
    no_recurrence_in_last_layer = True

    ### Spiking neuron configuration ###

    neuron_module = neuron.LIFNode
    backend = 'torch'

    # Matched to neuroseqbench Recurrent_LIF: v = decay*v + x, hard reset to rest=0.
    # spikingjelly LIFNode with decay_input=False, v_reset=0 gives v = (1-1/tau)*v + x,
    # so leak (1-1/tau)=0.1 == neuroseqbench decay=0.1  ->  tau = 1/0.9.
    tau = 1/0.9
    v_threshold = 0.5
    v_reset = 0.0  # hard reset to rest=0 (None would be soft reset)

    surrogate_function = staticmethod(triangle_surrogate)  # triangle, gamma=0.6
    surrogate_gamma = SURROGATE_GAMMA  # consumed by the Triton backward (multi_step_forward_triton)
    detach_reset = False
    decay_input=False

    step_mode = 'm'
    store_v_seq = True

    ### Recurrent delays ###

    use_sig_p = False

    delay_std_init = 3 #7 #5 #12
    init_rec_delay = 'half_normal' # 'half_normal' or 'uniform'
    rec_delay_init_gain = 1.0

    # Recurrent weight init: 'kaiming_uniform' matches neuroseqbench's nn.Linear(N,N)
    # ('orthogonal' is the DelRec default used for SSC/SHD).
    init_rec_weights = 'kaiming_uniform'

    # Bias on the recurrent connections (not present on SSC yet)
    use_rec_bias = True

    sigma_init = 0. #2. #5. #0. #10.0
    sigma_decay = 0.95 #0.98 #0.95

    # Round the (fractional) delays to integers before each eval epoch
    round_pos_each_epoch = True

    # How round_pos() projects the delays onto the integer grid:
    #   'nearest'    -> d.round()                (deterministic, the default)
    #   'stochastic' -> floor(d + U[0,1)), i.e. Round up with probability frac(d).
    #                   Unbiased (E[SR(d)] = d), so a sub-half-step within-epoch
    #                   drift is not erased by nearest-rounding's deadzone.
    round_mode = 'nearest'

    # Straight-through estimator on the delay discretization: the recurrent forward
    # taps at round(d), the gradient flows straight through to the fractional
    # recurrent_delays. Use with round_pos_each_epoch=False (the in-place per-epoch
    # round would destroy the fractional master copy this exists to maintain).
    round_delays = False

    # STE forward (only used when forward_version='triton_ste'):
    # True  -> single rounded tap (one-hot at round(1+d))
    # False -> 2-tap linear interpolation of the fractional delay
    ste_round_forward = False

    ### Feedforward delays ###

    DCLSversion = 'gauss' # 'gauss' not implemented yet

    kernel_count = 1
    max_feedforward_delay = 30
    max_feedforward_delay = max_feedforward_delay if max_feedforward_delay%2==1 else max_feedforward_delay+1

    left_padding = max_feedforward_delay - 1
    right_padding = 0

    init_pos_a = -max_feedforward_delay//2
    init_pos_b = max_feedforward_delay//2

    siginit = max_feedforward_delay//2

    ### Optimization ###

    # Matched to neuroseqbench HAR recurrent-LIF run:
    #   --lr 1.5e-3 --grad-clip 0. (no clip) . "adam" -> AdamW . Weight_decay=0 . StepLR(10, 0.8).
    # lr_positions is the DelRec delay-LR (no analogue in neuroseqbench. Unused by SNN_vanilla_recurrent).
    optim = 'adamW'

    scheduler_weights = 'step' # 'step' (StepLR), 'cos' or 'onecycle'
    scheduler_pos = 'step'
    step_size = 10
    step_gamma = 0.8

    lr_w = 1.5e-3
    lr_positions = 5e-2 #1e-1 #1e-2 #5e-2

    weight_decay = 0.0
    # L2 (decoupled) weight decay applied to the delay parameters only (the
    # "positions" optimizer group: recurrent_delays). 0.0 == original behavior.
    # >0 shrinks delays toward 0. Swept by sweep_HAR_weightdecay.py.
    weight_decay_positions = 0.0

    # L2 firing-rate penalty applied to every layer's spikes (via spike_registrator).
    # Added as spike_penalty * get_spike_cost(model). 0.0 == no penalty. Swept by
    # sweep_HAR_spikepenalty.py.
    spike_penalty = 0.0
