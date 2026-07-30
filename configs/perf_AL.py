"""Hyperparameters for Autonomous Localization (the hyperparameter tables of the paper).

Matched to the ~100k budget of [30]: hidden layers [128, 176, 176] over T = 400,
last layer non-recurrent. Delays are drawn uniformly on [0, 20] and, unlike the
other benchmarks, are not projected onto the integer grid each epoch
(`round_pos_each_epoch = False`), see the discretization appendix.
"""

from spikingjelly.activation_based import neuron, surrogate
from delrec.utils import Triangle

SURROGATE_GAMMA = 0.4  # triangle surrogate width, matching neuroseqbench's AL alpha=0.4

def triangle_surrogate(x):
    # Triangle surrogate, matching neuroseqbench's AL alpha
    return Triangle.apply(x, SURROGATE_GAMMA)

class Config():

    ### Dataset ###

    dataset = 'AL'
    # The AL dataset class creates an 'AL/' subdir under this root and caches its generated
    # HDF5 there (matching neuroseqbench's data_path semantics), so the data lands in 'Datasets/AL/'.
    datasets_path = 'Datasets'
    seed = 1234

    # AL (autonomous localization) is fully synthetic and generated on first run, then cached.
    capacity = 20            # expected number of turns, setting turn probability = capacity/seq_length
    al_train_num = 50000     # neuroseqbench AL train num_data
    al_test_num = 5000       # neuroseqbench AL test num_data
    time_window = 400        # T (sequence length), == neuroseqbench --time-window 400
    # In-memory tensors (cached HDF5). Keep workers at 0 (see HAR config note on
    # os.fork() ENOMEM when forking from the Triton/CUDA process).
    num_workers = 0

    ### General ###

    epochs = 200 #100
    batch_size = 256

    bias = True
    use_batch_norm = False

    results_dir = ''

    ### Model architechture ###

    hidden_layers = [128, 176, 176]

    input_size = 3     # one-hot action (move / turn-left / turn-right, no-op channel dropped)
    output_size = 2    # binary: final x-coordinate clamped to {0, 1}

    recurrent_dropout_rate = 0.   # strict neuroseqbench baseline has no recurrent dropout
    feedforward_dropout_rate = 0.

    init_ff_weights = 'default' # 'default' == PyTorch nn.Linear init == neuroseqbench feedforward init
    init_dcls_weights = 'default' # 'kaiming' or 'normal' or 'default'

    no_delay_in_first_layer = False
    no_delay_in_last_layer = False

    # neuroseqbench FFSNN makes the last hidden layer a plain (non-recurrent) LIF:
    # only the first len(hidden_layers)-1 layers carry recurrent connections/delays.
    no_recurrence_in_last_layer = True

    ### Spiking neuron configuration ###

    neuron_module = neuron.LIFNode
    backend = 'torch'

    # Matched to neuroseqbench Recurrent_LIF: v = decay*v + x, hard reset to rest=0.
    # spikingjelly LIFNode with decay_input=False, v_reset=0 gives v = (1-1/tau)*v + x,
    # so leak (1-1/tau)=decay=0.5  ->  tau = 1/(1-0.5) = 2.0.
    tau = 2.0
    v_threshold = 0.5
    v_reset = 0.0  # hard reset to rest=0 (None would be soft reset)

    surrogate_function = staticmethod(triangle_surrogate)  # triangle, gamma=0.4
    surrogate_gamma = SURROGATE_GAMMA  # consumed by the Triton backward (multi_step_forward_triton)
    detach_reset = False
    decay_input=False

    step_mode = 'm'
    store_v_seq = True

    ### Recurrent delays ###

    use_sig_p = False

    init_rec_delay = 'uniform' # 'half_normal' or 'uniform'
    delay_std_init = 25 #50 #20 #7 # if half_normal
    rec_delay_init_gain = 1.0 
    max_rec_delay = 20 #50. # uniform
    init_recdel_offset = 0.     # uniform

    # Recurrent weight init: 'kaiming_uniform' matches neuroseqbench's nn.Linear(N,N)
    # ('orthogonal' is the DelRec default used for SSC/SHD).
    init_rec_weights = 'kaiming_uniform'

    # Bias on the recurrent connections, matching nn.Linear(N, N) in Recurrent_LIF.
    use_rec_bias = True

    sigma_init = 0. #0.5 #0. #5. #10. #0.
    sigma_decay = 0.95


    # Round the (fractional) delays to integers before each eval epoch
    round_pos_each_epoch = False #True

    # STE forward (only used when forward_version='triton_ste'):
    # True  -> single rounded tap (one-hot at round(1+d))
    # False -> 2-tap linear interpolation of the fractional delay
    ste_round_forward = False

    ### Feedforward delays ###

    DCLSversion = 'gauss' # only used by the feedforward-delay model variants

    kernel_count = 1
    max_feedforward_delay = 30
    max_feedforward_delay = max_feedforward_delay if max_feedforward_delay%2==1 else max_feedforward_delay+1

    left_padding = max_feedforward_delay - 1
    right_padding = 0

    init_pos_a = -max_feedforward_delay//2
    init_pos_b = max_feedforward_delay//2

    siginit = max_feedforward_delay//2

    ### Optimization ###

    # Matched to neuroseqbench AL recurrent-LIF run:
    #   --lr 1e-3, "adam" -> AdamW, weight_decay=0, StepLR(step_size=10, gamma=0.8),
    #   --grad-clip 1.0 (clip_grad_norm_ over all params), epochs=100, batch=256, T=400.
    # lr_positions is the DelRec delay-LR (no analogue in neuroseqbench, unused by SNN_vanilla_recurrent).
    optim = 'adamW'

    scheduler_weights = 'cos' # 'step' (StepLR), 'cos' or 'onecycle'
    scheduler_pos = 'cos' # 'step'
    step_size = 10
    step_gamma = 0.8

    lr_w = 5e-4 #1e-3
    lr_positions = 5e-3 #5e-2 #5e-3 #5e-3 #5e-2 #1e-2 #5e-2

    weight_decay = 0.0

    # Decoupled weight decay applied ONLY to the recurrent axonal/synaptic delay
    # parameters (recurrent_delays), via the positions optimizer. Pulls delays
    # toward 0. Does not touch p_spread or feedforward (DCLS) delays. 0.0 == off.
    delay_weight_decay = 0.0 # 0.01

    # neuroseqbench applies clip_grad_norm_(model.parameters(), grad_clip) with grad_clip=1.0.
    grad_clip = 1.0
