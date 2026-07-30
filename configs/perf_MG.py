"""Hyperparameters for Mackey-Glass forecasting (see the paper's Methods).

One hidden layer of 128 recurrent LIF over windows of 150 steps, predicting a
scalar `prediction_horizon` steps ahead. `mg_tau` sets the chaos of the generated
series. Both are swept by experiments/sweeps/sweep_MG.py. The series is
integrated in memory from these parameters. There is no dataset to download.
"""

from spikingjelly.activation_based import neuron, surrogate

class Config():

    ### Dataset ###

    dataset = 'MG'
    seed = 0

    # Mackey-Glass dynamical system: dx/dt = beta*x(t-tau)/(1+x(t-tau)^n), gamma*x(t).
    # Integrated with Euler at step mg_dt, then sampled every 1.0 time unit. The first
    # mg_discard time units (transient) are dropped and mg_total samples are kept.
    # mg_tau is the delay of the system itself, the axis this sweep varies.
    mg_beta    = 0.2
    mg_gamma   = 0.1
    mg_n       = 10
    mg_tau     = 30
    mg_dt      = 0.1
    mg_x0      = 1.2
    mg_discard = 5000
    mg_total   = 6000

    # Task: from a window of window_size past samples, predict the value
    # prediction_horizon steps after the end of the window.
    window_size        = 150 #100 #200 
    time_window        = window_size   # T (== window_size)
    prediction_horizon = 6

    # Contiguous splits of the series (the rest is the test split).
    train_frac = 0.60
    val_frac   = 0.20

    # The series is generated in memory and windowed into a TensorDataset, so
    # dataloader workers would only add fork/IPC overhead, and forking from a
    # Triton/CUDA process is what triggers ENOMEM under a cgroup memory limit.
    num_workers = 0

    ### General ###

    epochs = 100
    batch_size = 512

    bias = False
    use_batch_norm = False

    results_dir = ''

    ### Model architechture ###

    hidden_layers = [128]

    input_size = 1     # scalar time series
    output_size = 1    # scalar regression target

    # Leaky-integrator readout: the (T, B, 1) output sequence is reduced to one
    # prediction per sample by an EMA with decay (1 minus 1/tau_readout), so the
    # prediction is dominated by the last ~tau_readout steps of the window
    # (see readout_MG in src/utils.py). MG's analogue of calc_loss_*'s output.mean(0).
    tau_readout = 20.0

    recurrent_dropout_rate = 0.2   #0.2
    feedforward_dropout_rate = 0.2 #0.2

    init_ff_weights = 'kaiming'   # 'default', 'kaiming' or 'normal'

    no_recurrence_in_last_layer = False

    ### Spiking neuron configuration ###

    neuron_module = neuron.LIFNode
    backend = 'torch'

    tau = 2.0
    v_threshold = 1.0
    v_reset = 0.0  # hard reset to rest=0 (None would be soft reset)

    surrogate_function = surrogate.ATan(alpha=5.0)
    detach_reset = True
    decay_input = False

    step_mode = 'm'
    store_v_seq = True

    ### Recurrent delays ###

    use_sig_p = False

    init_rec_delay = 'uniform'  # 'half_normal' or 'uniform'
    max_rec_delay = 20          # upper bound of the uniform delay init
    init_recdel_offset = 0      # lower bound of the uniform delay init
    delay_std_init = 12         # only read by the 'half_normal' init
    rec_delay_init_gain = 1.0

    init_rec_weights = 'orthogonal'  # DelRec default (SSC/SHD)
    use_rec_bias = False

    sigma_init = 10.0
    sigma_decay = 0.95 #0.8 #0.95

    # Round the (fractional) delays to integers before each eval epoch
    round_pos_each_epoch = True


    ### Optimization ###

    optim = 'adam'

    scheduler_weights = 'cos'  # 'cos' or 'onecycle'
    scheduler_pos = 'cos'

    lr_w = 5e-4
    lr_positions = 5e-2

    weight_decay = 1e-4
