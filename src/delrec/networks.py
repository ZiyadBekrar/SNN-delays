"""Network zoo for SSC, PS-MNIST, HAR, AL and Mackey-Glass.

A hidden block is Linear -> Dropout -> neuron -> spike_registrator -> [BatchNorm]. The network ends in a Linear plus a leaky-integrate layer with an infinite
threshold, so ``forward`` returns a membrane sequence rather than spikes.

The class hierarchy is the experimental matrix: ``SNN_recurrent_delays`` (axonal,
learned) is the base, and each sibling changes exactly one thing, the delay shape
(synaptic), whether the delays are trained (fixed), or whether they exist at all
(vanilla). ``spike_registrator`` is a pass-through that stashes spike trains for
the firing-rate penalty and the efficiency analyses.
"""

import torch
import math

from spikingjelly.activation_based import layer
from DCLS.construct.modules import Dcls1d

from delrec.delay_layers import axonal_recdel, synaptic_recdel, common_recdel

class dcls_module(Dcls1d):
    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        groups,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_count=config.kernel_count,
            groups=groups,
            dilated_kernel_size=config.max_feedforward_delay,
            bias=config.bias,
            version=config.DCLSversion,
            )
        
        self.config = config
        self.left_padding, self.right_padding = config.left_padding, config.right_padding

    def forward(self, x):
        assert x.dim() == 3 # (T, B, N)
        x = x.permute(1,2,0) # (batch, neurons, time)
        x = torch.nn.functional.pad(x, (self.left_padding, self.right_padding), 'constant', 0)
        # _conv_forward(self.P) is identical to super().forward().
        x = self._conv_forward(x, self.weight, self.bias, self.P, self.SIG)
        x = x.permute(2,0,1) # (time, batch, neurons)
        return x
        
class modified_batchnorm(layer.BatchNorm1d):
    def __init__(self, num_features, step_mode='m'):
        super().__init__(num_features, step_mode=step_mode)
        
    def forward(self, x):
        assert x.dim() == 3 # (T, B, N)
        return super().forward(x.unsqueeze(3)).squeeze(-1) # We apply batchnorm to shape (T, B, N, 1)

class spike_registrator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spikes = None

    def forward(self, x):
        assert x.dim() == 3 # (T, B, N)
        self.spikes = x.clone()
        return x

class SNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        assert config.dataset in ['SSC', 'PSMNIST', 'HAR', 'AL', 'MG'], "This SNN is designed for SSC, PSMNIST, HAR, AL or MG datasets."
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias)) # (T, B, N_in) -> (T, B, N_hidden)
            dim_buffer = layer_dim
            
            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))
            
            layers.append(config.neuron_module(
                tau = config.tau,
                decay_input = config.decay_input,
                v_reset = config.v_reset,
                v_threshold = config.v_threshold,
                surrogate_function = config.surrogate_function,
                detach_reset = config.detach_reset,
                step_mode = config.step_mode,
                backend = config.backend,
                store_v_seq = config.store_v_seq,
                )
                          )
                
            layers.append(spike_registrator())
            
            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))
            
        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        
        self.layers = torch.nn.Sequential(*layers)
        
        self.init_weights()
        
    def forward(self, x): 
        assert x.dim() == 3 # (T, B, N)
        x = self.layers(x)
        return x
        
    def log_params(self):
        logs = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Linear):
                
                w = torch.abs(layer.weight).mean()
                w_grad_max = layer.weight.grad.abs().max().item() if layer.weight.grad is not None else 0.0
                logs.update({
                        f'w_linear_{idx}': w,
                        f'w_linear_grad_max_{idx}': w_grad_max,
                    })
                
        return logs
        
    def init_weights(self):
        for m in self.layers:
            if isinstance(m, torch.nn.Linear):
                if self.config.init_ff_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # For big models
                elif self.config.init_ff_weights == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                elif self.config.init_ff_weights == 'default':
                    pass
    
class SNN_recurrent_delays(SNN):
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias)) # (T, B, N_in) -> (T, B, N_hidden)
            dim_buffer = layer_dim

            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))

            is_last_hidden = (idx == len(config.hidden_layers) - 1)
            if is_last_hidden and getattr(config, 'no_recurrence_in_last_layer', False):
                # plain (non-recurrent) LIF readout layer, matching neuroseqbench's HAR FFSNN
                layers.append(config.neuron_module(
                    tau = config.tau,
                    decay_input = config.decay_input,
                    v_reset = config.v_reset,
                    v_threshold = config.v_threshold,
                    surrogate_function = config.surrogate_function,
                    detach_reset = config.detach_reset,
                    step_mode = config.step_mode,
                    backend = config.backend,
                    store_v_seq = config.store_v_seq,
                    ))
            else:
                layers.append(axonal_recdel(config, layer_dim, config.neuron_module))

            layers.append(spike_registrator())

            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))

        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))

        self.layers = torch.nn.Sequential(*layers)

        self.init_weights()

    def clamp_delays(self):
        for m in self.layers:
            if isinstance(m, axonal_recdel):
                m.clamp_recurrent_delays()
                
    def round_pos(self):
        """Project the fractional recurrent delays onto the integer grid, in place."""
        stochastic = getattr(self.config, 'round_mode', 'nearest') == 'stochastic'
        with torch.no_grad():
            for m in self.layers:
                if isinstance(m, axonal_recdel):
                    d = m.recurrent_delays
                    if stochastic:
                        d.copy_(torch.floor(d + torch.rand_like(d)))
                    else:
                        d.round_()
                    m.clamp_recurrent_delays()

    def forward(self, x):
        return super().forward(x)
    
    def log_params(self):
        logs = super().log_params()
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, axonal_recdel):
                    logs[f'sigma_rec{idx}'] = layer.sigma
                    curr_pos_rec = layer.recurrent_delays.cpu().detach().numpy()
                    logs[f'pos_rec{idx}'] = curr_pos_rec.mean()
                    
                    
                    rec_w = layer.recurrent_weights
                    rec_w_mean = torch.abs(rec_w).mean()
                    rec_w_grad_max = rec_w.grad.abs().max().item() if rec_w.grad is not None else 0.0

                    logs.update({
                        f'recurrent_w_{idx}': rec_w_mean,
                        f'recurrent_w_grad_max_{idx}': rec_w_grad_max,
                    })
        
                    rec_d = layer.recurrent_delays
                    rec_d_grad_max = rec_d.grad.abs().max().item() if rec_d.grad is not None else 0.0

                    logs[f'recurrent_delay_grad_max_{idx}'] = rec_d_grad_max
                    
                    # if layer.use_sig_p:
                    #     logs[f"p_spread_mean_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().mean().item()
                    #     logs[f"p_spread_std_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().std().item()
                    
        return logs


class SNN_synaptic_recurrent_delays(SNN_recurrent_delays):
    """Identical to SNN_recurrent_delays but with per-synapse delays (synaptic_recdel,
    recurrent_delays shape (N, N)) instead of per-neuron axonal delays.
    """

    def __init__(self, config):
        # Skip SNN_recurrent_delays.__init__ (it hardcodes axonal_recdel). Rebuild
        # the same layer stack with synaptic_recdel at the one recurrent site.
        SNN.__init__(self, config)

        self.config = config

        layers = []
        dim_buffer = config.input_size

        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            dim_buffer = layer_dim

            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))

            is_last_hidden = (idx == len(config.hidden_layers) - 1)
            if is_last_hidden and getattr(config, 'no_recurrence_in_last_layer', False):
                layers.append(config.neuron_module(
                    tau = config.tau,
                    decay_input = config.decay_input,
                    v_reset = config.v_reset,
                    v_threshold = config.v_threshold,
                    surrogate_function = config.surrogate_function,
                    detach_reset = config.detach_reset,
                    step_mode = config.step_mode,
                    backend = config.backend,
                    store_v_seq = config.store_v_seq,
                    ))
            else:
                layers.append(synaptic_recdel(config, layer_dim, config.neuron_module))

            layers.append(spike_registrator())

            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))

        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))

        self.layers = torch.nn.Sequential(*layers)

        self.init_weights()


class SNN_common_recurrent_delays(SNN_recurrent_delays):
    """Identical to SNN_recurrent_delays but each recurrent layer learns a single shared delay
    (common_recdel, recurrent_delays shape (1,)) instead of per-neuron axonal delays.
    """

    def __init__(self, config):
        # Skip SNN_recurrent_delays.__init__ (it hardcodes axonal_recdel). Rebuild
        # the same layer stack with common_recdel at the one recurrent site.
        SNN.__init__(self, config)

        self.config = config

        layers = []
        dim_buffer = config.input_size

        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            dim_buffer = layer_dim

            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))

            is_last_hidden = (idx == len(config.hidden_layers) - 1)
            if is_last_hidden and getattr(config, 'no_recurrence_in_last_layer', False):
                layers.append(config.neuron_module(
                    tau = config.tau,
                    decay_input = config.decay_input,
                    v_reset = config.v_reset,
                    v_threshold = config.v_threshold,
                    surrogate_function = config.surrogate_function,
                    detach_reset = config.detach_reset,
                    step_mode = config.step_mode,
                    backend = config.backend,
                    store_v_seq = config.store_v_seq,
                    ))
            else:
                layers.append(common_recdel(config, layer_dim, config.neuron_module))

            layers.append(spike_registrator())

            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))

        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))

        self.layers = torch.nn.Sequential(*layers)

        self.init_weights()


class SNN_vanilla_recurrent(SNN_recurrent_delays):
    def __init__(self, config):
        super().__init__(config)

        for layer in self.layers:
            if isinstance(layer, axonal_recdel):
                with torch.no_grad():
                    layer.recurrent_delays.fill_(0.)  
                
                layer.recurrent_delays.requires_grad = False

                layer.sigma = 0.
                layer.config.sigma_init = 0.

                # Fixed delays don't learn a spread. Block p_spread even if
                # use_sig_p is set in the config.
                layer.use_sig_p = False
                if hasattr(layer, 'p_spread'):
                    del layer.p_spread

    def forward(self, x):
        return super().forward(x)

class SNN_fixed_recurrent_delays(SNN_recurrent_delays):
    def __init__(self, config):
        super().__init__(config)
        
        for layer in self.layers:
            if isinstance(layer, axonal_recdel):
                # Fixed delays are frozen at their init values. Round them to
                # integers so they match the round_pos eval regime with no epoch-0 jump.
                with torch.no_grad():
                    layer.recurrent_delays.round_()
                    layer.clamp_recurrent_delays()
                layer.recurrent_delays.requires_grad = False
                layer.sigma = 0.
                layer.config.sigma_init = 0.

                # Fixed delays don't learn a spread. Block p_spread even if
                # use_sig_p is set in the config.
                layer.use_sig_p = False
                if hasattr(layer, 'p_spread'):
                    del layer.p_spread

    def forward(self, x):
        return super().forward(x)

class SNN_fixed_synaptic_recurrent_delays(SNN_synaptic_recurrent_delays):
    """Per-synapse analogue of SNN_fixed_recurrent_delays: builds the synaptic
    delay stack, then freezes the (N, N) recurrent_delays at their rounded init
    values with sigma=0 (no learning, no spread)."""
    def __init__(self, config):
        super().__init__(config)

        for layer in self.layers:
            if isinstance(layer, axonal_recdel):   # covers synaptic_recdel
                with torch.no_grad():
                    layer.recurrent_delays.round_()
                    layer.clamp_recurrent_delays()
                layer.recurrent_delays.requires_grad = False
                layer.sigma = 0.
                layer.config.sigma_init = 0.

                layer.use_sig_p = False
                if hasattr(layer, 'p_spread'):
                    del layer.p_spread

    def forward(self, x):
        return super().forward(x)

class SNN_feedforward_delays(SNN):
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        for idx, layer_dim in enumerate(config.hidden_layers):
            
            if config.no_delay_in_first_layer and idx == 0:
                layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            else:
                layers.append(
                    dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = layer_dim,
                        groups = 1,
                    )
                    ) # (T, B, N_in) -> (T, B, N_hidden)
            dim_buffer = layer_dim
            
            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))
                
            layers.append(config.neuron_module(
            tau = config.tau,
            decay_input = config.decay_input,
            v_reset = config.v_reset,
            v_threshold = config.v_threshold,
            surrogate_function = config.surrogate_function,
            detach_reset = config.detach_reset,
            step_mode = config.step_mode,
            backend = config.backend,
            store_v_seq = config.store_v_seq,
            )
                            )
            
            layers.append(spike_registrator())
            
            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))
                
        if config.no_delay_in_last_layer:
            layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        else:
                layers.append(dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = config.output_size,
                        groups = 1,
                    ))
            
        self.layers = torch.nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.layers:
            # Feedforward weights init
            if isinstance(m, torch.nn.Linear):
                if self.config.init_ff_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # For big models
                elif self.config.init_ff_weights == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                elif self.config.init_ff_weights == 'default':
                    pass
                
            if isinstance(m, dcls_module):
                if self.config.init_dcls_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # For big models
                elif self.config.init_dcls_weights == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                elif self.config.init_dcls_weights == 'default':
                    torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # same as nn.Linear default
                
            # Feedforward delays init
            if isinstance(m, dcls_module):
                torch.nn.init.uniform_(m.P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                m.clamp_parameters()
                
    def clamp_delays(self, train=True):
        for m in self.layers:
            if isinstance(m, dcls_module):
                m.clamp_parameters()
                    
    def round_pos(self):
        with torch.no_grad():
            for m in self.layers:
                if isinstance(m, dcls_module):
                    m.P.round_()
                    m.clamp_parameters()
                    
    def log_params(self):
        logs = super().log_params()
        
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, dcls_module):
                
                curr_pos_ff = layer.P.cpu().detach().numpy()
                logs[f'pos_feedforward{idx}'] = curr_pos_ff.mean()
                
                
                ff_w = layer.weight
                ff_w_mean = torch.abs(ff_w).mean()
                ff_w_grad_max = ff_w.grad.abs().max().item() if ff_w.grad is not None else 0.0
                ff_d_grad_max = layer.P.grad.abs().max().item() if layer.P.grad is not None else 0.0

                logs.update({
                    f'feedforward_dcls_w_{idx}': ff_w_mean,
                    f'feedforward_dcls_w_grad_max_{idx}': ff_w_grad_max,
                    f'feedforward_dcls_delay_grad_max_{idx}': ff_d_grad_max,
                })
                
        return logs

class SNN_axonal_feedforward_delays(SNN_feedforward_delays):
    def __init__(self, config):
        super().__init__(config)  

        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        for idx, layer_dim in enumerate(config.hidden_layers):
            
            print(idx, layer_dim, dim_buffer)
            
            if config.no_delay_in_first_layer and idx == 0:
                layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            else:
                layers.append(
                    dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = dim_buffer,
                        groups = dim_buffer,
                    )
                    ) # (T, B, N_in) -> (T, B, N_hidden)
                layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias)) 
            dim_buffer = layer_dim
            
            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))
                
            layers.append(config.neuron_module(
            tau = config.tau,
            decay_input = config.decay_input,
            v_reset = config.v_reset,
            v_threshold = config.v_threshold,
            surrogate_function = config.surrogate_function,
            detach_reset = config.detach_reset,
            step_mode = config.step_mode,
            backend = config.backend,
            store_v_seq = config.store_v_seq,
            )
                            )
            
            layers.append(spike_registrator())
                
            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))
            
        if config.no_delay_in_last_layer:
            layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        else:
                layers.append(dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = dim_buffer,
                        groups = dim_buffer,
                    ))
                layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias)) 
            
        self.layers = torch.nn.Sequential(*layers)
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.layers:
            # Feedforward weights init
            if isinstance(m, torch.nn.Linear):
                if self.config.init_ff_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # For big models
                elif self.config.init_ff_weights == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                elif self.config.init_ff_weights == 'default':
                    pass
                
            if isinstance(m, dcls_module):
                torch.nn.init.constant_(m.weight, 1.0)
                m.weight.requires_grad = False
                        
            # Feedforward delays init
            if isinstance(m, dcls_module):
                torch.nn.init.uniform_(m.P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                m.clamp_parameters()
                
                if self.config.DCLSversion == 'gauss':
                    torch.nn.init.constant_(m.SIG, self.config.siginit)
                    m.SIG.requires_grad = False
                    
    
    def log_params(self):
        logs = super().log_params()
        
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, dcls_module):
                
                curr_pos_ff = layer.P.cpu().detach().numpy()
                logs[f'pos_feedforward{idx}'] = curr_pos_ff.mean()
                
                
                ff_w = layer.weight
                ff_w_mean = torch.abs(ff_w).mean()
                ff_w_grad_max = ff_w.grad.abs().max().item() if ff_w.grad is not None else 0.0
                ff_d_grad_max = layer.P.grad.abs().max().item() if layer.P.grad is not None else 0.0

                logs.update({
                    f'feedforward_w_{idx}': ff_w_mean,
                    f'feedforward_w_grad_max_{idx}': ff_w_grad_max,
                    f'feedforward_delay_grad_max_{idx}': ff_d_grad_max,
                })
                
                # Log sigma values for each DCLS layer
                if self.config.DCLSversion == 'gauss':
                    sigma_val = layer.SIG.item() if layer.SIG.numel() == 1 else layer.SIG.mean().item()
                    logs.update({
                        f'dcls/sigma_layer_{idx}': sigma_val,
                    })
                
        return logs

class SNN_recurrent_and_feedforward_delays(SNN_feedforward_delays, SNN_recurrent_delays):

    def __init__(self, config):
        super(SNN_feedforward_delays, self).__init__(config)  

        self.config = config

        layers = []
        dim_buffer = config.input_size

        for idx, layer_dim in enumerate(config.hidden_layers):
            if config.no_delay_in_first_layer and idx == 0:
                layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            else:
                layers.append(
                    dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = layer_dim,
                        groups = 1,
                    )
                    ) # (T, B, N_in) -> (T, B, N_hidden)
            dim_buffer = layer_dim
            
            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))

            layers.append(axonal_recdel(config, layer_dim, config.neuron_module))
                
            layers.append(spike_registrator())

            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))

        if config.no_delay_in_last_layer:
            layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        else:
                layers.append(dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = config.output_size,
                        groups = 1,
                    ))

        self.layers = torch.nn.Sequential(*layers)

        SNN_feedforward_delays.init_weights(self)

    def forward(self, x):
        return SNN.forward(self, x)
    
    def clamp_delays(self):
        SNN_feedforward_delays.clamp_delays(self)
        SNN_recurrent_delays.clamp_delays(self)
        
    def round_pos(self):
        SNN_feedforward_delays.round_pos(self)
        SNN_recurrent_delays.round_pos(self)
        
    def log_params(self):
        logs = SNN_feedforward_delays.log_params(self)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, axonal_recdel):
                    logs[f'sigma_rec{idx}'] = layer.sigma
                    curr_pos_rec = layer.recurrent_delays.cpu().detach().numpy()
                    logs[f'pos_rec{idx}'] = curr_pos_rec.mean()
                    
                    
                    rec_w = layer.recurrent_weights
                    rec_w_mean = torch.abs(rec_w).mean()
                    rec_w_grad_max = rec_w.grad.abs().max().item() if rec_w.grad is not None else 0.0

                    logs.update({
                        f'recurrent_w_{idx}': rec_w_mean,
                        f'recurrent_w_grad_max_{idx}': rec_w_grad_max,
                    })
        
                    rec_d = layer.recurrent_delays
                    rec_d_grad_max = rec_d.grad.abs().max().item() if rec_d.grad is not None else 0.0

                    logs[f'recurrent_delay_grad_max_{idx}'] = rec_d_grad_max
                    
                    if layer.use_sig_p:
                        logs[f"p_spread_mean_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().mean().item()
                        logs[f"p_spread_std_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().std().item()
                    
        return logs
        