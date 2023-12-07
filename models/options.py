import os
import json

class Options():
    def get_default():
        opt = {}

        # For descriptions of all variables, see train.py
        opt['num_dims']                             = 2       
        opt['num_outputs']                          = 3
        opt['training_data']                        = "pluto.png"
        opt['training_data_type']                   = "image" 

        opt['num_starting_prims']                   = 10000      
        opt['num_total_prims']                      = 100000

        opt['max_frequency']                        = 128
        opt['num_frequencies']                      = 1
        opt['num_total_frequencies']                = 128
        opt['gaussian_only']                        = False

        opt['train_iterations']                     = 30000
        opt['batch_size']                           = 100000  
        opt['fine_tune_iterations']                 = 15000
        opt['split_every_iters']                    = 1000
        opt['prune_every_iters']                    = 100
        opt['blackout_every_iters']                 = 3000

        opt['device']                               = 'cuda:0'
        opt['data_device']                          = 'cuda:0'
 
        opt['lr']                                   = 0.01
        opt['beta_1']                               = 0.9
        opt['beta_2']                               = 0.99

        opt['iteration_number']                     = 0
        opt['log_every']                            = 100
        opt['log_image_every']                      = 1000
        opt['log_image']                            = True
        opt['profile']                              = False

        opt['save_name']                            = "test"

        return opt

def save_options(opt, save_location):
    with open(os.path.join(save_location, "options.json"), 'w') as fp:
        json.dump(opt, fp, sort_keys=True, indent=4)
    
def load_options(load_location):
    opt = Options.get_default()
    #print(load_location)
    if not os.path.exists(load_location):
        print("%s doesn't exist, load failed" % load_location)
        return
        
    if os.path.exists(os.path.join(load_location, "options.json")):
        with open(os.path.join(load_location, "options.json"), 'r') as fp:
            opt2 = json.load(fp)
    else:
        print("%s doesn't exist, load failed" % "options.json")
        return
    
    # For forward compatibility with new attributes in the options file
    for attr in opt2.keys():
        opt[attr] = opt2[attr]

    return opt

def update_options_from_args(opt, args):
    for k in args.keys():
        if args[k] is not None:
            opt[k] = args[k]
    return opt