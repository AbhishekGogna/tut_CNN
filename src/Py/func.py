# Add custom functions here to be loaded for all analysis
from .libs import *

def inverse_scale(scaler, myyield_tanh, verbose):
    #undo scalling
    myyield_tanh_inv = scaler.inverse_transform(myyield_tanh.reshape(-1, 1))
    myyield_tanh_inv = myyield_tanh_inv.flatten()
    if verbose:
        print(scaler, myyield_tanh.shape, myyield_tanh_inv.shape, myyield_tanh_inv.max(), myyield_tanh_inv.min(), myyield_tanh_inv.mean())
    return myyield_tanh_inv

def read_pkl(path):
    with open(path, "rb") as fp:   # Unpickling
        data = pickle.load(fp)
    return data

def read_json(path):
    with open(path, encoding = "utf8") as json_file:
        data = json.load(json_file)
    return data
def write_json(data, path, verbose = False):
    with open(path, "w") as fp:   
        json.dump(data, fp)
    if verbose:
        return print("Done")
def get_random_string(length):
    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    # print random string
    return result_str

def set_dirs(base_dir_path, verbose = True, run_id = None):
    if run_id is None:
        run_id = time.strftime("run_%Y_%m_%d")
        base_folder = base_dir_path + '/' + run_id
    else:
        base_folder = base_dir_path + '/' + f'{str(run_id)}'
    cb_at = base_folder + '/callback_data'
    tb_cb = cb_at + '/tb_cb'
    mc_cb = cb_at + '/mc_cb/'
    pred_at = base_folder + '/pred'
    model_at = base_folder + '/model'
    tmp_at = base_folder + '/tmp_data'
    if(not os.path.isdir(base_folder)):
        os.system(f'mkdir -p  {base_folder} {pred_at} {model_at} {cb_at} {tb_cb} {mc_cb} {tmp_at}')
    if (verbose):
        print(f'base folder at {base_folder}, \ncallbacks at {cb_at}, \npredictions at {pred_at}, \nmodel at {model_at}, \ntmp at {tmp_at}')
    # output
    out = {}
    out['base_folder'] = base_folder
    out['tb_cb'] = tb_cb
    out['mc_cb'] = mc_cb
    out['pred_at'] = pred_at
    out['model_at'] = model_at
    out['tmp_at'] = tmp_at
    return out 

def create_train_val_data(index_train, index_test, index_val = None, prop = 0.1):
    if index_val is None:
        val_set = random.sample(index_train, int(len(index_train)*prop)) # cretes validation set from the remaining non_test set 
        train_set = list(set(index_train).difference(val_set))
    else:
        val_set = index_val
        train_set = index_train
    
    test_set = index_test
    check = any(item in val_set for item in train_set)
    
    if check:
        print("function failed since some elemets of val arer in the train set")
    else:
        return train_set, val_set, test_set
    
def fit_model(final_model, params, train_x, val_x, train_y, val_y):
     
    # set variables
    tb_filepath, cp_filepath, b_size, epoch, vbs, sfl = [params['fit'][key] for key in ['tensorboard_fp', 'checkpoint_fp', 'batch_size', 'epochs', 'verbose', 'shuffle']]
    
    #set call backs
    tensorboard_cb = TensorBoard(tb_filepath)
    modelcheck_cb = ModelCheckpoint(filepath=cp_filepath,
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True)
    model_cb = EarlyStopping(monitor='val_loss',
                                     min_delta=0.00001,
                                     patience=5,
                                     verbose=0,
                                     mode='min',
                                     baseline=None,
                                     restore_best_weights=True)
    final_model.fit(train_x, train_y, validation_data=(val_x, val_y),
                    batch_size = b_size,
                    epochs = epoch,
                    verbose = vbs,
                    shuffle = sfl,
                    callbacks=[modelcheck_cb, 
                               tensorboard_cb,
                               model_cb])
    
    final_model.load_weights(cp_filepath) # loads best weights
    return final_model

def predict_values (model, test_x, test_y, index, scaler):
    
    # perform predictions
    prediction = model.predict(test_x)
    
    # re-scale data
    obs = inverse_scale(scaler, test_y, verbose = False)
    pred = inverse_scale(scaler, prediction, verbose = False)
    out_data = pd.DataFrame([index, obs, pred], index=["index","obs","pred"]).T
    out_data["index"] = out_data["index"].astype('int')
    return out_data

def save_model(model, path, model_name = "kibreed_pred"):
    model_json = model.to_json()
    with open(path + '/' + model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path + '/' + model_name + ".h5")
    print("Saved model to disk")
    return