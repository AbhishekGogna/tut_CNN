# Add custom functions here to be loaded for all analysis
from .libs import *

def scale_data(to_transform, pd_cols, pd_index):
    scaler = MinMaxScaler((0,1))
    data_scaled = scaler.fit_transform(to_transform)
    data_scaled_df = pd.DataFrame(data_scaled, columns = pd_cols, index = pd_index)
    return data_scaled_df, scaler

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

def write_pkl(data, path, verbose = False):
    with open(path, "wb") as fp:   # pickling
        pickle.dump(data, fp)
    if verbose:
        return print("Done")
    
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

## Additional for ravi
def fit_regression_model(X_log, y):
    """Fits a linear regression model on log-transformed X and returns the intercept and coefficient."""
    reg = LinearRegression()
    reg.fit(X_log, y)
    return reg.intercept_, reg.coef_[0]

def generate_smooth_line(a, b, X_log):
    """Generates the smooth line y-values using the linear regression parameters."""
    return (a + b * X_log).flatten()

def plot_relationship(ax, X, y, 
                      #y_vals_log, 
                      max_y, 
                      max_y_x, 
                      title, ylabel):
    """Plots the scatter plot, regression line, max intercept lines, and labels for a given axis."""
    ax.plot(X, y, 'o', color='blue')
    #ax.plot(X, y_vals_log, color='red', label=f'Line: y ~ a + b * log(x)')
    ax.axvline(x=max_y_x, color='green', linestyle='--', label=f'X intercept at {max_y_x:.2f}')
    ax.axhline(y=max_y, color='purple', linestyle='--', label=f'Y intercept at {max_y:.2f}')
    ax.set_ylim(0, 1)  # Fix y-axis range from 0 to 1
    ax.set_xlabel('train_size')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
# Function to extract log times from a single log file (with conversion to hours and Run as first column)
def extract_log_times(log_file_path):
    # Extract the run identifier from the file path
    run_id = re.search(r'run_\d+', log_file_path).group(0)  # Extract run_xx (e.g., run_0)

    with open(log_file_path, 'r') as f:
        logs = f.readlines()

    # Patterns for extracting the relevant times
    tuning_time_pattern = re.compile(r"INFO:root:HP tuning took (\d+\.\d+) seconds")
    fitting_time_pattern = re.compile(r"INFO:root:Model fitting took (\d+\.\d+) seconds")

    # Extracting the times using regex
    tuning_time_seconds = next((float(tuning_time_pattern.search(line).group(1)) for line in logs if tuning_time_pattern.search(line)), None)
    fitting_time_seconds = next((float(fitting_time_pattern.search(line).group(1)) for line in logs if fitting_time_pattern.search(line)), None)

    # Convert times from seconds to hours
    times = {
        'Run': run_id,
        'Tuning Complete (hours)': tuning_time_seconds / 3600 if tuning_time_seconds is not None else None,
        'Fitting Complete (hours)': fitting_time_seconds / 3600 if fitting_time_seconds is not None else None
    }
    
    return times

# Compact function to create DataFrame from multiple log files
def log_times_to_dataframe(log_file_paths):
    return pd.DataFrame([extract_log_times(f) for f in log_file_paths])

# Function to train a linear regression model on train_size and Tuning Time
def train_tuning_model(df):
    df_clean = df.dropna()
    # Extract train_size and Tuning Time (in hours)
    X = df_clean[['train_size']].values  # train_size as feature
    y = df_clean['Tuning Complete (hours)'].values  # Tuning Time as target
    
    # Initialize and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Function to predict Tuning Time based on train_size
def predict_tuning_time(model, train_length):
    return model.predict(np.array([[train_length]]))[0]