#!/proj/py_env/bin/python3
# sample command line input - pred_script.py 1 /proj/tmp/
# load functions -------------------------------------------------------------------------------------------
import sys
functions_at = '/proj/ext_dir/src'
sys.path.append(functions_at)
from Py.libs import *
from Py.func import *

# Define variables -------------------------------------------------------------------------------------------
## From command line
all_args = sys.argv[1:]
model_name = all_args[0]
model = importlib.import_module(f'Py.model_{model_name}')
cv_schema = all_args[1]
key = all_args[2]
model_inputs = read_json("/proj/model_args.json") # a josn formatted as string for model inputs if any
tune = model_inputs[f'{model_name}']["tune"]
#hparams_at = '/proj/ext_dir/results/M2/hp_tuning/best_params.pkl' #todo: add this if you want to add one hyparameter for all runs

## Constant variables
base_dir = '/proj' # where to save predictions
inputs_at = '/proj/ext_dir/inputs'
tmp_at = '/proj/tmp_data'
os.system(f'export TMPDIR={tmp_at}')

## Run specifc
tb_cb = f'{base_dir}/callback_data/tb_cb'
mc_cb = f'{base_dir}/callback_data/mc_cb/model.ckpt'
tb_cb_tuning = f'{base_dir}/callback_data/tb_cb_tuning'
tuning_save_at = f'{base_dir}'
tune_dir = f'{model_name}/hp_tuning'
tuned_model_at = f'{base_dir}/model/model_tuned'
model_save_at = f'{base_dir}/model/model_fitted'
param_save_at = f'{base_dir}/model/best_params.json'
pred_save_at = f'{base_dir}/pred'
logs_at = f'{base_dir}/predictions.log'
path_to_pred_file = f'{pred_save_at}/output.csv'
cv_data = read_json(f'{inputs_at}/{cv_schema}.json')

# define logger -----------------------------------------------------------------------------------------------------
logging.basicConfig(filename=logs_at, level=logging.DEBUG, filemode='w')
logger = logging.getLogger(__name__)
logging.info(f'First attempt at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

# check for gpu -----------------------------------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    #tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    logging.info(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU')
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    logging.info(e)

# conditional execution -------------------------------------------------------------------------------------------------
if not exists(path_to_pred_file):

    # read in data ------------------------------------------------------------------------------------------------------
    if "acr" in model_name:
        ## g_a data
        g_data = np.load(f'{inputs_at}/acr_g.npy')
        scaler_g_a = read_pkl(f'{inputs_at}/acr_g.scl')
        
        ## p_data
        p_data = pd.read_csv(f'{inputs_at}/acr_p.csv')
        scaler_p = read_pkl(f'{inputs_at}/acr_p.scl')
        
    ## add further information
    p_data["run_idx"] = key
    p_data["model_idx"] = model_name
    p_data["run_type"] = re.sub(r"(\S+)\.json", r"\1", cv_schema)
    
    out_cols = ['trait', 'genotype', 'blups_raw', 'std.error', 'dataset', 'type', 
                'run_idx', 'model_idx', 'run_type', 'blups_scaled'] # these are the columns that must be in the output

    p_data = p_data.loc[:, out_cols]

    # create train, val and test sets -----------------------------------------------------------------------------------
    train_set, val_set, test_set = create_train_val_data(index_train = cv_data[key]["train"], 
                                                         index_test = cv_data[key]["test"]) # makes a val set out of the training set
    
    ## get data as tensors 
    target_data = [p_data.loc[:, "blups_scaled"].values.astype('float32'), \
                   g_data.astype('float32')]
        
    train_data = [x[train_set] for x in target_data]
    val_data = [x[val_set] for x in target_data]
    test_data = [x[test_set] for x in target_data]
    
    # customize input data and model tuner ------------------------------------------------------------------------------
    train_y = train_data[0]
    val_y = val_data[0]
    test_y = test_data[0]
    
    train_x = train_data[1]
    val_x = val_data[1]
    test_x = test_data[1]
    
    model_tuner = model.tuner(marker_n = g_data.shape[1],
                             tune = tune)
        
    # Hparam Tuning -------------------------------------------------------------------------------------------------
    if not exists(tuned_model_at):
        start_time_tuning = time.time()
        stop_early = EarlyStopping(monitor='val_loss', patience=5, min_delta = 0.01) # tweak here
        tb_cv_tuner = TensorBoard(tb_cb_tuning)
        tuner = kt.Hyperband(hypermodel=model_tuner,
                             objective=kt.Objective("val_mean_squared_error", direction="min"),
                             max_epochs=100,
                             factor=4,
                             hyperband_iterations=1,
                             overwrite = True,
                             directory=tuning_save_at,
                             project_name=tune_dir,
                             seed=30)
        tuner.search(train_x, train_y,
                     epochs=100,
                     validation_data=(val_x, val_y),
                     callbacks=[stop_early, tb_cv_tuner],
                     verbose=0)
        
        # save parameters
        for num_params in [3, 2, 1]:
            print(num_params)
            try:
                top3_params = tuner.get_best_hyperparameters(num_trials=num_params)
                if top3_params:
                    break  # If successful, exit the loop
            except tf.errors.NotFoundError as e:
                print("An error occurred:", e)
                if num_params == 1:
                    raise Exception("Error: Failed to retrieve best models with num_models=1. Script halted.")
        params = top3_params[0].values  # best hyperparameter values # can igonore warnings # https://stackoverflow.com/questions/58289342/tf2-0-translation-model-error-when-restoring-the-saved-model-unresolved-object
        write_json(params, param_save_at)
        
        # save model
        for num_models in [3, 2, 1]:
            print(num_models)
            try:
                top3_models = tuner.get_best_models(num_models=num_models)
                if top3_models:
                    break  # If successful, exit the loop
            except tf.errors.NotFoundError as e:
                print("An error occurred:", e)
                if num_models == 1:
                    raise Exception("Error: Failed to retrieve best models with num_models=1. Script halted.")
        best_model = top3_models[0]
        best_model.save(tuned_model_at)
        best_model = load_model(tuned_model_at) # loads a model with weights to run with test set directely
        #best_model = model_tuner(top3_params[0]) # alternative, but gives a weaker trend
 
        # clear space
        try:
            os.system(f'rm -rf {tuning_save_at}/{tune_dir}')
        except:
            logging.debug(f'Cannot delete {tuning_save_at}/{tune_dir}. Do it manually')
        
        # write Hparam tuning log
        end_time_tuning = time.time()
        elapsed_time_tuning = end_time_tuning - start_time_tuning
        logging.info(f'HP tuning took {elapsed_time_tuning} seconds, or {elapsed_time_tuning/60} minutes, or {elapsed_time_tuning/3600} hours')
        logging.info(f'parameters \n {pformat(params)}')
    else:
        logging.info(f'subsequent attempt at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}. Hparams were tuned earlier and were loaded for this run.')
        #os.system(f'rm -rf tune_dir')
        best_model = load_model(tuned_model_at) #todo: if log file exists then figure out a way to apeend lines rather than scratch all the previpous logs
    # Perform predictions -----------------------------------------------------------------------------------------------
    my_model = best_model
    if not exists(model_save_at):
        start_time_fit = time.time()
        fit_params = {'fit' : {'batch_size' : 32, # default is 32
                               'epochs' : 100,
                               'verbose' : 2,
                               'shuffle' : True,
                               'tensorboard_fp' : tb_cb,
                               'checkpoint_fp' : mc_cb}}
        my_model_fit = fit_model(final_model = my_model, params = fit_params, 
                                 train_x = train_x, 
                                 train_y = train_y, 
                                 val_x = val_x,
                                 val_y = val_y)
        my_model_fit.save(model_save_at)
        #save_model(model = my_model_fit, path = model_save_at, model_name = f'model_{key}')
        end_time_fit = time.time()
        elapsed_time_fit = end_time_fit - start_time_fit
        logging.info(f'Model fitting took {elapsed_time_fit} seconds, or {elapsed_time_fit/60} minutes, or {elapsed_time_fit/3600} hours')
    else:
        logging.info(f'subsequent attempt at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}. Model fit was done earlier and weights were loaded for this run.')
        my_model_fit = load_model(model_save_at) #todo: if log file exists then figure out a way to apeend lines rather than scratch all the previpous logs
    # Export data -------------------------------------------------------------------------------------------------------
    ## raw results
    pred_vals_test = predict_values(model = my_model_fit, 
                                    test_x = test_x, 
                                    test_y = test_y, 
                                    index = test_set, 
                                    scaler = scaler_p)
    pred_vals_test = pd.merge(pred_vals_test, 
                              p_data, 
                              how='left', 
                              left_on=['index'], 
                              right_index=True)
    pred_vals_test.to_csv(path_to_pred_file, index=False)

else:
    # define logger -----------------------------------------------------------------------------------------------------
    logging.basicConfig(filename=logs_at, level=logging.DEBUG, filemode='a')
    logger = logging.getLogger(__name__)
    logging.info(f'Followup attempt at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}. The whole script was not executed since pred/output.csv exists. You will need to delete it for script to work.')
