# Tutorial for runnning a CNN for genomic preidiction

This project extends Convolutional Neural Networks (CNNs) to Big Data, particularly focusing on genomic predictions with **BGLR**. Below are the steps and requirements for setting up the necessary computing environment using **OpenBLAS** and **Python** inside containers.

## First steps
Clone this repository and ty running
     ```bash
    ./manage_cc help
    ```

This call to the script will try to find a container and associated scripts at /filer-5/agruppen/QG/gogna. If this is not found you will get an error. See requirements section below for a solution to this. 

## Requirements
Ensure that you have access to the container. I put one at /filer-5/agruppen/QG/gogna. 

In addition, ensure the following paths are correctly set up in manage_cc:
- `cc_dir="../computing_containers/containers"`
- `usr_scr="../computing_containers/usr_scr"`
- `ext_lib_blas="../computing_containers/openblas_3.23/inst/qg-10.ipk-gatersleben.de/lib/libopenblas.so"` # see below

Lastly, you need to create a directory for storing large volume files at /qg-10/data/AGR-QG/temp. Modify this path at line 21 in manage_cc. Easiest way to do this is
    ```bash
    usr_name=$(whoami)
    mkdir /qg-10/data/AGR-QG/temp/${usr_name}/tut_CNN
    ```

In case you did not set up BLAS (next section), i suggest you put "#" in front of line 18 in manage_cc. You can use 
     ```bash
    nano manage_cc
    ```
to edit the script in the terminal itself.

## Starting the container

By starting the container i mean the start of jupyter server. This is possible with
     ```bash
    ./manage_cc start_jup
    ```
   This creates a directory cc_data where all the data/files associated with the session are present. For instance, the token for settign up jupyter password is present in
    ```bash
    ./cc_data/jup/{session_name}/run.err
    ```
session name, e.g. jup_tut_CNN_xx is output on the terminal when you start the container. An address to access the container is also printed, e.g. 127.0.0.1:8026. To access the container you need to access this address using a web browser like chrome. 

For safety reasons, you need to set up a tunnel (read more at https://github.com/IPK-QG/bench_setup/blob/master/docs/computing-clusters.md). I would prefer using MobaXterm for this over putty for ease of use. A portable version is available at https://mobaxterm.mobatek.net/download-home-edition.html. You can start it by simply running the .exe file and then start a session at qg-10/slurm using the "session" button on top left corner. Also close to the session button, a tunneling button is available. Use this to set up a tunnel with following specs but with your user name,   
![image](https://github.com/user-attachments/assets/f41f76d8-b22f-49d0-9396-b74d222a020f)


## OpenBLAS Setup

**OpenBLAS** is required when running genomic predictions with **BGLR**. Follow these steps to install and configure OpenBLAS:

1. Start a shell session in the container.
2. Clone the OpenBLAS repository:
    ```bash
    git clone -b v0.3.23 https://github.com/xianyi/OpenBLAS
    cd OpenBLAS
    ```
3. Create a symbolic link for the `libmpfr.so` library:
    ```bash
    ln -s /usr/lib/x86_64-linux-gnu/libmpfr.so.6 ../computing_containers/lib_symlinks/libmpfr.so.4
    ```
4. Set the `LD_LIBRARY_PATH` environment variable:
    ```bash
    export LD_LIBRARY_PATH="/qg-10/data/AGR-QG/Gogna/computing_containers/lib_symlinks:$LD_LIBRARY_PATH"
    ```
5. Build and install OpenBLAS:
    ```bash
    make DYNAMIC_ARCH=1
    make install PREFIX="inst/$(hostname)"
    ```
6. Use `sessionInfo()` in R to find the default BLAS library location.
7. Bind the absolute path of the installed BLAS library to the path provided by `sessionInfo`. For example:
    ```bash
    "${OpenBLAS_lib}:/usr/local/lib/R/lib/libRblas.so"
    ```
   where `OpenBLAS_lib="/qg-10/data/AGR-QG/Gogna/computing_containers/OpenBLAS/inst/qg-10.ipk-gatersleben.de/lib/libopenblas.so"`

8. Optionally, you can benchmark OpenBLAS using this script:
    ```bash
    https://mac.r-project.org/benchmarks/R-benchmark-25.R
    ```
    OpenBLAS completed this benchmark in 8 seconds on `qg-10`.

## Python Environment Setup

To set up the Python environment inside the container, follow these steps:

1. Run the following command inside the container to create a requirements file:
    ```bash
    echo -e "## for DL\n\
    tensorflow==2.8\n\
    tensorboard==2.8\n\
    pyarrow==5.0.0\n\
    matplotlib==3.5.1\n\
    pandas==1.4\n\
    scikit-learn==1.0.2\n\
    patsy==0.5.2\n\
    protobuf==3.19.6\n\
    keras-tuner==1.1.3\n\
    ipykernel==6.22.0\n\
    ## for ML\n\
    xgboost==2.0.0\n\
    ## for doit\n\
    graphviz==0.20\n\
    doit==0.36.0\n\
    pygraphviz==1.9\n\
    import_deps==0.2.0" > "/proj/requirements.txt"
    ```
    You can add or remove libraries as needed in the `requirements.txt` file.

2. Set up the Python environment:
    ```bash
    python3 -m venv /proj/py_env
    source /proj/py_env/bin/activate
    pip3 --no-cache-dir install -r /proj/requirements.txt
    ```

3. Ensure the environment is loaded each time the container starts:
    ```bash
    echo "source /proj/py_env/bin/activate" > /proj/.bash_profile
    ```

4. Add this environment to Jupyter:
    ```bash
    python3 -m ipykernel install --user --name=py_env
    ```
   Note: You may need to run this agian and referh if the notebook does not detect your kernel (see step 5.). 

5. Refresh Jupyter and select `py_env` from the kernel dropdown in the top-right corner of the notebook.

---

This setup will ensure you have a reproducible environment for your project, including all necessary dependencies for both genomic predictions with BGLR and machine learning tasks.

## Using this directory

1.  Try to run the command in a terminal inside the container
        ```bash
        cd /proj/run/Py
        doit
        ```
    This should work and output "works" on your terminal 

2.  Try running the notebook file in notebooks directory. 
