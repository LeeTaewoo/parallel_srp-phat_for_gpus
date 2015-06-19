# Parallel SRP-PHAT for GPUs
Simple Experiment on Simulated Data.
### Execution
On Windows and MATLAB, (the codes were tested on Windows7 64-bit and MATLAB2009b).
 1. Download `LDC93S1.wav` file from [here](https://www.google.com/url?q=https%3A%2F%2Fcatalog.ldc.upenn.edu%2Fdesc%2Faddenda%2FLDC93S1.wav&sa=D&sntz=1&usg=AFQjCNE1QtQownD3lvimnRxuWBXkutWotg). Copy the file into `1_gen_simul_data` folder.
 2. Execute `./1_gen_simul_data/main.m`.
 3. Execute `./2_data_preprocessing/main.m`.
 4. Execute `./3_gen_TDOA_tables/frequency_domain/main.m`.
 5. Execute `./3_gen_TDOA_tables/time_domain/main.m`.

On Linux which is installed a CUDA, (the codes were tested on CUDA 7.0, GTX TITAN X, Ubuntu 14.04.02 64-bit).
 1. Execute `./4_parallel_srp-phat/run.sh` on a terminal. (Please make sure that the execution permissions of `*.sh` files are set correctly. If not, you may set the permission with `chmod 777 *.sh` command).

Experiment result log files are saved in `./4_parallel_srp-phat/exp_total/result/log`. For more detail explanation, please refer to the paper below.

### Reference
Taewoo Lee, Sukmoon Chang, and Dongsuk Yook, ["Parallel SRP-PHAT for GPUs,"](http://www.sciencedirect.com/science/article/pii/S0885230815000455)
 Computer Speech and Language, vol. 35, pp. 1-13, Jan. 2016
