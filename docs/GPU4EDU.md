#[How to use GPU4EDU](https://ilk.uvt.nl/~shterion/gpu4edu.html)

## Structure of the Cluster
The computing cluster consists of multiple **nodes**, which are individual machines equipped with GPUs. These nodes handle all computational tasks. When you run a job, you are effectively requesting time and resources on one or more of these nodes, depending on the complexity of your task. The cluster's management system, SLURM, allocates these nodes to users dynamically.

## Step-by-Step Plan to Run a Script
1. **Access the Cluster**: Connect to the cluster using your university credentials via SSH.
   ```bash
   ssh [your_u-number]@aurometalsaurus.uvt.nl
   ```
3. **Request a Node**: Use the SLURM system to reserve a node where you can test and run your scripts interactively.
   ```bash
   srun --nodes=1 --pty /bin/bash -l
   ```
5. **Set Up Your Environment**: Activate or create an Anaconda environment tailored to your project's requirements to ensure the correct software and libraries are available.
   - To activate an existing environment:
     ```bash
     source activate [ENVNAME]
     ```
   - To create a new environment:
     ```bash
     conda create -n [ENVNAME]
     ```
7. **Test Your Code**: Run small tests interactively on the node to verify everything works as expected.
8. **Submit a Job**: Create a SLURM job script that specifies the resources your job needs, such as GPUs and runtime limits. Submit this script to the cluster for execution.
   ```bash
   #!/bin/bash
   #SBATCH -p GPU
   #SBATCH -N 1
   #SBATCH -t 0-36:00
   #SBATCH -o slurm.%N.%j.out
   #SBATCH -e slurm.%N.%j.err
   #SBATCH --gres=gpu:1
   
   if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
       . "/usr/local/anaconda3/etc/profile.d/conda.sh"
   else
       export PATH="/usr/local/anaconda3/bin:$PATH"
   fi
   
   source activate [ENVNAME]
   python your_script.py
   ```
   Submit your job using:
   ```bash
   sbatch your_job_script.sh
   ```
10. **Monitor Progress**: While the script is running it will generate two files - slurm.[NODE].[JOBNUMBER].err and slurm.[NODE].[JOBNUMBER].out, where [NODE] and [JOBNUMBER] are the name of the node which is running your script and the number of the job that has been submitted, respectively. Here is an example: slurm.cerulean.118.err slurm.cerulean.118.out The .err file contains logging information such as warnings and errors; the .out file contains the actual output that you have asked your script to generate.

## Waiting Times and Queues
If the cluster is busy, your job may not start immediately and will be placed in a queue. SLURM prioritizes jobs based on several factors, such as the resources requested, the current load on the cluster, and fair share policies that ensure all users get equitable access. Jobs requiring fewer resources or submitted during off-peak hours typically experience shorter waiting times.

## Downloading Results
Once your job is complete, you can retrieve the output and log files from the cluster to your local computer. Use secure file transfer methods like SCP or FileZilla to download your results. This allows you to analyze or archive your data conveniently on your own machine.
```bash
scp [your_u-number]@aurometalsaurus.uvt.nl:/path/to/remote/file /path/to/local/destination
```
