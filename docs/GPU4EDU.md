# How to use GPU4EDU (Notes to myself)

## Structure of the Cluster
The computing cluster consists of multiple **nodes**, which are individual machines equipped with resources like CPUs, GPUs, memory, and storage. These nodes are the workhorses that handle all computational tasks. When you run a job, you are effectively requesting time and resources on one or more of these nodes, depending on the complexity of your task. The cluster's management system, SLURM, allocates these nodes to users dynamically.

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
6. **Test Your Code**: Run small tests interactively on the node to verify everything works as expected.
7. **Submit a Job**: Create a SLURM job script that specifies the resources your job needs, such as GPUs and runtime limits. Submit this script to the cluster for execution.
8. **Monitor Progress**: While your job is running, logs and outputs will be saved to files that you can check to monitor its status.

## Waiting Times and Queues
If the cluster is busy, your job may not start immediately and will be placed in a queue. SLURM prioritizes jobs based on several factors, such as the resources requested, the current load on the cluster, and fair share policies that ensure all users get equitable access. Jobs requiring fewer resources or submitted during off-peak hours typically experience shorter waiting times.

## Downloading Results
Once your job is complete, you can retrieve the output and log files from the cluster to your local computer. Use secure file transfer methods like SCP, FileZilla, or RSYNC to download your results. This allows you to analyze or archive your data conveniently on your own machine.
