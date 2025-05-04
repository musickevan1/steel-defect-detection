# Running Steel Defect Detection Training on RunPod.io GPU

This guide provides step-by-step instructions for running the steel defect detection training on a RunPod.io GPU instance.

## Step 1: Prepare Your Project

First, prepare your project for upload:

1. Zip the entire project directory:
   ```bash
   # On Windows
   Compress-Archive -Path "C:\path\to\project\folder" -DestinationPath "steel_defect_project.zip"
   
   # On Linux/Mac
   zip -r steel_defect_project.zip "path/to/project/folder"
   ```

2. Make sure your zip file includes:
   - All Python code files
   - The NEU-DET dataset
   - requirements.txt and setup.py
   - All wrapper scripts

## Step 2: Set Up RunPod.io

1. Create an account on [RunPod.io](https://www.runpod.io/) if you don't have one already.

2. Deploy a GPU pod:
   - Select a GPU type (A100, A6000, etc. - even a T4 would be sufficient for this project)
   - Choose a template with PyTorch (e.g., "PyTorch 2.0.1 with CUDA 11.8")
   - Set the disk size (at least 10GB)
   - Deploy the pod

3. Once the pod is running, click "Connect" and select "Jupyter Lab" or "SSH" (we'll use Jupyter Lab for this guide).

## Step 3: Upload and Extract Your Project

1. In Jupyter Lab, navigate to the file browser on the left.

2. Click the upload button (up arrow) and select your `steel_defect_project.zip` file.

3. Open a terminal in Jupyter Lab by clicking the Terminal icon.

4. Extract the zip file:
   ```bash
   unzip steel_defect_project.zip
   ```

5. Navigate to the extracted directory:
   ```bash
   cd "Final Project"
   ```

## Step 4: Install Dependencies

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Specifically install tensorboard and tensorboardX:
   ```bash
   python install_tensorboard.py
   ```

## Step 5: Run Training

### Option 1: Run a Single Experiment

To run a single experiment with specific hyperparameters:

```bash
python debug_experiment.py --batch_size 32 --lr 0.001 --epochs 15
```

This will run a single training experiment with batch size 32, learning rate 0.001, and 15 epochs.

### Option 2: Run Multiple Experiments

To run multiple experiments with different hyperparameters:

```bash
python run_experiments.py
```

When prompted, select option 1 to run CNN experiments. This will run experiments with all combinations of batch sizes and learning rates defined in `config.py`.

### Option 3: Run Specific Training Script

To run the training script directly with GPU acceleration:

```bash
python CSC_573_Final_Project_Code/train_lightning.py --batch_size 32 --lr 0.001 --epochs 15
```

## Step 6: Monitor Training

1. You can monitor the training progress in the terminal output.

2. For TensorBoard visualization, open a new terminal and run:
   ```bash
   tensorboard --logdir=logs
   ```

3. Access TensorBoard by clicking on the provided URL or by going to the "Services" tab in RunPod and clicking on the TensorBoard port (usually 6006).

## Step 7: Download Results

After training is complete:

1. Zip the results directory:
   ```bash
   zip -r results.zip results
   ```

2. In Jupyter Lab, right-click on `results.zip` and select "Download".

3. You can also download the saved models:
   ```bash
   zip -r saved_models.zip saved_models
   ```

## Additional Tips for RunPod

1. **Persistent Storage**: Consider using RunPod's persistent storage to keep your data and results between sessions.

2. **GPU Monitoring**: Use `nvidia-smi` command to monitor GPU usage:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Cost Management**: Remember to stop your pod when not in use to avoid unnecessary charges.

4. **Jupyter Notebook**: You can also create a Jupyter notebook to run and document your experiments interactively.

5. **Environment Variables**: Set the following environment variable to use all available GPU memory:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

## Troubleshooting

1. **CUDA Out of Memory**: If you encounter CUDA out of memory errors, try reducing the batch size.

2. **Module Not Found Errors**: Make sure all dependencies are installed correctly.

3. **Permission Issues**: If you encounter permission issues, try:
   ```bash
   chmod -R 755 "Final Project"
   ```

4. **Data Loading Issues**: Verify that the NEU-DET dataset is correctly extracted and located in the expected directory.