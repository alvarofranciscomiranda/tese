
# Tese

This README provides instructions for setting up and running a Python project. The project assumes you have at least Python 3.3 installed and uses virtual environments for managing dependencies. The project also requires the installation of Python packages specified in the requirements.txt file.

## Setting Up the Project
Follow these steps to set up the project:

### 1. Ensure Python 3.8.12 is Installed for example
Make sure you have Python 3.8.12 installed on your system. You can check your Python version by running:

```python --version```

If Python 3.8.12 is not installed, please download and install it from the official Python website: Python Downloads.

### 2. Install pyenv (Optional)
If you wish to manage multiple Python versions, you can use pyenv. To set your local Python version to 3.8.12 with pyenv, run:

```pyenv local 3.8.12```

This step is optional, and you can skip it if you're not using pyenv.

### 3. Create a Virtual Environment
Create a Python virtual environment (venv) to isolate the project dependencies. You can do this by running:

```python -m venv venv```

This command will create a virtual environment named "venv" in your project directory.

### 4. Activate the Virtual Environment
Before running the project, activate the virtual environment:

On Linux/macOS:

```source venv/bin/activate```

On Windows:

'venv\Scripts\activate'
You should see your terminal prompt change to indicate that the virtual environment is active, for example: (venv).

### 5. Install Required Packages
Install the project dependencies specified in the requirements.txt file using pip:

```pip install -r requirements.txt```

This will install the necessary packages for your project.

## Running the Project
After setting up the project, you can run it with the following command:

```python main.py```

This command will execute the main.py script, which should contain the main functionality of your project.

Deactivating the Virtual Environment
When you're done working on the project, you can deactivate the virtual environment and return to your global Python environment:

```deactivate```

## Additional Notes
Remember to manage your project-specific dependencies within the virtual environment to avoid conflicts with other Python projects.

You can customize the project's main script, dependencies, and project structure to suit your specific needs.
