**## Setting Up Your Data Analysis Environment for CardioGood Fitness Customer Profiles**

This guide will help you establish a virtual environment and install the necessary libraries to run the Jupyter Notebook for analyzing CardioGood Fitness customer profiles. Additionally, it will explain how to run the Streamlit application for uploading PDFs and having AI read them using Ollama.

**Prerequisites:**

* **Git:** Ensure you have Git installed on your system. You can download it from [https://git-scm.com/](https://git-scm.com/).
* **Python:** Download and install Python from [https://www.python.org/](https://www.python.org/).
* **Text Editor/IDE:** Choose your preferred text editor or IDE (e.g., Visual Studio Code, PyCharm) to follow along with the code.

**Steps:**

1. **Clone the Repository:**

  Open a terminal or command prompt and navigate to your desired working directory. Run the following command, replacing `[repo]` with the actual URL of the repository containing the Jupyter Notebook:

  ```bash
  git clone [repo]
  ```

2. **Create and Activate a Virtual Environment:**

  A virtual environment isolates project dependencies, preventing conflicts with other Python projects.

  ```bash
  py -m venv [name of environment]  # Replace [name of environment] with your preferred name (e.g., cardiogood_env)
  ```

  Activate the virtual environment (replace `[name of environment]` accordingly):

  **Windows:**

  ```bash
  [name of environment]\Scripts\activate
  ```

  **macOS/Linux:**

  ```bash
  source [name of environment]/bin/activate  # Or source [name of environment]/venv/bin/activate (depending on your Python version)
  ```

3. **Install Required Packages:**

  Navigate to the project directory:

  ```bash
  cd [repo]
  ```

  Install the libraries listed in the `requirements.txt` file:

  ```bash
  pip install -r requirements.txt
  ```

4. **Run the Jupyter Notebook (Optional):**

  Open a terminal or command prompt within the project directory and type:

  ```bash
  jupyter notebook
  ```

  This will launch the Jupyter Notebook interface in your web browser, typically at `http://localhost:8888`. You can use this for detailed analysis.

5. **Start Your Data Analysis Journey (Optional, for Jupyter Notebook):**

  If you chose to use the Jupyter Notebook:

    * Open the Jupyter Notebook named `Data-Analysis-Understanding-CardioGood-Fitness-Customer-Profiles-for-Market-Research.ipynb` (or a similar name, depending on the repository).
    * Follow the instructions and code examples in the notebook to analyze and visualize CardioGood Fitness customer data.
    * Feel free to experiment with the code and customize it according to your specific needs.

**Setting Up Ollama for the Streamlit Application**

Now, let's set up Ollama for the Streamlit application that will handle PDF upload and AI analysis:

1. **Download and Install Ollama:**

  * **Install Ollama:**
      ```bash
      pip install ollama
      ```

  * **Download a Model:**
      * Ollama supports various models. Choose a model that suits your needs (e.g., `codellama`, `llama2`, `vicuna`).
      * Use the Ollama CLI to download the model:
        ```bash
        ollama models download <model_name> 
        ```
        (Replace `<model_name>` with the desired model name)

**Running the Streamlit Application**

6. **Run the Streamlit Application:**

  To launch the Streamlit application for PDF uploading and AI reading, navigate to the directory containing the `pdf_rag.py` script and run:

  ```bash
  streamlit run pdf_rag.py
  ```

  This will open a web app in your browser. You can then use the app to:

    * Upload a PDF document.
    * Trigger the AI to read the PDF content using the Ollama model.
    * (Potentially) interact with the AI's interpretation of the PDF (e.g., ask questions, summarize, etc.)

**Additional Tips:**

* The `requirements.txt` file likely includes libraries for PDF handling, natural language processing (NLP), and potentially interaction with an AI model (e.g., a language model API).
* If you encounter any errors during installation, please check the version of python used. At the time of installing the packages. version 3.13 was used.