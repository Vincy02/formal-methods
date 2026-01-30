# 📊 Process Mining Dashboard - BPI Challenge 2013

This project is a **Process Mining** application designed to analyze the **BPI Challenge 2013** dataset (Volvo IT Incident Management). It provides tools for data exploration, process discovery, and AI-driven insights using **Streamlit**, **PM4Py**, and **OpenAI**.

## 📂 Dataset
The dataset used in this project is the **BPI Challenge 2013** (closed problems).<br>
🔗 **Download here:** [4TU.ResearchData](https://data.4tu.nl/articles/_/12714476/1)

## ✨ Features
- **Data Exploration**: Detailed statistics on cases, events, and resources.
- **Anomaly Detection**: Automatic identification of bottlenecks, overloaded resources, and complex cases.
- **Process Discovery**: Visualization of process models using **Alpha Miner**, **Heuristic Miner**, and **Inductive Miner**.
- **AI Chatbot**: Ask natural language questions about the process (powered by OpenAI).
- **Predictive Analytics**: Predict the next activity in a case using a hybrid approach (Statistical Rules + LLM).

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd FormalMethods
   ```

2. **Create and activate a virtual environment** (optional but recommended).

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**:
   Create a `.env` file in the root directory and add your OpenAI API Key.
   ```
   OPENAI_API_KEY=YOUR_KEY_HERE
   ```

## 🖥️ Usage
### Run basic data exploration
```bash
python main.py
```

### Run process discovery and generate metrics
```bash
python discovery.py
```

### Generate AI-powered analysis report
```bash
python reasoning.py
```

### Launch interactive dashboard
```bash
streamlit run dashboard.py
```

## 📁 Project Structure
- `dashboard.py`: Main Streamlit application.
- `main.py`: Standalone script for basic stats and DFG visualization.
- `discovery.py`: Generates Petri net models (images) and calculates quality metrics (fitness, precision, etc.).
- `reasoning.py`: Generates the AI analytical report (`ai_report.md`).
- `requirements.txt`: Python dependencies.
- `*.png`: Generated process models and charts.
- `ai_report.md`: AI-generated business report.
