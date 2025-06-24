# Groundwater Level Forecasting - India 🌊

This project estimates and forecasts groundwater levels across various regions in India using data sourced from the [India Data Portal](https://ckandev.indiadataportal.com). The goal is to explore regional trends, seasonal patterns, and build predictive models that can assist in sustainable water management.

---

## 📊 Features

- 📥 Load and clean groundwater datasets (state, district, block level)
- 🔍 Perform exploratory data analysis (spatial, temporal trends)
- 🧠 Train predictive models (time series & machine learning)
- 🌐 Interactive visualizations and dashboard (Streamlit)
- 📌 Ready for extension with GIS mapping

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/groundwater-level-forecasting-india.git
   cd groundwater-level-forecasting-india

2. **Create a virtual environment & install dependencies**
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt

3. **Run the Streamlit app (after modeling)**
    ```bash
    streamlit run app/main.py
