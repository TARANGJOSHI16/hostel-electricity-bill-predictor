# ⚡ Hostel Electricity Consumption Predictor

An end-to-end Machine Learning web application designed to estimate and analyze the electricity consumption of hostel rooms based on room occupancy and electrical appliance schedules.

Developed as a college-level academic project using **Python**, **Pandas**, **Scikit-learn**, and **Streamlit**.

---

## 📖 1. What is an Electricity Unit?

> **1 Unit of Electricity = 1 kWh (Kilowatt-Hour)**
> 
> A **Kilowatt-Hour (kWh)** represents the amount of electrical energy consumed by a device of **1,000 Watts** operating continuously for **1 Hour**.
> 
> *Examples:*
> - Running a 1,000 W water heater for 1 hour consumes **1 unit** (1 kWh).
> - Running a 100 W bulb for 10 hours consumes \(100 \times 10 / 1000 = \mathbf{1\text{ unit}}\).
> - Running a 75 W ceiling fan for 14 hours consumes \(75 \times 14 / 1000 = \mathbf{1.05\text{ units}}\).

State electricity boards bill consumer electricity tariffs based on these units.

---

## 🗂️ 2. Project Directory Structure

```text
hostel-electricity-predictor/
├── data/
│   └── hostel_electricity.csv       # Synthetic dataset with 1,500 room-day records
├── models/
│   └── electricity_model.joblib     # Saved machine learning model & metadata
├── generate_data.py                 # Generates reproducible synthetic dataset
├── train_model.py                   # Trains and compares ML models (evaluates MAE, RMSE, R²)
├── predictor.py                     # Prediction helper module & appliance breakdown logic
├── app.py                           # Streamlit interactive web frontend with Matplotlib charts
├── requirements.txt                 # Project library dependencies
├── README.md                        # Documentation & setup guide
└── .gitignore                       # Git ignore configuration
```

---

## 🔬 3. Appliance Wattage Assumptions & Physics Formula

Energy consumption is governed by the standard physical formula:

$$\text{Energy (kWh)} = \frac{\text{Power (Watts)} \times \text{Usage Duration (Hours)}}{1000}$$

| Appliance | Rated Power (Watts) | Typical Daily Usage | Realistic Daily Energy |
| :--- | :--- | :--- | :--- |
| **Ceiling Fan** | 75 W | 6 – 22 hours | 0.45 – 1.65 kWh per fan |
| **LED Tube / Light** | 25 W | 3 – 12 hours | 0.08 – 0.30 kWh per light |
| **Air Conditioner** (1.5 Ton) | 1,200 W (effective load) | 3 – 12 hours | 3.60 – 14.40 kWh per AC |
| **Student Laptop Charger** | 65 W | 2 – 7 hours/student | 0.13 – 0.45 kWh per laptop |
| **Other Appliances** (kettle, phone charger, iron) | Variable | Intermittent | 0.20 – 2.00 kWh base load |

---

## 🚀 4. Step-by-Step Setup Guide (Mac M3 & VS Code)

Follow these simple steps in order:

### Step 1: Open the Project in VS Code
1. Open **Visual Studio Code**.
2. Click **File > Open Folder...** and select the `hostel-electricity-predictor` directory.
3. Open the built-in terminal by pressing `` Ctrl + ` `` (Control + Backtick) or selecting **Terminal > New Terminal**.

### Step 2: Create and Activate a Python Virtual Environment
Run the following commands in the terminal:
```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```
*(Your terminal prompt will now show `(.venv)` at the beginning, confirming it is active).*

### Step 3: Install Required Libraries
```bash
pip install -r requirements.txt
```

### Step 4: Generate the Dataset
Run `generate_data.py` to create the reproducible 1,500-sample dataset:
```bash
python generate_data.py
```
This automatically creates the `data/` folder and saves `data/hostel_electricity.csv`.

### Step 5: Train the Machine Learning Model
Train and evaluate the regression models:
```bash
python train_model.py
```
This trains both **Linear Regression** and **Random Forest Regressor**, logs evaluation metrics, creates the `models/` folder, and saves `models/electricity_model.joblib`.

### Step 6: Test the Predictor Module (Optional)
Verify that the model loads and predicts without issues:
```bash
python predictor.py
```

### Step 7: Launch the Streamlit Web Application
Start the interactive user interface:
```bash
streamlit run app.py
```
Streamlit will launch local server and automatically open the application in your browser (usually at `http://localhost:8501`).

---

## 📊 5. Machine Learning Methodology

### Features Used:
1. `occupants`: Total students living in the room
2. `fans`: Number of ceiling fans
3. `fan_hours`: Operating hours per fan per day
4. `lights`: Number of tube lights / bulbs
5. `light_hours`: Operating hours per light fixture
6. `ac_units`: Number of air conditioners installed
7. `ac_hours`: Operating hours per AC unit
8. `laptop_hours`: Total room laptop usage hours
9. `other_appliance_kwh`: Miscellaneous appliances (chargers, kettle, iron)

### Target Variable:
- `daily_kwh`: Daily electricity consumption in Kilowatt-Hours (Units).

### Evaluation Metrics:
- **\(R^2\) Score (Coefficient of Determination):** Measures the proportion of variance explained by the model (> 0.98 on our dataset).
- **Mean Absolute Error (MAE):** Average error magnitude in units/kWh (~0.32 kWh).
- **Root Mean Squared Error (RMSE):** Penalizes larger prediction deviations (~0.46 kWh).

---

## 🎓 6. College Viva / Project Presentation Q&A

**Q1: Why did you use Machine Learning instead of just a static formula?**  
> *Answer:* In real hostel buildings, electrical consumption does not strictly follow simple linear formulas. Occupancy density, phantom standby losses, simultaneous usage, ambient seasonal conditions, and metering fluctuations introduce non-linearities. Machine Learning models learn these patterns from historical metering data and generalize better to complex occupant behaviors.

**Q2: What does \(R^2\) score represent?**  
> *Answer:* The \(R^2\) score (or Coefficient of Determination) indicates the percentage of variance in electricity consumption that the model successfully captures. A score of 0.99 means that 99% of variations in daily electricity consumption are accurately accounted for by the room and appliance features.

**Q3: How is the monthly estimate calculated?**  
> *Answer:* The machine learning model first predicts the room's baseline daily consumption in kWh. The monthly estimate is then calculated as:
> $$\text{Monthly Units} = \text{Daily Predicted kWh} \times \text{Configured Days (default 30)}$$

**Q4: Which appliance consumes the most electricity in a typical hostel room?**  
> *Answer:* As demonstrated in the feature importance analysis and appliance breakdown visualizations, the **Air Conditioner** is by far the highest consumer (~1.2 kWh per operating hour), followed by ceiling fans when operated for prolonged durations (16+ hours).

---

## 🛠️ 7. Technologies Used
- **Python 3.10+**: Core programming language.
- **Pandas**: Tabular dataset processing and manipulation.
- **NumPy**: Numerical computation and reproducible data synthesis.
- **Scikit-learn**: Machine learning model training, evaluation, and train-test splitting.
- **Streamlit**: Responsive, interactive web frontend.
- **Matplotlib**: Donut charts and horizontal bar charts for appliance breakdown.
- **Joblib**: Efficient serialization of trained model objects and metadata.
