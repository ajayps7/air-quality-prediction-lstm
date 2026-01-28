**1. Library imports and environment setup**

The notebook begins by importing all required libraries for:

Data handling: pandas, numpy

Visualization: matplotlib, seaborn

Machine learning utilities: sklearn

Deep learning: torch, torch.nn, torch.optim

Serialization: joblib

PyTorch is used instead of Keras/TensorFlow, which means the model training loop is fully manual and explicit, not abstracted.

**2. Data loading**

The AQI dataset is loaded from a CSV file into a pandas DataFrame.

Immediately after loading:

The Date column is converted to datetime

Data is sorted by City and Date

This ordering is mandatory because later operations (lag features, rolling windows, LSTM sequences) assume strict temporal order per city.

**3. Initial exploratory data analysis (EDA)**

The notebook performs exploratory analysis in three phases:

Univariate analysis

Distribution plots are generated for AQI and pollutant variables

This is used to understand skewness, spread, and abnormal values

Bivariate analysis

Relationships between AQI and pollutant concentrations are visualized

This confirms which features have strong influence on AQI

Outlier detection

Outliers are detected visually using boxplots

Extreme values are inspected before treatment

This EDA phase is diagnostic only—it does not modify the dataset yet.

**4. Missing value handling (city-aware)**

Missing values are handled per city, not globally.

Steps:

All numeric columns are identified programmatically

For each city:

Missing values are interpolated using time-based interpolation

Any remaining NaNs are filled with the median of that column for that city

This preserves:

Temporal continuity

City-specific pollution behavior

No leakage across cities

**5. Explicit NaN replacement for pollutant columns**

After interpolation and median filling, a second safeguard is applied.

Specific pollutant columns (PM10, NOx, NH3, O3, Benzene, Toluene, Xylene) have remaining NaNs replaced with fixed constants.

This guarantees zero missing values, even at time-series boundaries.

**6. Outlier treatment**

Outliers detected earlier are treated using statistical thresholds (IQR-based logic).

This step:

Reduces extreme noise

Prevents unstable gradients during LSTM training

Preserves overall distribution shape

A second visualization pass confirms outliers have been handled.

**7. Feature engineering (time-series features)**

This is the most important transformation step.

For each city independently, the following features are created:

AQI_lag_1 → AQI from the previous day

AQI_lag_7 → AQI from 7 days earlier

AQI_roll_7 → 7-day rolling mean of AQI

These features inject temporal memory into the dataset before it ever reaches the LSTM.

**8. Post-engineering cleanup**

Lag and rolling operations introduce NaNs at the beginning of each city’s timeline.

All rows containing NaN values after feature engineering are dropped.

This ensures:

Every training sample is fully defined

No conditional logic inside the model

**9. Column pruning**

The categorical column AQI_Bucket is dropped.

Reason:

The task is regression, not classification

Buckets would introduce leakage and redundancy

**10. Feature–target separation**

The dataset is split into:

FEATURES → all predictor variables

TARGET → AQI

This separation is explicit and fixed, preventing accidental feature drift later.

**11. Feature scaling**

Two independent MinMaxScaler instances are used:

Feature scaler → scales input features

Target scaler → scales AQI values

Both are scaled into the [0, 1] range.

This is critical because:

LSTMs are sensitive to feature magnitude

AQI values are much larger than pollutant concentrations

**12. Train–test split (time-aware)**

The dataset is split sequentially:

First 80% → training set

Last 20% → test set

No shuffling is performed.

This preserves causality and prevents future data from leaking into training.

**13. Conversion to PyTorch tensors**

The scaled NumPy arrays are converted into PyTorch tensors.

The feature tensors are reshaped into 3D format:

(samples, time_steps, features)


This shape is mandatory for LSTM input.

**14. LSTM model definition**

A custom PyTorch model class is defined:

One LSTM layer with:

Input size = number of features

Hidden size = 64

One fully connected (dense) layer mapping hidden state → AQI

The model outputs a single continuous value per sequence.

**15. Training configuration**

The training setup includes:

Loss function: Mean Squared Error (MSE)

Optimizer: Adam

Learning rate: 0.001

Epochs: 20

Loss tracking lists are initialized for both training and testing.

**16. Manual training loop**

The training loop is written explicitly (no framework abstraction):

For each epoch:

Model set to training mode

Forward pass on training data

Loss computed

Gradients cleared

Backpropagation executed

Optimizer updates weights

Training loss recorded

This explicit loop makes model behavior fully transparent.

**17. Model evaluation per epoch**

After each epoch:

Model switches to evaluation mode

Predictions are generated on test data

Test loss (MSE) is calculated

Test loss is recorded

This allows monitoring overfitting in real time.

**18. Loss visualization**

Training and testing losses are plotted across epochs.

This visualization confirms:

Convergence behavior

Stability of learning

Absence of severe overfitting

**19. Model persistence**

Finally, the trained artifacts are saved:

Trained LSTM model weights

Feature scaler

Target scaler

These files are later reused unchanged by:

Streamlit frontend

FastAPI backend
