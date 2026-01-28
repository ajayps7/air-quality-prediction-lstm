**FastAPI Backend Description**

The FastAPI backend acts as a RESTful inference service for the trained LSTM-based AQI prediction model. Its sole responsibility is to receive structured input data, preprocess it exactly as done during training, perform model inference, and return a clean, machine-readable prediction response.

There is no training, no data persistence, and no state mutation—this backend is purely for prediction.

**Application initialization**

At startup, the FastAPI application initializes a single app instance and immediately loads all required artifacts:

The trained LSTM model

The feature scaler (MinMaxScaler)

The target scaler (MinMaxScaler)

These objects are loaded once into memory, not per request. This design:

Minimizes latency

Avoids repeated disk I/O

Ensures consistent preprocessing across all requests

**Request schema definition (data validation layer)**

A Pydantic model is defined to represent the expected input payload.

This schema:

Enumerates every required feature explicitly

Enforces strict data types (floats for all numerical inputs)

Rejects malformed or incomplete requests before they reach the model

FastAPI automatically validates incoming JSON against this schema, meaning:

Missing fields → rejected

Wrong data types → rejected

Extra fields → ignored or rejected depending on configuration

This makes the backend robust against invalid client input.

**Prediction endpoint**

A single POST endpoint (typically /predict) is exposed.

This endpoint:

Accepts a JSON payload matching the Pydantic schema

Converts the validated input into a pandas DataFrame

Preserves exact feature order as expected by the trained model

This step is critical: LSTMs are position-sensitive, and feature misalignment would silently corrupt predictions.

**Input preprocessing (model-aligned)**

Once converted into a DataFrame:

The input features are scaled using the preloaded feature scaler

Scaling maps values into the [0, 1] range

The scaled array is reshaped into a 3D tensor:

(1, time_steps, number_of_features)


This tensor shape exactly mirrors what the LSTM saw during training.

No interpolation, imputation, or feature engineering is performed here—the backend assumes inputs are already fully defined and valid.

**Model inference**

The preprocessed tensor is passed to the LSTM model for inference.

The model outputs a scaled AQI value

No batching or parallel inference is used (single-request inference)

The model runs in evaluation mode only

**Postprocessing (output normalization)**

The predicted AQI value is then:

Inversely transformed using the target scaler

Converted from normalized scale back to the original AQI range

Cast into a standard Python numeric type for JSON serialization

This ensures the client receives a human-interpretable AQI value, not a normalized score.

**Response structure**

The endpoint returns a JSON response containing:

The predicted AQI value

A simple key-value structure suitable for frontend or API consumption

The response is intentionally minimal, avoiding unnecessary metadata or verbosity.

**Error handling and stability**

FastAPI’s built-in mechanisms handle most failure modes automatically:

Invalid input → HTTP 422 validation error

Runtime inference issues → HTTP 500 server error

Because:

The model and scalers are preloaded

Input types are strictly validated

No mutable state exists

The backend is deterministic and stable under repeated requests.
