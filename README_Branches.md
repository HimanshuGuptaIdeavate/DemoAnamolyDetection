* M0_XGBOOST – Branch containing the standalone implementation of the XGBoost model.

* Ensemble_XGBOOST_LSTM – Branch implementing the ensemble approach combining XGBoost and LSTM.

* Ensemble_XGBOOST_LSTM_Eval1 – Branch including the ensemble (XGBoost + LSTM) along with extended evaluation improvements:
    -   Expanded features: from 4 raw features to 8 raw + 8 delta features (16 total)
    -   Upgraded architecture: Bidirectional LSTM with Batch Normalization
    -   Hyperparameters tuned specifically for a small dataset (871 samples)