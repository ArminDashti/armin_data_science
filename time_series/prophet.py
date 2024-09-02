import pandas as pd
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics
from prophet import Prophet



class Prophet:
    def __init__(self, df):
        self.model = fit(df)
        
    
    def fit(self):
        model = Prophet()
        model.fit(df)
        return model
        
        
    def predict(self, period=200):
        future = model.make_future_dataframe(periods=600)  # Predicting 365 days into the future
        self.forecast = model.predict(future)
        return self.forecast
    
    
    def plots(self):
        fig1 = model.plot(self.forecast)
        fig2 = model.plot_components(self.forecast)
        return fig1, fig2
        
        
    def cross_validation(self):
        df_cv = cross_validation(self.model, initial='730 days', period='180 days', horizon='365 days')
        return df_cv

