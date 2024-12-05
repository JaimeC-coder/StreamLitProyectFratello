import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sqlalchemy
import plotly.graph_objs as go
import urllib.parse

class ProductOutputPredictor:
    def __init__(self):

        password = '0xVL}]LGr?+M'
        escaped_password = urllib.parse.quote(password)

        # Formando la URL de conexión
        self.engine = sqlalchemy.create_engine(
            f'mysql+pymysql://firetens_prueba:{escaped_password}@firetensor.com:3306/firetens_fratello_bd')

    def load_product_data(self):
        query = """
                SELECT
                    p.productId,
                    p.productName,
                    c.categoryName,
                    YEAR(it.transactionDate) AS transaction_year,
                    SUM(
                        CASE
                            WHEN it.transactionType = 'input' THEN it.transactionCount
                            ELSE 0
                        END
                    ) AS input_transactions,
                    SUM(
                        CASE
                            WHEN it.transactionType = 'output' THEN it.transactionCount
                            ELSE 0
                        END
                    ) AS output_transactions
                FROM
                    products p
                    JOIN categories c ON p.categoryId = c.categoryId
                    JOIN product_unit_price_by_measurements pupm ON pupm.productId = p.productId
                    LEFT JOIN inventory_transactions it ON it.productUnitPriceId = pupm.productUnitPriceId
                GROUP BY
                    p.productId,
                    p.productName,
                    c.categoryName,
                    transaction_year
                HAVING SUM(
                    CASE
                        WHEN it.transactionType = 'input' THEN it.transactionCount
                        ELSE 0
                    END
                ) > 0 OR SUM(
                    CASE
                        WHEN it.transactionType = 'output' THEN it.transactionCount
                        ELSE 0
                    END
                ) > 0
                ORDER BY p.productName, transaction_year;
        """
        return pd.read_sql(query, self.engine)

    def prepare_product_data(self, df, selected_product):
        if selected_product == "":  # Si no se selecciona un producto, devuelve todos los datos.
            return df
        product_df = df[df['productName'] == selected_product]
        return product_df.groupby('transaction_year').agg({
            'output_transactions': 'sum',
            'input_transactions': 'sum'
        }).reset_index()

    def train_product_model(self, df):
        if df.empty:
            raise ValueError("The dataset is empty. Cannot train the model.")
        X = df[['transaction_year']]
        y_output = df['output_transactions']
        model = RandomForestRegressor(n_estimators=50)
        model.fit(X, y_output)
        return model

    def predict_product_output(self, model, historical_data, years_to_predict):
        last_year = historical_data['transaction_year'].max()
        future_years = [last_year + i for i in range(1, years_to_predict + 1)]
        future_X = pd.DataFrame({'transaction_year': future_years})
        predictions = model.predict(future_X)
        future_df = pd.DataFrame({
            'transaction_year': future_years,
            'predicted_output': predictions
        })
        return pd.concat([historical_data, future_df], ignore_index=True)

    def visualize_product_output(self, combined_data, selected_product):
        fig = go.Figure()

        # Historical Output Transactions
        historical_mask = combined_data['predicted_output'].isna()
        fig.add_trace(go.Scatter(
            x=combined_data.loc[historical_mask, 'transaction_year'],
            y=combined_data.loc[historical_mask, 'output_transactions'],
            mode='lines+markers',
            name='Historical Output',
            line=dict(color='blue')
        ))

        # Predicted Output Transactions
        prediction_mask = ~combined_data['predicted_output'].isna()
        fig.add_trace(go.Scatter(
            x=combined_data.loc[prediction_mask, 'transaction_year'],
            y=combined_data.loc[prediction_mask, 'predicted_output'],
            mode='lines+markers',
            name='Predicted Output',
            line=dict(color='red', dash='dot')
        ))

        fig.update_layout(
            title=f'Output Transactions for {selected_product if selected_product else "All Products"}',
            xaxis_title='Year',
            yaxis_title='Output Volume',
            hovermode='x unified'
        )

        return fig

    def run_streamlit_app(self):
        st.title('Product Output Prediction')

        # Load product data
        product_data = self.load_product_data()

        # Sidebar selections
        st.sidebar.header('Prediction Settings')

        # Product selection with only products that have transactions
        available_products = [""] + product_data['productName'].unique().tolist()
        selected_product = st.sidebar.selectbox(
            'Select Product (leave blank for all)',
            options=available_products
        )

        # Years to predict
        last_historical_year = int(product_data['transaction_year'].max())
        available_predict_years = list(range(1, 6))
        years_to_predict = st.sidebar.selectbox(
            'Select Prediction Years',
            options=available_predict_years
        )

        # Prepare data for selected product or all
        product_historical_data = self.prepare_product_data(product_data, selected_product)

        if product_historical_data.empty:
            st.warning(f"No valid data available for {'the selected product' if selected_product else 'any product'}.")
            return

        # Train model and predict
        try:
            model = self.train_product_model(product_historical_data)
            combined_data = self.predict_product_output(
                model,
                product_historical_data,
                years_to_predict
            )
        except ValueError as e:
            st.error(str(e))
            return

        # Visualization
        st.plotly_chart(self.visualize_product_output(combined_data, selected_product))

        # Predictions table
        st.subheader('Detailed Predictions')
        st.dataframe(combined_data)


if __name__ == "__main__":
    app = ProductOutputPredictor()
    app.run_streamlit_app()
