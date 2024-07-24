import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

selection_list = ['User Based Filter', 'Item Based Filter']
select_mode = st.selectbox('Select Filter Model:', selection_list)

if select_mode == 'User Based Filter':

    products = pd.read_csv('Product_Details.csv')
    # customer_item_matrix = pd.read_csv('customer_item_matrix.csv')
    product_options = list(products['Description'].apply(lambda x: str(x)))

    st.title('User Based Filter')
    st.write("Purchase Items")

    purchased_items = st.multiselect('Purchase Items:', product_options)
    submit_button = st.button('Submit') 

    if submit_button == True:
        print(submit_button)
        customer_item_matrix = pd.read_csv('customer_item_matrix.csv')
        product_id_vector = list(customer_item_matrix.columns[1:])
        array = customer_item_matrix.drop('CustomerID', axis = 1)
        array = array.values

        product_ids = products.loc[products['Description'].isin(purchased_items), 'unique_ID'].tolist()
        vector = customer_item_matrix.columns[1:].isin(product_ids)
        cosine_similarities = cosine_similarity(array, vector.reshape(1, -1))
        cosine_similarities = cosine_similarities.flatten().tolist()
        customer_sim = pd.DataFrame({
                            "CustomerID": customer_item_matrix['CustomerID'],
                            "Similarity": cosine_similarities
                                    })
        print(customer_sim)
                                    
        cus_sim_max = customer_sim.loc[np.where(customer_sim['Similarity'] == customer_sim['Similarity'].max())[0][0], 'CustomerID']
        cus_sim_max_2 = customer_sim.loc[customer_sim['Similarity'].nlargest(2).index[-1], 'CustomerID']

        items_bought_by_sim_cus = set(customer_item_matrix.loc[customer_item_matrix['CustomerID'] == cus_sim_max][customer_item_matrix.loc[customer_item_matrix['CustomerID'] == cus_sim_max]>0])
        items_bought_by_sim_cus_2 = set(customer_item_matrix.loc[customer_item_matrix['CustomerID'] == cus_sim_max_2][customer_item_matrix.loc[customer_item_matrix['CustomerID'] == cus_sim_max_2]>0])
        items_intersection = items_bought_by_sim_cus.union(items_bought_by_sim_cus_2)
        
        items_to_recommend_to_user = set(product_ids) - set(items_bought_by_sim_cus)
        st.write("Items to Recommend to User")

        product_recommendation= products.loc[products['unique_ID'].isin(items_to_recommend_to_user),['unique_ID', 'Description']].drop_duplicates().set_index('unique_ID')
        st.dataframe(product_recommendation)


if select_mode == 'Item Based Filter':
    products_ = pd.read_csv('Product_Details.csv')
    item_item_sim_matrix_ = pd.read_csv('Item_Recommendation.csv')

    product_options_ = products_.loc[products_['unique_ID'].isin(item_item_sim_matrix_['StockCode'].astype(str)), 'Description'].tolist()

    st.title('Item Based Filter')
    st.write("Items Available")

    purchased_item = st.selectbox('Select Items:', product_options_)
    submit_button = st.button('Submit')

    # product_options = list(products['Description'].apply(lambda x: str(x)))
    #  item_item_sim_matrix = pd.read_csv('Item_Recommendation.csv', index_col= 'StockCode')

    if submit_button:
        # st.write(purchased_item)

        product_id_selected = products_.loc[products_['Description'] == purchased_item, 'unique_ID'].tolist()
        # st.write(product_id_selected[0])
        top_10_similar_items = item_item_sim_matrix_.loc[item_item_sim_matrix_['StockCode'].astype(str) == product_id_selected[0]].reset_index()
        top_10_similar_items = top_10_similar_items.T.sort_values(by = 0, ascending = False).iloc[3:13].index.tolist()
        # st.write(top_10_similar_items)
        # top_10_similar_items = top_10_similar_items.iloc[1:11][['index']].values.flatten().tolist()
        # st.dataframe(top_10_similar_items)
        # product_recommendation= products.loc[products['unique_ID'].isin(items_to_recommend_to_user),['unique_ID', 'Description']].drop_duplicates().set_index('unique_ID')
        st.dataframe(products_.loc[products_['unique_ID'].isin(top_10_similar_items), ['unique_ID', 'Description']].drop_duplicates().set_index('unique_ID'))

    