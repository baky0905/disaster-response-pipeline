import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Read messages and categories data.

    Args:
        messages_filepath (str): Filepath for a messages csv file.
        categories_filepath (str): Filepath for a categories csv file.

    Returns:
        df (DataFrame): Pandas DataFrame consisting of merged messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, how="left")

    return df


def clean_data(df):
    """
    Cleans messages and categories data.
    1) Drops duplicate rows.
    2) Splits categories column into separate category columns.
    3) Converts category values to just numbers 0 or 1.
    4) Replaces categories column in df with new category columns and merges with 
    the rest of the dataframe.

    Args:
        df (DataFrame): Merged messages and categories pandas DataFrame.

    Returns:
        df (DataFrame): Cleaned pandas Dataframe.
    """

    df = df.drop_duplicates()

    # Split categories into separate category columns.
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.str.extract(r"([a-zA-Z _]*)")[0].tolist()
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for col in categories.columns:
        categories[col] = categories[col].str.extract(
            r"([0-9].*)").astype("int")

    # Replace categories column in df with new category columns and merge with df.
    df = df.drop(["categories"], axis="columns")
    df = pd.concat([df, categories], axis="columns")

    # Create dummy variables from genre column
    df = pd.concat([df, pd.get_dummies(df.genre)], axis=1)
    df = df.drop(['genre', 'social'], axis=1)

    return df


def save_data(df, database_filename):
    """Saving a pandas dataframe into a sqllite database.

    Args:
        df (DataFrame): Cleaned pandas DataFrame.
        database_filename (str): String of a database where DataFrame will be stored.
    Returns:
        engine (): 
    """
    engine = create_engine('sqlite:///disaster_response.db')
    df.to_sql('messages_categories', engine, index=False, if_exists="replace")

    return engine


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        engine = save_data(df, database_filepath)

        print('Cleaned data saved to database!')

        print('Testing read from the database...')
        try:
            import sqlalchemy as db
            connection = engine.connect()
            test_query = pd.read_sql(
                "SELECT * FROM messages_categories LIMIT 10", con=connection)
            print(test_query)
            print('query works')
            connection.close()
        except:
            print('Cannot read table...')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
